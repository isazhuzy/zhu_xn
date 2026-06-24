"""threshold_scan_ddb.py — does raising the per-minute tick threshold (L>=T) raise return?
For every valid minute M (consecutive M+1, valid momentum sign d), record the minute's
tick-count L and two P&Ls, then bin by L so we can read mean return at any threshold T.

 step1 (S=0 full-minute momentum):  pnl1 = d * (open_{M+1} - open_M)
 step3 (V-trade defined in SECONDS, L-independent): short -d from 0->flip_s, long +d
        flip_s->exit_s, using price as-of each second:
        pnl3 = d * ( p(exit_s) - 2*p(flip_s) + p(0) )

Output threshold_scan.csv: code, Lbucket, s1_sum,s1_ss,s1_n, s3_sum,s3_ss,s3_n
Run: /Users/zhuisabella/xn/.venv/bin/python threshold_scan_ddb.py   (sandbox OFF)
"""
import calendar
import numpy as np
import pandas as pd
import dolphindb as ddb
from ddb_config import HOST, PORT, USER, PW

CODES = ["IC0000", "IF0000", "IH0000", "IM0000"]
# per-contract V-trade flip/exit in SECONDS (from fig51 tick optima, /2)
SEC = {"IC0000": (21, 55), "IF0000": (21, 55), "IH0000": (20, 47), "IM0000": (45, 60)}
WINDOWS = [(y, m) for y in range(2020, 2027) for m in range(1, 13)
           if (2020, 1) <= (y, m) <= (2026, 5)]
OUTCSV = "/Users/zhuisabella/xn/intraminute/threshold_scan.csv"


def fetch(sess, code, start, end):
    q = f"""
    pt=loadTable("dfs://hft_future_ts","TickPartitioned")
    select m_nDatetime as ts, (m_nBidPrice+m_nAskPrice)/2.0 as mid
    from pt where code_init=`{code[:2]}, m_nDatetime>={start}T00:00:00,
          m_nDatetime<={end}T23:59:59, code=`{code},
          minute(m_nDatetime) between 09:30m:15:00m
    """
    return sess.run(q)


def compute(df, flip_s, exit_s):
    df = df.drop_duplicates("ts").sort_values("ts").copy()
    df["mkey"] = df["ts"].dt.floor("min")
    tod = df["mkey"].dt.hour * 60 + df["mkey"].dt.minute
    df = df[((tod >= 570) & (tod <= 690)) | ((tod >= 780) & (tod <= 900))]
    if df.empty:
        return None
    df["sec"] = (df["ts"] - df["mkey"]).dt.total_seconds()

    per = pd.DataFrame({"last": df.groupby("mkey")["mid"].last()})
    per["tod"] = per.index.hour * 60 + per.index.minute
    per["day"] = per.index.normalize()
    per["session"] = np.where(per["tod"] <= 690, "AM", "PM")
    per = per.sort_values(["day", "session", "tod"])
    g = per.groupby(["day", "session"])
    c1, c2 = g["last"].shift(1), g["last"].shift(2)
    t1, t2 = g["tod"].shift(1), g["tod"].shift(2)
    per["d"] = np.where((per["tod"] - t1 == 1) & (t1 - t2 == 1), np.sign(c1 - c2), np.nan)
    dmap = per["d"].to_dict(); todmap = per["tod"].to_dict()
    daymap = per["day"].to_dict(); sesmap = per["session"].to_dict()

    # per-minute: tick count L, open, and as-of prices at 0/flip/exit seconds
    info = {}
    for mkey, grp in df.groupby("mkey"):
        secs = grp["sec"].to_numpy(); mids = grp["mid"].to_numpy()
        L = len(mids)
        def asof(S):
            i = np.searchsorted(secs, S, side="right") - 1
            return mids[max(i, 0)]
        info[mkey] = (L, mids[0], asof(flip_s), asof(exit_s))

    rows = []   # (L, pnl1, pnl3)
    for mkey, (L, op, pf, pe) in info.items():
        d = dmap.get(mkey, np.nan)
        if not np.isfinite(d) or d == 0 or L < 10:
            continue
        nxt = mkey + pd.Timedelta(minutes=1)
        if nxt not in info or daymap.get(nxt) != daymap.get(mkey) \
                or sesmap.get(nxt) != sesmap.get(mkey) \
                or todmap.get(nxt, -99) - todmap.get(mkey, 0) != 1:
            continue
        nxt_open = info[nxt][1]
        pnl1 = d * (nxt_open - op)
        pnl3 = d * (pe - 2 * pf + op)
        rows.append((L, pnl1, pnl3))
    return rows


if __name__ == "__main__":
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    acc = {}   # (code, Lbucket) -> [s1,ss1,n1, s3,ss3,n3]
    for yr, mo in WINDOWS:
        last = calendar.monthrange(yr, mo)[1]
        start, end = f"{yr}.{mo:02d}.01", f"{yr}.{mo:02d}.{last:02d}"
        for code in CODES:
            df = fetch(sess, code, start, end)
            if not len(df):
                continue
            rows = compute(df, *SEC[code])
            if not rows:
                continue
            for L, p1, p3 in rows:
                Lb = min(L // 10 * 10, 130)     # 10,20,...,120,130(=130+)
                k = (code, Lb)
                a = acc.setdefault(k, [0.0, 0.0, 0, 0.0, 0.0, 0])
                a[0] += p1; a[1] += p1 * p1; a[2] += 1
                a[3] += p3; a[4] += p3 * p3; a[5] += 1
        print(f"{yr}-{mo:02d} done", flush=True)
    sess.close()
    rows = [{"code": c, "Lbucket": Lb, "s1_sum": a[0], "s1_ss": a[1], "s1_n": a[2],
             "s3_sum": a[3], "s3_ss": a[4], "s3_n": a[5]} for (c, Lb), a in acc.items()]
    pd.DataFrame(rows).sort_values(["code", "Lbucket"]).to_csv(OUTCSV, index=False)
    print(f"saved {OUTCSV}")
