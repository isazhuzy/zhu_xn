"""step1_startscan_ddb.py — ENTRY-SECOND SCAN of the prior-minute momentum trade.
INDEPENDENT experiment (not a continuation): based on fig26, test whether the
momentum/reversal full-minute edge depends on WHERE in the minute we enter.

Trade: signal d = sign(close_{M-1} - close_{M-2}); enter at elapsed second S of
minute M, EXIT at elapsed second S of minute M+1 (a full one-minute hold, just
shifted by S). pnl_S = d * (mid_asof_{M+1}(S) - mid_asof_M(S)).
  >0 = momentum/continuation pays ; <0 = reversal pays  (same sign convention as fig26).
Scan S in {0,10,20,30,40,50} seconds (50 ~= 10s before the next open; S=0 == open).

Output per (code, year, month, S): sum, sumsq, n  ->  later: mean & t = mean/(sd/sqrt n).
Run:  /Users/zhuisabella/xn/.venv/bin/python step1_startscan_ddb.py   (sandbox OFF)
"""
import os
import calendar
import numpy as np
import pandas as pd
import dolphindb as ddb

from ddb_config import HOST, PORT, USER, PW
CODES = ["IC0000", "IF0000", "IH0000", "IM0000"]
MIN_TICKS = 10                       # threshold: only minutes with >=10 ticks
GRID = [0, 10, 20, 30, 40, 50]       # entry second within the minute
SMALL = os.environ.get("SMALL", "1") == "1"    # SMALL=0 -> full 76-month window
WINDOWS = ([(2024, 4), (2024, 5), (2024, 6)] if SMALL else
           [(y, m) for y in range(2020, 2027) for m in range(1, 13)
            if (2020, 1) <= (y, m) <= (2026, 5)])
OUTCSV = "/Users/zhuisabella/xn/intraminute/step1_startscan.csv"


def fetch(sess, code, start, end):
    q = f"""
    pt=loadTable("dfs://hft_future_ts","TickPartitioned")
    select m_nDatetime as ts, (m_nBidPrice+m_nAskPrice)/2.0 as mid
    from pt where code_init=`{code[:2]}, m_nDatetime>={start}T00:00:00,
          m_nDatetime<={end}T23:59:59, code=`{code},
          minute(m_nDatetime) between 09:30m:15:00m
    """
    return sess.run(q)


def scan(df):
    df = df.drop_duplicates("ts").sort_values("ts").copy()
    df["mkey"] = df["ts"].dt.floor("min")
    tod = df["mkey"].dt.hour * 60 + df["mkey"].dt.minute
    df = df[((tod >= 570) & (tod <= 690)) | ((tod >= 780) & (tod <= 900))]
    if df.empty:
        return None
    df["sec"] = (df["ts"] - df["mkey"]).dt.total_seconds()   # 0 .. 59.5 within minute

    # prior-minute momentum signal d, computed on per-minute last (close) prices
    per = pd.DataFrame({"last": df.groupby("mkey")["mid"].last()})
    per["tod"] = per.index.hour * 60 + per.index.minute
    per["day"] = per.index.normalize()
    per["session"] = np.where(per["tod"] <= 690, "AM", "PM")
    per = per.sort_values(["day", "session", "tod"])
    g = per.groupby(["day", "session"])
    c1, c2 = g["last"].shift(1), g["last"].shift(2)
    t1, t2 = g["tod"].shift(1), g["tod"].shift(2)
    per["d"] = np.where((per["tod"] - t1 == 1) & (t1 - t2 == 1), np.sign(c1 - c2), np.nan)
    dmap = per["d"].to_dict()
    todmap = per["tod"].to_dict()
    daymap = per["day"].to_dict()
    sesmap = per["session"].to_dict()

    # mid as-of each grid second (last tick with sec<=S) for every minute with >=MIN_TICKS
    asof = {}
    for mkey, grp in df.groupby("mkey"):
        secs = grp["sec"].to_numpy()
        mids = grp["mid"].to_numpy()
        if len(mids) < MIN_TICKS:
            continue
        vals = np.empty(len(GRID))
        for j, S in enumerate(GRID):
            idx = np.searchsorted(secs, S, side="right") - 1   # last tick with sec<=S
            vals[j] = mids[max(idx, 0)]
        asof[mkey] = vals

    acc = {S: [0.0, 0.0, 0] for S in GRID}     # sum, sumsq, n
    for mkey, vcur in asof.items():
        d = dmap.get(mkey, np.nan)
        if not np.isfinite(d) or d == 0:
            continue
        # next consecutive minute, same day+session
        nxt = mkey + pd.Timedelta(minutes=1)
        if nxt not in asof:
            continue
        if daymap.get(nxt) != daymap.get(mkey) or sesmap.get(nxt) != sesmap.get(mkey):
            continue
        if todmap.get(nxt, -99) - todmap.get(mkey, 0) != 1:
            continue
        vnxt = asof[nxt]
        for j, S in enumerate(GRID):
            p = d * (vnxt[j] - vcur[j])
            acc[S][0] += p; acc[S][1] += p * p; acc[S][2] += 1
    return acc


if __name__ == "__main__":
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    rows = []
    for yr, mo in WINDOWS:
        last = calendar.monthrange(yr, mo)[1]
        start, end = f"{yr}.{mo:02d}.01", f"{yr}.{mo:02d}.{last:02d}"
        for code in CODES:
            df = fetch(sess, code, start, end)
            if not len(df):
                print(f"  {yr}-{mo:02d} {code}: no data", flush=True); continue
            acc = scan(df)
            if acc is None:
                continue
            for S, (s, ss, n) in acc.items():
                rows.append({"code": code, "year": yr, "month": mo, "S": S,
                             "sum": s, "sumsq": ss, "n": n})
        print(f"{yr}-{mo:02d} done", flush=True)
    sess.close()
    pd.DataFrame(rows).to_csv(OUTCSV, index=False)
    print(f"saved {OUTCSV}")
