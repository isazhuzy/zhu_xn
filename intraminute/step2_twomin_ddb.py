"""step2_twomin_ddb.py — fig26 analog on 2-MINUTE BARS.
INDEPENDENT experiment (not a continuation): does the within-window reversal->recovery
shape still exist if we trade every 2 minutes instead of every minute?

2-min bars built per (day, session) by pairing consecutive minutes (offset//2).
Signal d = sign(close_{bar k-1} - close_{bar k-2}); curve = mean over bars of
d * (path - path[0]) aligned by within-bar tick index (1..~240). Same momentum
sign convention as fig26.

Output per (code, year, month, tick): sum, n  ->  mean = sum/n.
Run:  /Users/zhuisabella/xn/.venv/bin/python step2_twomin_ddb.py   (sandbox OFF)
"""
import os
import calendar
import numpy as np
import pandas as pd
import dolphindb as ddb

from ddb_config import HOST, PORT, USER, PW
CODES = ["IC0000", "IF0000", "IH0000", "IM0000"]
MIN_TICKS = int(os.environ.get("MIN_TICKS", "10"))   # only 2-min bars with >=MIN_TICKS ticks
SMALL = os.environ.get("SMALL", "1") == "1"    # SMALL=0 -> full 76-month window
WINDOWS = ([(2024, 4), (2024, 5), (2024, 6)] if SMALL else
           [(y, m) for y in range(2020, 2027) for m in range(1, 13)
            if (2020, 1) <= (y, m) <= (2026, 5)])
OUTCSV = ("/Users/zhuisabella/xn/intraminute/step2_twomin.csv" if MIN_TICKS == 10 else
          f"/Users/zhuisabella/xn/intraminute/step2_twomin_t{MIN_TICKS}.csv")


def fetch(sess, code, start, end):
    q = f"""
    pt=loadTable("dfs://hft_future_ts","TickPartitioned")
    select m_nDatetime as ts, (m_nBidPrice+m_nAskPrice)/2.0 as mid
    from pt where code_init=`{code[:2]}, m_nDatetime>={start}T00:00:00,
          m_nDatetime<={end}T23:59:59, code=`{code},
          minute(m_nDatetime) between 09:30m:15:00m
    """
    return sess.run(q)


def curve(df):
    df = df.drop_duplicates("ts").sort_values("ts").copy()
    df["mkey"] = df["ts"].dt.floor("min")
    tod = df["mkey"].dt.hour * 60 + df["mkey"].dt.minute
    am = (tod >= 570) & (tod <= 690)
    pm = (tod >= 780) & (tod <= 900)
    df = df[am | pm]
    if df.empty:
        return None
    tod = df["mkey"].dt.hour * 60 + df["mkey"].dt.minute
    df["day"] = df["mkey"].dt.normalize()
    df["session"] = np.where(tod <= 690, "AM", "PM")
    soff = np.where(tod <= 690, tod - 570, tod - 780)        # 0..120 within session
    df["bar"] = soff // 2                                    # 2-min bar index in session
    # drop the lone closing-minute partial bar (offset 120 -> bar 60)
    df = df[soff < 120]
    if df.empty:
        return None

    # per-bar close + prior-bar momentum signal
    key = ["day", "session", "bar"]
    bclose = df.sort_values("ts").groupby(key)["mid"].last().reset_index(name="last")
    bclose = bclose.sort_values(key)
    gb = bclose.groupby(["day", "session"])
    c1, c2 = gb["last"].shift(1), gb["last"].shift(2)
    b1, b2 = gb["bar"].shift(1), gb["bar"].shift(2)
    bclose["d"] = np.where((bclose["bar"] - b1 == 1) & (b1 - b2 == 1),
                           np.sign(c1 - c2), np.nan)
    dmap = {(r.day, r.session, r.bar): r.d for r in bclose.itertuples()}

    vecs = []
    for (day, ses, bar), grp in df.groupby(key):
        d = dmap.get((day, ses, bar), np.nan)
        if not np.isfinite(d) or d == 0:
            continue
        path = grp.sort_values("ts")["mid"].to_numpy()
        if len(path) < MIN_TICKS:
            continue
        vecs.append(d * (path - path[0]))
    if not vecs:
        return None
    L = max(len(v) for v in vecs)
    M = np.full((len(vecs), L), np.nan)
    for i, v in enumerate(vecs):
        M[i, :len(v)] = v
    return np.nanmean(M, axis=0), np.sum(~np.isnan(M), axis=0), len(vecs)


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
            res = curve(df)
            if res is None:
                continue
            mean, n, nsamp = res
            print(f"  {yr}-{mo:02d} {code}: {nsamp} bar-samples, len {len(mean)}", flush=True)
            for i, (mv, nv) in enumerate(zip(mean, n), start=1):
                rows.append({"code": code, "year": yr, "month": mo,
                             "tick": i, "sum": mv * int(nv), "n": int(nv)})
        print(f"{yr}-{mo:02d} done", flush=True)
    sess.close()
    pd.DataFrame(rows).to_csv(OUTCSV, index=False)
    print(f"saved {OUTCSV}")
