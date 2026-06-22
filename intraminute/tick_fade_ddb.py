"""tick_fade_ddb.py — TICK-LEVEL fade backtest, all 76 months.
Each minute: fade the prior-minute direction (enter at open), EXIT at a fixed within-minute tick K
(instead of the minute close). Sweep K in {20,42,60,90,close} to find the best exit.
fade_pnl_K(minute) = -d * (price[K] - price[open]),  d = sign(prev-minute move).
Output per (code, year, month): for each exit, sum of fade P&L and count -> mean & equity later.
Run with venv python (sandbox off)."""
import calendar
import numpy as np
import pandas as pd
import dolphindb as ddb

from ddb_config import HOST, PORT, USER, PW
CODES = ["IC0000", "IF0000", "IH0000", "IM0000"]
WINDOWS = [(y, m) for y in range(2020, 2027) for m in range(1, 13)
           if (2020, 1) <= (y, m) <= (2026, 5)]
EXITS = [20, 42, 60, 90]           # exit-tick choices (plus 'close')
OUTCSV = "/Users/zhuisabella/xn/intraminute/tick_fade.csv"


def fetch(sess, code, start, end):
    q = f"""
    pt=loadTable("dfs://hft_future_ts","TickPartitioned")
    select m_nDatetime as ts, (m_nBidPrice+m_nAskPrice)/2.0 as mid
    from pt where code_init=`{code[:2]}, m_nDatetime>={start}T00:00:00,
          m_nDatetime<={end}T23:59:59, code=`{code},
          minute(m_nDatetime) between 09:30m:15:00m
    """
    return sess.run(q)


def backtest(df):
    df = df.drop_duplicates("ts").sort_values("ts").copy()
    df["mkey"] = df["ts"].dt.floor("min")
    tod = df["mkey"].dt.hour * 60 + df["mkey"].dt.minute
    df = df[((tod >= 570) & (tod <= 690)) | ((tod >= 780) & (tod <= 900))]
    if df.empty:
        return None
    per = pd.DataFrame({"last": df.groupby("mkey")["mid"].last()})
    per["day"] = per.index.normalize()
    per["tod"] = per.index.hour * 60 + per.index.minute
    per["session"] = np.where(per["tod"] <= 690, "AM", "PM")
    per = per.sort_values(["day", "session", "tod"])
    g = per.groupby(["day", "session"])
    c1, c2 = g["last"].shift(1), g["last"].shift(2)
    t1, t2 = g["tod"].shift(1), g["tod"].shift(2)
    per["d"] = np.where((per["tod"] - t1 == 1) & (t1 - t2 == 1), np.sign(c1 - c2), np.nan)
    dmap = per["d"].to_dict()

    acc = {f"K{k}": [0.0, 0] for k in EXITS}; acc["close"] = [0.0, 0]
    for mkey, grp in df.groupby("mkey"):
        d = dmap.get(mkey, np.nan)
        if not np.isfinite(d) or d == 0:
            continue
        path = grp["mid"].to_numpy(); L = len(path)
        if L < 2:
            continue
        op = path[0]
        for k in EXITS:
            if L >= k:                                  # minute reached tick k
                acc[f"K{k}"][0] += -d * (path[k - 1] - op); acc[f"K{k}"][1] += 1
        acc["close"][0] += -d * (path[-1] - op); acc["close"][1] += 1
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
                continue
            acc = backtest(df)
            if acc is None:
                continue
            row = {"code": code, "year": yr, "month": mo}
            for kk, (s, n) in acc.items():
                row[f"sum_{kk}"] = s; row[f"n_{kk}"] = n
            rows.append(row)
        print(f"{yr}-{mo:02d} done", flush=True)
    sess.close()
    pd.DataFrame(rows).to_csv(OUTCSV, index=False)
    print(f"saved {OUTCSV}")
