"""minute_conditional_ddb.py — per-minute records for conditional analysis (IS + OOS).
For each valid minute M: prior-minute move dprev = close(M-1)-close(M-2), within-minute
return ret = close(M)-open(M). Momentum P&L = sign(dprev)*ret. Keeps RAW per-minute rows
so conditioning (by |dprev|, etc.) can be sliced in pandas without re-fetching.
Output: minute_conditional_isoos.csv (code, ym, sample, date, tod, dprev, ret, nticks).
Run: /Users/zhuisabella/xn/.venv/bin/python minute_conditional_ddb.py   (sandbox off)
"""
import calendar
import numpy as np
import pandas as pd
import dolphindb as ddb

from ddb_config import HOST, PORT, USER, PW
CODES = ["IC0000", "IF0000", "IH0000", "IM0000"]
IS_MONTHS = [(2023, 5), (2023, 6), (2023, 9), (2024, 6), (2024, 7), (2025, 6)]
OOS_MONTHS = [(2023, 10), (2024, 8), (2024, 11), (2025, 5), (2025, 8), (2025, 10)]
OUTCSV = "/Users/zhuisabella/xn/intraminute/minute_conditional_isoos.csv"


def fetch(sess, code, start, end):
    q = f"""
    pt=loadTable("dfs://hft_future_ts","TickPartitioned")
    select m_nDatetime as ts, (m_nBidPrice+m_nAskPrice)/2.0 as mid
    from pt where code_init=`{code[:2]}, m_nDatetime>={start}T00:00:00,
          m_nDatetime<={end}T23:59:59, code=`{code},
          minute(m_nDatetime) between 09:30m:15:00m
    """
    return sess.run(q)


def per_minute(df):
    df = df.drop_duplicates("ts").sort_values("ts").copy()
    df["mkey"] = df["ts"].dt.floor("min")
    per = df.groupby("mkey")["mid"].agg(open="first", close="last", nticks="count")
    per["tod"] = per.index.hour * 60 + per.index.minute
    per = per[((per.tod >= 570) & (per.tod <= 690)) | ((per.tod >= 780) & (per.tod <= 900))]
    per["date"] = per.index.normalize()
    per["session"] = np.where(per["tod"] <= 690, "AM", "PM")
    per = per.sort_values(["date", "session", "tod"])
    g = per.groupby(["date", "session"])
    c1, c2 = g["close"].shift(1), g["close"].shift(2)
    t1, t2 = g["tod"].shift(1), g["tod"].shift(2)
    per["dprev"] = np.where((per["tod"] - t1 == 1) & (t1 - t2 == 1), c1 - c2, np.nan)
    per["ret"] = per["close"] - per["open"]
    per = per[np.isfinite(per["dprev"]) & (per["dprev"] != 0) & (per["nticks"] >= 2)]
    return per[["date", "tod", "dprev", "ret", "nticks"]].reset_index(drop=True)


if __name__ == "__main__":
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    rows = []
    for sample, months in [("IS", IS_MONTHS), ("OOS", OOS_MONTHS)]:
        for yr, mo in months:
            last = calendar.monthrange(yr, mo)[1]
            start, end = f"{yr}.{mo:02d}.01", f"{yr}.{mo:02d}.{last:02d}"
            got = []
            for code in CODES:
                df = fetch(sess, code, start, end)
                if not len(df):
                    continue
                o = per_minute(df)
                o["code"] = code; o["ym"] = f"{yr}-{mo:02d}"; o["sample"] = sample
                rows.append(o); got.append(f"{code}({len(o)})")
            print(f"[{sample}] {yr}-{mo:02d}: {', '.join(got)}", flush=True)
    sess.close()
    pd.concat(rows, ignore_index=True).to_csv(OUTCSV, index=False)
    print(f"saved {OUTCSV}")
