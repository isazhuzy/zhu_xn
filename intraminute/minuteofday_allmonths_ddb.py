"""minuteofday_allmonths_ddb.py — per-minute-of-day momentum P&L, all 76 months.
Like window_allmonths but at 1-minute resolution (group by tod, not 15-min) so we can
pinpoint precise sub-segments. Output: code, year, month, tod, mean, n.
Run: /Users/zhuisabella/xn/.venv/bin/python minuteofday_allmonths_ddb.py   (sandbox off)
"""
import calendar
import numpy as np
import pandas as pd
import dolphindb as ddb

from ddb_config import HOST, PORT, USER, PW
CODES = ["IC0000", "IF0000", "IH0000", "IM0000"]
WINDOWS = [(y, m) for y in range(2020, 2027) for m in range(1, 13)
           if (2020, 1) <= (y, m) <= (2026, 5)]
OUTCSV = "/Users/zhuisabella/xn/intraminute/minuteofday_allmonths.csv"


def fetch(sess, code, start, end):
    q = f"""
    pt=loadTable("dfs://hft_future_ts","TickPartitioned")
    select m_nDatetime as ts, (m_nBidPrice+m_nAskPrice)/2.0 as mid
    from pt where code_init=`{code[:2]}, m_nDatetime>={start}T00:00:00,
          m_nDatetime<={end}T23:59:59, code=`{code},
          minute(m_nDatetime) between 09:30m:15:00m
    """
    return sess.run(q)


def mod_pnl(df):
    df = df.drop_duplicates("ts").sort_values("ts").copy()
    df["mkey"] = df["ts"].dt.floor("min")
    per = df.groupby("mkey")["mid"].agg(open="first", close="last", nt="count")
    per["tod"] = per.index.hour * 60 + per.index.minute
    per = per[((per.tod >= 570) & (per.tod <= 690)) | ((per.tod >= 780) & (per.tod <= 900))]
    per["date"] = per.index.normalize()
    per["session"] = np.where(per["tod"] <= 690, "AM", "PM")
    per = per.sort_values(["date", "session", "tod"])
    g = per.groupby(["date", "session"])
    c1, c2 = g["close"].shift(1), g["close"].shift(2)
    t1, t2 = g["tod"].shift(1), g["tod"].shift(2)
    per["d"] = np.where((per["tod"] - t1 == 1) & (t1 - t2 == 1), np.sign(c1 - c2), np.nan)
    per["pnl"] = per["d"] * (per["close"] - per["open"])
    per = per[np.isfinite(per["pnl"]) & (per["d"] != 0) & (per["nt"] >= 2)]
    return per.groupby("tod")["pnl"].agg(mean="mean", n="count").reset_index()


if __name__ == "__main__":
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    rows = []
    for yr, mo in WINDOWS:
        last = calendar.monthrange(yr, mo)[1]
        start, end = f"{yr}.{mo:02d}.01", f"{yr}.{mo:02d}.{last:02d}"
        got = []
        for code in CODES:
            df = fetch(sess, code, start, end)
            if not len(df):
                continue
            o = mod_pnl(df); o["code"] = code; o["year"] = yr; o["month"] = mo
            rows.append(o); got.append(code)
        print(f"{yr}-{mo:02d}: {len(got)} codes", flush=True)
    sess.close()
    pd.concat(rows, ignore_index=True).to_csv(OUTCSV, index=False)
    print(f"saved {OUTCSV}")
