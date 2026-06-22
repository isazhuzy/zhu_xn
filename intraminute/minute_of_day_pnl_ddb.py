"""minute_of_day_pnl_ddb.py — momentum P&L per MINUTE-OF-DAY (one month).
For each session minute M and each day:  d = sign(close(M-1)-close(M-2)),
                                          pnl = d * (close(M) - open(M))    [index points]
Then aggregate per minute-of-day across the month's days: mean, std, n, t-stat, hit-rate.
Goal: spot specific minutes (e.g. session open) where momentum actually pays, vs the ~0 average.
Run:  /Users/zhuisabella/xn/.venv/bin/python minute_of_day_pnl_ddb.py   (sandbox off)
"""
import calendar
import numpy as np
import pandas as pd
import dolphindb as ddb

from ddb_config import HOST, PORT, USER, PW
CODES = ["IC0000", "IF0000", "IH0000", "IM0000"]
YEAR, MONTH = 2023, 5                       # calm/normal month to start small
OUTCSV = f"/Users/zhuisabella/xn/intraminute/minute_of_day_pnl_{YEAR}_{MONTH:02d}.csv"


def fetch(sess, code, start, end):
    q = f"""
    pt=loadTable("dfs://hft_future_ts","TickPartitioned")
    select m_nDatetime as ts, (m_nBidPrice+m_nAskPrice)/2.0 as mid
    from pt where code_init=`{code[:2]}, m_nDatetime>={start}T00:00:00,
          m_nDatetime<={end}T23:59:59, code=`{code},
          minute(m_nDatetime) between 09:30m:15:00m
    """
    return sess.run(q)


def pnl_by_minute(df):
    df = df.drop_duplicates("ts").sort_values("ts").copy()
    df["mkey"] = df["ts"].dt.floor("min")
    # per (day, minute): open = first mid, close = last mid
    per = df.groupby("mkey")["mid"].agg(open="first", close="last")
    per["tod"] = per.index.hour * 60 + per.index.minute
    per = per[((per.tod >= 570) & (per.tod <= 690)) | ((per.tod >= 780) & (per.tod <= 900))]
    per["day"] = per.index.normalize()
    per["session"] = np.where(per["tod"] <= 690, "AM", "PM")
    per = per.sort_values(["day", "session", "tod"])
    g = per.groupby(["day", "session"])
    c1, c2 = g["close"].shift(1), g["close"].shift(2)
    t1, t2 = g["tod"].shift(1), g["tod"].shift(2)
    per["d"] = np.where((per["tod"] - t1 == 1) & (t1 - t2 == 1), np.sign(c1 - c2), np.nan)
    per["pnl"] = per["d"] * (per["close"] - per["open"])
    per = per[np.isfinite(per["pnl"]) & (per["d"] != 0)]
    out = per.groupby("tod")["pnl"].agg(n="count", mean="mean", std="std")
    out["t"] = out["mean"] / (out["std"] / np.sqrt(out["n"]))
    out["hit"] = per.groupby("tod")["pnl"].apply(lambda x: (x > 0).mean())
    out["minute"] = [f"{int(t)//60:02d}:{int(t)%60:02d}" for t in out.index]
    return out.reset_index()


if __name__ == "__main__":
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    last = calendar.monthrange(YEAR, MONTH)[1]
    start, end = f"{YEAR}.{MONTH:02d}.01", f"{YEAR}.{MONTH:02d}.{last:02d}"
    rows = []
    for code in CODES:
        df = fetch(sess, code, start, end)
        if not len(df):
            print(f"{code}: no data"); continue
        o = pnl_by_minute(df)
        o["code"] = code
        rows.append(o)
        days = df["ts"].dt.normalize().nunique()
        top = o.reindex(o["mean"].abs().sort_values(ascending=False).index).head(5)
        print(f"\n{code}  ({days} days)  top-5 minutes by |avg pnl|:")
        for _, r in top.iterrows():
            print(f"   {r.minute}  mean={r['mean']:+.3f}pt  t={r.t:+.2f}  hit={r.hit:.0%}  n={int(r.n)}")
    sess.close()
    pd.concat(rows, ignore_index=True).to_csv(OUTCSV, index=False)
    print(f"\nsaved {OUTCSV}")
