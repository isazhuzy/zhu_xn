"""focus_1121_ddb.py — deep-dive on the 11:21->11:30 hold (into lunch close), all 76 months.
For each day/contract pull the minute closes needed for entry signals + the block entry/exit:
 c1115,c1119,c1120 (signals), entry=open(11:21), exit=close(11:29 ~lunch close).
Lets us test held-block strategies (1 trade) in pandas. Output per (code,ym,date).
Run: /Users/zhuisabella/xn/.venv/bin/python focus_1121_ddb.py   (sandbox off)
"""
import calendar
import numpy as np
import pandas as pd
import dolphindb as ddb

from ddb_config import HOST, PORT, USER, PW
CODES = ["IC0000", "IF0000", "IH0000", "IM0000"]
WINDOWS = [(y, m) for y in range(2020, 2027) for m in range(1, 13)
           if (2020, 1) <= (y, m) <= (2026, 5)]
OUTCSV = "/Users/zhuisabella/xn/intraminute/focus_1121.csv"


def fetch(sess, code, start, end):
    q = f"""
    pt=loadTable("dfs://hft_future_ts","TickPartitioned")
    select m_nDatetime as ts, (m_nBidPrice+m_nAskPrice)/2.0 as mid
    from pt where code_init=`{code[:2]}, m_nDatetime>={start}T00:00:00,
          m_nDatetime<={end}T23:59:59, code=`{code},
          minute(m_nDatetime) between 11:10m:11:30m
    """
    return sess.run(q)


def build(df):
    df = df.drop_duplicates("ts").sort_values("ts").copy()
    df["mkey"] = df["ts"].dt.floor("min")
    df["tod"] = df["mkey"].dt.hour * 60 + df["mkey"].dt.minute
    df["date"] = df["mkey"].dt.normalize()
    bar = df.groupby(["date", "tod"])["mid"].agg(open="first", close="last").reset_index()
    cl = bar.pivot(index="date", columns="tod", values="close")
    op = bar.pivot(index="date", columns="tod", values="open")
    out = pd.DataFrame(index=cl.index)
    for c, t in [("c1115", 675), ("c1119", 679), ("c1120", 680), ("c1129", 689)]:
        out[c] = cl[t] if t in cl.columns else np.nan
    out["o1121"] = op[681] if 681 in op.columns else np.nan
    return out.dropna().reset_index()


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
            o = build(df)
            if not len(o):
                continue
            o["code"] = code; o["ym"] = f"{yr}-{mo:02d}"
            rows.append(o); got.append(code)
        print(f"{yr}-{mo:02d}: {len(got)} codes", flush=True)
    sess.close()
    pd.concat(rows, ignore_index=True).to_csv(OUTCSV, index=False)
    print(f"saved {OUTCSV}")
