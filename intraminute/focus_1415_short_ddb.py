"""focus_1415_short_ddb.py — pure SHORT 14:15->14:30 (one trade/day), all 76 months.
Tests an unconditional directional bet (just short the block), independent of any signal.
Pull minute bars 14:14-14:30 per day; compute block short P&L and per-minute short P&L.
Output per (code, ym, date): o1415, c1429, plus the minute (close-open) list for per-minute test.
Run with venv python (sandbox off)."""
import calendar
import numpy as np
import pandas as pd
import dolphindb as ddb

from ddb_config import HOST, PORT, USER, PW
CODES = ["IC0000", "IF0000", "IH0000", "IM0000"]
WINDOWS = [(y, m) for y in range(2020, 2027) for m in range(1, 13)
           if (2020, 1) <= (y, m) <= (2026, 5)]
OUTCSV = "/Users/zhuisabella/xn/intraminute/focus_1415_short.csv"


def fetch(sess, code, start, end):
    q = f"""
    pt=loadTable("dfs://hft_future_ts","TickPartitioned")
    select m_nDatetime as ts, (m_nBidPrice+m_nAskPrice)/2.0 as mid
    from pt where code_init=`{code[:2]}, m_nDatetime>={start}T00:00:00,
          m_nDatetime<={end}T23:59:59, code=`{code},
          minute(m_nDatetime) between 14:15m:14:30m
    """
    return sess.run(q)


def build(df):
    df = df.drop_duplicates("ts").sort_values("ts").copy()
    df["mkey"] = df["ts"].dt.floor("min")
    df["tod"] = df["mkey"].dt.hour * 60 + df["mkey"].dt.minute
    df["date"] = df["mkey"].dt.normalize()
    bar = df.groupby(["date", "tod"])["mid"].agg(open="first", close="last").reset_index()
    op = bar.pivot(index="date", columns="tod", values="open")
    cl = bar.pivot(index="date", columns="tod", values="close")
    out = pd.DataFrame(index=op.index)
    out["o1415"] = op[855] if 855 in op.columns else np.nan          # 14:15 open (entry)
    out["c1429"] = cl[869] if 869 in cl.columns else np.nan          # 14:29 close (~14:30 exit)
    # per-minute close-open sum over 14:15..14:29 (for per-minute short)
    mins = [t for t in range(855, 870) if t in op.columns and t in cl.columns]
    out["minmove_sum"] = sum((cl[t] - op[t]) for t in mins)
    return out.dropna(subset=["o1415", "c1429"]).reset_index()


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
            o = build(df)
            if not len(o):
                continue
            o["code"] = code; o["ym"] = f"{yr}-{mo:02d}"
            rows.append(o)
        print(f"{yr}-{mo:02d} done", flush=True)
    sess.close()
    pd.concat(rows, ignore_index=True).to_csv(OUTCSV, index=False)
    print(f"saved {OUTCSV}")
