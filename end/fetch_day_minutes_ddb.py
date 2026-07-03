"""
fetch_day_minutes_ddb.py — full-history 1-min mid bars, WHOLE SESSION (09:30-15:00),
server-side aggregated (last tick mid per minute), for the end-of-day reversal study.

Same bar definition used across xn/ticker: mid=(bid+ask)/2, 1-min bar = last tick
in the minute. Chunked by year to keep each pull small (avoids DDB "too many open
files" on wide-range scans) and to allow partition pruning via a direct m_nDatetime
range (a .date()-style function does NOT prune on this table).

Run:  /Users/zhuisabella/xn/.venv/bin/python fetch_day_minutes_ddb.py   (sandbox off)
"""
import dolphindb as ddb
import pandas as pd

from ddb_config import HOST, PORT, USER, PW

CODES = ["IC0000", "IF0000", "IH0000", "IM0000"]
YEARS = list(range(2015, 2027))  # chunk by calendar year; table run out around 2026
OUT = "/Users/zhuisabella/xn/end/day_minutes_full.csv"


def fetch_year(sess, year):
    inits = "[`" + ",`".join(sorted({c[:2] for c in CODES})) + "]"
    codes = "[`" + ",`".join(CODES) + "]"
    start = f"{year}.01.01"
    end = f"{year}.12.31"
    q = f"""
    pt=loadTable("dfs://hft_future_ts","TickPartitioned")
    select last((m_nBidPrice+m_nAskPrice)/2.0) as close
    from pt
    where code_init in {inits},
          m_nDatetime >= {start}T00:00:00, m_nDatetime <= {end}T23:59:59,
          code in {codes},
          minute(m_nDatetime) between 09:30m:15:00m
    group by code as code, date(m_nDatetime) as d, minute(m_nDatetime) as mn
    """
    return sess.run(q)


if __name__ == "__main__":
    sess = ddb.session(HOST, PORT)
    sess.login(USER, PW)
    parts = []
    for year in YEARS:
        try:
            df = fetch_year(sess, year)
        except Exception as e:
            print(f"{year}: FAILED ({e})", flush=True)
            continue
        print(f"{year}: {df.shape[0]} rows", flush=True)
        if df.shape[0] > 0:
            parts.append(df)
    sess.close()
    full = pd.concat(parts, ignore_index=True)
    print("total rows:", full.shape[0], flush=True)
    full.to_csv(OUT, index=False)
    print("saved", OUT, flush=True)
