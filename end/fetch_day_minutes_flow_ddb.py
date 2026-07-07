"""
fetch_day_minutes_flow_ddb.py — like fetch_day_minutes_ddb.py, but additionally
aggregates per-minute AGGRESSOR order flow so we can test whether selling pressure
(profit-taking) into an up-day predicts a weaker close.

Per (code, day, minute) bar, whole session 09:30-15:00, full history 2015-2026:
  close   = last mid = last((bid+ask)/2)
  actbid  = sum(m_nActBidVolume)   -- active volume on the bid side within the minute
  actask  = sum(m_nActAskVolume)   -- active volume on the ask side within the minute
(Semantics of "ActBid" vs "ActAsk" as buyer/seller-initiated are resolved in analysis
by checking the sign of corr(minute return, actbid-actask).)

Chunked by year (DDB open-file limit). Run with .venv, sandbox off.
"""
import dolphindb as ddb
import pandas as pd

from ddb_config import HOST, PORT, USER, PW

CODES = ["IC0000", "IF0000", "IH0000", "IM0000"]
YEARS = list(range(2015, 2027))
OUT = "/Users/zhuisabella/xn/end/day_minutes_flow_full.csv"


def fetch_year(sess, year):
    inits = "[`" + ",`".join(sorted({c[:2] for c in CODES})) + "]"
    codes = "[`" + ",`".join(CODES) + "]"
    q = f"""
    pt=loadTable("dfs://hft_future_ts","TickPartitioned")
    select last((m_nBidPrice+m_nAskPrice)/2.0) as close,
           sum(m_nActBidVolume) as actbid,
           sum(m_nActAskVolume) as actask
    from pt
    where code_init in {inits},
          m_nDatetime >= {year}.01.01T00:00:00, m_nDatetime <= {year}.12.31T23:59:59,
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
