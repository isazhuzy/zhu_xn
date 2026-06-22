"""focus_spread_ddb.py — measure the REAL bid-ask spread during 11:21-11:29, all 76 months.
Round-trip cost of a market-order strategy (enter at open, exit at close, both vs mid) ≈ the
average spread (ask-bid) in index points. Replaces the assumed 0.1/0.2 cost with a measured one.
Output: code, year, month, avg_spread, med_spread, n. Run with venv python (sandbox off)."""
import calendar
import numpy as np
import pandas as pd
import dolphindb as ddb

from ddb_config import HOST, PORT, USER, PW
CODES = ["IC0000", "IF0000", "IH0000", "IM0000"]
WINDOWS = [(y, m) for y in range(2020, 2027) for m in range(1, 13)
           if (2020, 1) <= (y, m) <= (2026, 5)]
OUTCSV = "/Users/zhuisabella/xn/intraminute/focus_spread.csv"


def fetch(sess, code, start, end):
    q = f"""
    pt=loadTable("dfs://hft_future_ts","TickPartitioned")
    select avg(m_nAskPrice - m_nBidPrice) as avg_spread,
           median(m_nAskPrice - m_nBidPrice) as med_spread, count(*) as n
    from pt where code_init=`{code[:2]}, m_nDatetime>={start}T00:00:00,
          m_nDatetime<={end}T23:59:59, code=`{code},
          minute(m_nDatetime) between 11:21m:11:29m, m_nAskPrice>0, m_nBidPrice>0
    """
    return sess.run(q)


if __name__ == "__main__":
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    rows = []
    for yr, mo in WINDOWS:
        last = calendar.monthrange(yr, mo)[1]
        start, end = f"{yr}.{mo:02d}.01", f"{yr}.{mo:02d}.{last:02d}"
        for code in CODES:
            r = fetch(sess, code, start, end)
            if r is None or not len(r) or not np.isfinite(r["avg_spread"][0]):
                continue
            rows.append({"code": code, "year": yr, "month": mo,
                         "avg_spread": float(r["avg_spread"][0]),
                         "med_spread": float(r["med_spread"][0]), "n": int(r["n"][0])})
        print(f"{yr}-{mo:02d} done", flush=True)
    sess.close()
    pd.DataFrame(rows).to_csv(OUTCSV, index=False)
    print(f"saved {OUTCSV}")
