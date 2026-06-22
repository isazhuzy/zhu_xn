"""open_spread_perminute_ddb.py — per-MINUTE real bid-ask spread, 09:30-09:45, all 76 months.
The open spread decays fast minute-by-minute, so we need per-minute (not 15-min avg) to honestly
net the open momentum trade. Output: code, year, month, tod, avg_spread, n. Venv python, sandbox off."""
import calendar
import numpy as np
import pandas as pd
import dolphindb as ddb

from ddb_config import HOST, PORT, USER, PW
CODES = ["IC0000", "IF0000", "IM0000"]
WINDOWS = [(y, m) for y in range(2020, 2027) for m in range(1, 13)
           if (2020, 1) <= (y, m) <= (2026, 5)]
OUTCSV = "/Users/zhuisabella/xn/intraminute/open_spread_perminute.csv"


def fetch(sess, code, start, end):
    q = f"""
    pt=loadTable("dfs://hft_future_ts","TickPartitioned")
    t=select minute(m_nDatetime).int() as tod, (m_nAskPrice-m_nBidPrice) as sp
      from pt where code_init=`{code[:2]}, m_nDatetime>={start}T00:00:00,
            m_nDatetime<={end}T23:59:59, code=`{code},
            minute(m_nDatetime) between 09:30m:09:45m, m_nAskPrice>0, m_nBidPrice>0
    select avg(sp) as avg_spread, count(*) as n from t group by tod
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
            if r is None or not len(r):
                continue
            r = r.copy(); r["code"] = code; r["year"] = yr; r["month"] = mo
            rows.append(r)
        print(f"{yr}-{mo:02d} done", flush=True)
    sess.close()
    pd.concat(rows, ignore_index=True).to_csv(OUTCSV, index=False)
    print(f"saved {OUTCSV}")
