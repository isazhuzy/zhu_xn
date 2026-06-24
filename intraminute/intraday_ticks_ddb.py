"""intraday_ticks_ddb.py — average ticks per minute-OF-DAY (09:30, 09:31, ... 15:00),
averaged across all days, per contract. 2020-01..2026-05, distinct timestamps per minute.
Saves intraday_ticks.csv (code, tod, avg_ticks, ndays). tod = minutes since midnight.
Run: /Users/zhuisabella/xn/.venv/bin/python intraday_ticks_ddb.py   (sandbox OFF)
"""
import pandas as pd, dolphindb as ddb
from ddb_config import HOST, PORT, USER, PW
CODES = ["IC0000", "IF0000", "IH0000", "IM0000"]
OUT = "/Users/zhuisabella/xn/intraminute/intraday_ticks.csv"

sess = ddb.session(HOST, PORT); sess.login(USER, PW)
frames = []
for code in CODES:
    q = f"""
    pt=loadTable("dfs://hft_future_ts","TickPartitioned")
    d0 = select count(*) as c from pt
         where code_init=`{code[:2]}, code=`{code},
               m_nDatetime>=2020.01.01T00:00:00, m_nDatetime<=2026.05.31T23:59:59,
               minute(m_nDatetime) between 09:30m:15:00m
         group by date(m_nDatetime) as dd, int(minute(m_nDatetime)) as tod, m_nDatetime
    mc = select count(*) as cnt from d0 group by dd, tod
    select avg(cnt) as avg_ticks, count(*) as ndays from mc group by tod order by tod
    """
    r = sess.run(q)
    r.insert(0, "code", code)
    frames.append(r)
    print(f"{code}: {len(r)} minute-of-day rows, open(09:30)={r[r.tod==570]['avg_ticks'].values}", flush=True)
sess.close()
pd.concat(frames).to_csv(OUT, index=False)
print(f"saved {OUT}")
