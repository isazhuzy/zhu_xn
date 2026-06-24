"""tickcount_ddb.py — actual ticks-per-minute distribution per contract, full window.
Counts DISTINCT timestamps per calendar minute (so 2024-02 duplicate rows don't inflate),
session minutes only, then summarises. Run: /Users/zhuisabella/xn/.venv/bin/python tickcount_ddb.py
"""
import numpy as np, pandas as pd, dolphindb as ddb
from ddb_config import HOST, PORT, USER, PW
CODES = ["IC0000", "IF0000", "IH0000", "IM0000"]

sess = ddb.session(HOST, PORT); sess.login(USER, PW)
print("%-12s %6s %6s %5s %5s %5s %5s  %7s  %s" %
      ("code", "mean", "med", "p10", "p25", "p75", "p90", "nmin", "%≥115(满tick)"))
for code in CODES:
    q = f"""
    pt=loadTable("dfs://hft_future_ts","TickPartitioned")
    dd0 = select count(*) as c1 from pt
          where code_init=`{code[:2]}, code=`{code},
                minute(m_nDatetime) between 09:30m:15:00m
          group by date(m_nDatetime) as dd, minute(m_nDatetime) as mm, m_nDatetime
    mc = select count(*) as cnt from dd0 group by dd, mm
    select avg(cnt) as mean, median(cnt) as med,
           percentile(cnt,10) as p10, percentile(cnt,25) as p25,
           percentile(cnt,75) as p75, percentile(cnt,90) as p90,
           count(*) as nmin, avg(cnt>=115)*100 as full_pct,
           avg(cnt==120)*100 as exact120_pct, max(cnt) as mx from mc
    """
    r = sess.run(q)
    row = r.iloc[0]
    print("%-12s %6.1f %6.0f %5.0f %5.0f %5.0f %5.0f  %7d  %.0f%% (恰好120的占%.0f%%, max=%d)" %
          (code, row["mean"], row["med"], row["p10"], row["p25"], row["p75"], row["p90"],
           row["nmin"], row["full_pct"], row["exact120_pct"], row["mx"]))
sess.close()
