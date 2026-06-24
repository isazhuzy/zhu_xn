"""tickcount_hist_ddb.py — verifiable histogram of ticks-per-minute, 2020-01..2026-05.
Per contract: distinct timestamps per session minute -> histogram (how many minutes had
exactly k ticks). Saves ticks_per_minute_hist.csv (code, cnt, n_minutes) so the summary
stats can be recomputed by hand. Also prints mean/median/percentiles from the histogram.
Run: /Users/zhuisabella/xn/.venv/bin/python tickcount_hist_ddb.py   (sandbox OFF)
"""
import numpy as np, pandas as pd, dolphindb as ddb
from ddb_config import HOST, PORT, USER, PW
CODES = ["IC0000", "IF0000", "IH0000", "IM0000"]
OUT = "/Users/zhuisabella/xn/intraminute/ticks_per_minute_hist.csv"


def pct(cnt, n, p):
    order = np.argsort(cnt); cnt, n = cnt[order], n[order]
    cum = np.cumsum(n); tot = cum[-1]
    return cnt[np.searchsorted(cum, p * tot)]


sess = ddb.session(HOST, PORT); sess.login(USER, PW)
frames = []
print("%-12s %6s %6s %5s %5s %5s %5s %5s  %9s" % ("code", "mean", "med", "p10", "p25", "p75", "p90", "p99", "n_minutes"))
for code in CODES:
    q = f"""
    pt=loadTable("dfs://hft_future_ts","TickPartitioned")
    d0 = select count(*) as c from pt
         where code_init=`{code[:2]}, code=`{code},
               m_nDatetime>=2020.01.01T00:00:00, m_nDatetime<=2026.05.31T23:59:59,
               minute(m_nDatetime) between 09:30m:15:00m
         group by date(m_nDatetime) as dd, minute(m_nDatetime) as mm, m_nDatetime
    mc = select count(*) as cnt from d0 group by dd, mm
    select count(*) as n_minutes from mc group by cnt order by cnt
    """
    h = sess.run(q)
    h.insert(0, "code", code)
    frames.append(h)
    cnt = h["cnt"].to_numpy().astype(float); n = h["n_minutes"].to_numpy().astype(float)
    tot = n.sum(); mean = (cnt * n).sum() / tot
    print("%-12s %6.1f %6.0f %5.0f %5.0f %5.0f %5.0f %5.0f  %9d" % (
        code, mean, pct(cnt, n, .5), pct(cnt, n, .1), pct(cnt, n, .25),
        pct(cnt, n, .75), pct(cnt, n, .9), pct(cnt, n, .99), int(tot)))
sess.close()
pd.concat(frames).to_csv(OUT, index=False)
print(f"\nsaved {OUT}")
