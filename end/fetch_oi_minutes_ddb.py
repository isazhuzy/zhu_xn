"""
fetch_oi_minutes_ddb.py — minute-level OPEN INTEREST (position) for the profit-taking
confirmation, from dfs://hft_future_realtime/RealtimeMinKLine (single-contract months,
2024-07 onward). Whole session 09:30-15:00, all four index-future families.

Per (code, code_init, day, minute): close, totalVolume (cumulative), oi (=position).
Dominant contract per (family, day) is picked in analysis (max end-of-day totalVolume),
then stitched into a continuous minute-OI series. Chunked by year.
Run with .venv, sandbox off.
"""
import dolphindb as ddb
import pandas as pd

from ddb_config import HOST, PORT, USER, PW

INITS = ["IC", "IF", "IH", "IM"]
YEARS = [(20240101, 20241231), (20250101, 20251231), (20260101, 20261231)]
OUT = "/Users/zhuisabella/xn/end/oi_minutes.csv"


def fetch_chunk(sess, d0, d1):
    inits = "[`" + ",`".join(INITS) + "]"
    q = f"""
    pt=loadTable("dfs://hft_future_realtime","RealtimeMinKLine")
    select last(close) as close, last(totalVolume) as tvol, last(position) as oi
    from pt
    where code_init in {inits}, m_nDate >= {d0}, m_nDate <= {d1},
          minute(m_nDatetime) between 09:30m:15:00m
    group by code as code, code_init as code_init,
             date(m_nDatetime) as d, minute(m_nDatetime) as mn
    """
    return sess.run(q)


if __name__ == "__main__":
    sess = ddb.session(HOST, PORT)
    sess.login(USER, PW)
    parts = []
    for d0, d1 in YEARS:
        try:
            df = fetch_chunk(sess, d0, d1)
        except Exception as e:
            print(f"{d0}-{d1}: FAILED ({e})", flush=True)
            continue
        print(f"{d0}-{d1}: {df.shape[0]} rows", flush=True)
        if df.shape[0] > 0:
            parts.append(df)
    sess.close()
    full = pd.concat(parts, ignore_index=True)
    print("total rows:", full.shape[0], flush=True)
    full.to_csv(OUT, index=False)
    print("saved", OUT, flush=True)
