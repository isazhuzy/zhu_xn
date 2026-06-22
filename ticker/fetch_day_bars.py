"""
fetch_day_bars.py — WHOLE-DAY 1-min mid bars (server-aggregated) for the 38-day
window, so we can profile the momentum/reversal signal across the full session.

Same bar definition as matrix.py: mid=(bid+ask)/2, 1-min bar = last tick in the
minute. Session minutes only (09:30–11:30, 13:00–15:00 has no lunch ticks anyway).

Run:  python fetch_day_bars.py            (needs DolphinDB; run with sandbox off)
"""
import dolphindb as ddb

from ddb_config import HOST, PORT, USER, PW
CODES = ["IC0000", "IF0000", "IH0000", "IM0000"]
START, END = "2023.01.04", "2023.03.04"
OUT = "/Users/zhuisabella/xn/ticker/open_breakdown/day_bars_2months.csv"


def fetch():
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    inits = "[`" + ",`".join(sorted({c[:2] for c in CODES})) + "]"
    codes = "[`" + ",`".join(CODES) + "]"
    q = f"""
    pt=loadTable("dfs://hft_future_ts","TickPartitioned")
    select last((m_nBidPrice+m_nAskPrice)/2.0) as close
    from pt
    where code_init in {inits},
          m_nDatetime >= {START}T00:00:00, m_nDatetime <= {END}T23:59:59,
          code in {codes},
          minute(m_nDatetime) between 09:30m:15:00m
    group by code as code, date(m_nDatetime) as d, minute(m_nDatetime) as mn
    """
    df = sess.run(q); sess.close()
    return df


if __name__ == "__main__":
    print(f"fetching whole-day bars {START}..{END} ...", flush=True)
    df = fetch()
    print("rows:", df.shape, flush=True)
    df.to_csv(OUT, index=False)
    print("saved", OUT, flush=True)
