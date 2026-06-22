"""
fetch_open_bars.py — pull open-minute mid bars (9:30..9:33) from DolphinDB,
server-side aggregated so the download is tiny (~last tick mid per minute).

Reproduces matrix.py's bar definition: mid = (bid+ask)/2, 1-min bar = last tick
in the minute. We only need minutes 09:30/09:31/09:32/09:33 to form the first
two momentum signals (09:31, 09:32).

Usage:
    python fetch_open_bars.py 2023.01.04 2023.03.04 out.csv   # a range (validation)
    python fetch_open_bars.py ALL ALL out.csv                  # full history
"""
import sys
import dolphindb as ddb

from ddb_config import HOST, PORT, USER, PW
CODES = ["IC0000", "IF0000", "IH0000", "IM0000"]


def fetch(start=None, end=None):
    sess = ddb.session(HOST, PORT)
    sess.login(USER, PW)
    code_list = "[`" + ",`".join(CODES) + "]"
    inits = sorted({c[:2] for c in CODES})              # IC0000 -> IC (partition col)
    init_list = "[`" + ",`".join(inits) + "]"
    # partition-pruning predicates FIRST: code_init + a direct range on the raw
    # partition column m_nDatetime (a function like .date() does NOT prune here).
    where = [f"code_init in {init_list}"]
    if start and start != "ALL":
        where.append(f"m_nDatetime >= {start}T00:00:00")
    if end and end != "ALL":
        where.append(f"m_nDatetime <= {end}T23:59:59")
    where += [f"code in {code_list}",                   # continuous contract only
              "minute(m_nDatetime) in [09:30m,09:31m,09:32m,09:33m]"]
    q = f"""
    pt=loadTable("dfs://hft_future_ts","TickPartitioned")
    select last((m_nBidPrice+m_nAskPrice)/2.0) as close
    from pt
    where {", ".join(where)}
    group by code as code, date(m_nDatetime) as d, minute(m_nDatetime) as mn
    """
    df = sess.run(q)
    sess.close()
    return df


if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else "ALL"
    end = sys.argv[2] if len(sys.argv) > 2 else "ALL"
    out = sys.argv[3] if len(sys.argv) > 3 else "/tmp/open_bars.csv"
    print(f"fetching {start}..{end} for {CODES} ...", flush=True)
    df = fetch(start, end)
    print("rows:", df.shape, flush=True)
    print(df.head(8).to_string(index=False), flush=True)
    df.to_csv(out, index=False)
    print("saved", out, flush=True)
