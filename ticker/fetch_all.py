import dolphindb as ddb
import pandas as pd

from ddb_config import HOST, PORT, USER, PW
sess = ddb.session(HOST, PORT)
sess.login(USER, PW)

products = ["IC", "IF", "IH", "IM"]

st_d = "2023.01.04"
en_d = "2023.03.04"

for product in products:

    print(f"Downloading {product}...")

    spt = f"""
    pt=loadTable("dfs://hft_future_ts","TickPartitioned")

    select
        code,
        m_nDatetime,
        m_nPrice,
        m_iVolume,
        m_nBidPrice,
        m_nBidVolume,
        m_nAskPrice,
        m_nAskVolume
    from pt
    where
        m_nDatetime.date()>={st_d}d,
        m_nDatetime.date()<={en_d}d,
        code_init=`{product}
    """

    df = sess.run(spt)

    print(product, df.shape)

    filename = f"{product}_{st_d.replace('.','')}_{en_d.replace('.','')}.csv"

    df.to_csv(filename, index=False)

    print(f"Saved {filename}")


