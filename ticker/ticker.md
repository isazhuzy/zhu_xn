import dolphindb as ddb
import pandas as pd

st_d = '2023.01.04'
en_d = '2023.01.04'
code_init = 'IC'

spt = """pt=loadTable("dfs://hft_future_ts","TickPartitioned")
select code,m_nDatetime,m_nPrice,m_iVolume,m_nBidPrice,m_nBidVolume,m_nAskPrice,m_nAskVolume from pt where m_nDatetime.date()>={st_d}d and m_nDatetime.date()<={en_d}d and code_init=`{code_init}
""".format(st_d=st_d, en_d=en_d, code_init=code_init)

block = sess.run(spt, fetchSize=18432)

total = []
while block.hasNext():
    tem = block.read()
    total.append(tem)

df_ddb = pd.concat(total)
df_ddb
#####
spt = """
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
    m_nDatetime.date()>=2023.01.04d,
    m_nDatetime.date()<=2023.01.04d,
    code_init in [`IC,`IF,`IH,`IM]
"""





products = ["IC", "IF", "IH", "IM"]

for product in products:

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
        m_nDatetime.date()=2023.01.04d,
        code_init=`{product}
    """

    block = sess.run(spt, fetchSize=18432)

    total = []

    while block.hasNext():
        total.append(block.read())

    df = pd.concat(total)

    print(product, df.shape)

    df.to_csv(
        f"{product}_20230104.csv",
        index=False
    )