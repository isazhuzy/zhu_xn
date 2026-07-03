"""lob_common.py — shared fetch/prep for the order-book prediction studies
(qi_ddb.py / voi_ddb.py / microprice_ddb.py). Each script imports from here.
Sandbox must be OFF (LAN DolphinDB at 192.168.1.7).

Conventions (same as ofi_full_ddb.py):
  - trading sessions AM 09:30-11:30, PM 13:00-15:00, grouped by (day, session)
  - dedup timestamps (2024-02 duplicate-row quirk), drop crossed/locked/absurd quotes
  - mid_tk = mid-price in TICK units (CFFEX index futures tick = 0.2 pts)
"""
import numpy as np
import pandas as pd

TICK = 0.2
MULT = {"IC": 200, "IF": 300, "IH": 300, "IM": 200}   # contract multiplier (yuan/pt)
CODES = ["IC0000", "IF0000", "IH0000", "IM0000"]


def month_windows(pilot=False):
    if pilot:
        return [(2024, 6), (2024, 7)]
    return [(y, m) for y in range(2020, 2027) for m in range(1, 13)
            if (2020, 1) <= (y, m) <= (2026, 5)]


def train_end(pilot=False):
    """last (year, month) included in the TRAIN phase; later months are TEST."""
    return (2024, 6) if pilot else (2024, 12)


def fetch_l1(sess, code, start, end):
    q = f"""
    pt=loadTable("dfs://hft_future_ts","TickPartitioned")
    select m_nDatetime as ts, m_nBidPrice as pb, m_nBidVolume as qb,
           m_nAskPrice as pa, m_nAskVolume as qa,
           m_iVolume as vol, m_iTurnover as amt
    from pt where code_init=`{code[:2]}, code=`{code},
          m_nDatetime>={start}T00:00:00, m_nDatetime<={end}T23:59:59,
          minute(m_nDatetime) between 09:30m:15:00m
    """
    return sess.run(q)


def prep_l1(df):
    """Clean & annotate one (code, month) pull. Adds: spr (spread in ticks, int>=1),
    mid_tk (mid in ticks), gid (day-session group id). Keeps vol/amt for VOI's MPB."""
    df = df.drop_duplicates("ts").sort_values("ts").copy()
    df = df[(df.pb > 0) & (df.pa > 0) & (df.pa > df.pb)]
    tod = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    df = df[((tod >= 570) & (tod <= 690)) | ((tod >= 780) & (tod <= 900))]
    if df.empty:
        return df
    df["spr"] = np.rint((df.pa - df.pb) / TICK).astype(int)
    df = df[(df.spr >= 1) & (df.spr <= 50)]        # drop locked & one-sided bad quotes
    if df.empty:
        return df
    df["mid_tk"] = (df.pb + df.pa) / (2 * TICK)
    pm = ((df["ts"].dt.hour * 60 + df["ts"].dt.minute) >= 780).astype("int64")
    df["gid"] = df["ts"].dt.normalize().astype("int64") * 2 + pm
    return df.reset_index(drop=True)
