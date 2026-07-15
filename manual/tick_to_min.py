import os
import sys
import numpy as np
import pandas as pd
import dolphindb as ddb

sys.path.insert(0, "/Users/zhuisabella/xn/prediction")
from ddb_config import HOST, PORT, USER, PW
from lob_common import MULT

def fetch_min_bars(sess, code, start, end):
    """SQL"""
    q = f"""
    pt = loadTable("dfs://hft_future_ts","TickPartitioned")
    select first(m_nPrice)    as open,
           max(m_nPrice)      as high,
           min(m_nPrice)      as low,
           last(m_nPrice)     as close,
           last(m_iAccVolume)   as accvol,
           last(m_iAccTurnover) as accamt,
           last(m_nBidPrice)  as pb,
           last(m_nAskPrice)  as pa,
           count(*)           as nticks
    from pt
    where code_init=`{code[:2]}, code=`{code},
          m_nDatetime>={start}T00:00:00, m_nDatetime<={end}T23:59:59,
          minute(m_nDatetime) between 09:25m : 15:00m
    group by code, bar(m_nDatetime, 60s) as ts
    order by ts
    """
    return sess.run(q)

def to_session_bars(b, code):
    """SQL to per minute bars"""
    b = b.copy()
    b["ts"] = pd.to_datetime(b["ts"])
    # fold 11:30:00 / 15:00:00 close-snapshot bars into 11:29 / 14:59
    hm = b.ts.dt.hour * 100 + b.ts.dt.minute
    b.loc[hm == 1130, "ts"] -= pd.Timedelta(minutes=1)
    b.loc[hm == 1500, "ts"] -= pd.Timedelta(minutes=1)
    b = b.groupby(["code", "ts"], as_index=False).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), accvol=("accvol", "last"), accamt=("accamt", "last"),
        pb=("pb", "last"), pa=("pa", "last"), nticks=("nticks", "sum"))

    day = b.ts.dt.normalize()
    b["vol"] = b.groupby(day)["accvol"].diff() #accum volume after per min ends
    b["amt"] = b.groupby(day)["accamt"].diff()
    first_of_day = day != day.shift()
    b.loc[first_of_day, ["vol", "amt"]] = np.nan   # no baseline before day's 1st bar
    hm = b.ts.dt.hour * 100 + b.ts.dt.minute

    #filter out time
    b = b[((hm >= 930) & (hm <= 1129)) | ((hm >= 1300) & (hm <= 1459))].copy()

    b["vwap"] = np.where(b.vol > 0, b.amt / (b.vol * MULT[code[:2]]), np.nan)
    b["mid_close"] = (b.pb + b.pa) / 2
    return b.drop(columns=["accvol", "accamt"]).reset_index(drop=True)

# def check_one_day(sess, code, day, bars):
#     """Rebuild in pandas"""
#     q = f"""
#     pt = loadTable("dfs://hft_future_ts","TickPartitioned")
#     select m_nDatetime as ts, m_nPrice as px, m_iVolume as dvol, m_iTurnover as damt
#     from pt where code_init=`{code[:2]}, code=`{code}, date(m_nDatetime)={day},
#           minute(m_nDatetime) between 09:30m : 15:00m
#     order by m_nDatetime
#     """
#     t = sess.run(q)
#     t["ts"] = pd.to_datetime(t["ts"])
#     t = t.drop_duplicates("ts")                       # 2024-02 duplicate-row quirk
#     t.loc[t.ts.dt.hour * 100 + t.ts.dt.minute == 1130, "ts"] -= pd.Timedelta(minutes=1)
#     t.loc[t.ts.dt.hour * 100 + t.ts.dt.minute == 1500, "ts"] -= pd.Timedelta(minutes=1)
#     m = t.set_index("ts").resample("1min").agg(
#         open=("px", "first"), high=("px", "max"), low=("px", "min"),
#         close=("px", "last"), vol=("dvol", "sum")).dropna(subset=["open"])
#     ref = bars[bars.ts.dt.normalize() == pd.Timestamp(day.replace(".", "-"))]
#     ref = ref.set_index("ts")[["open", "high", "low", "close", "vol"]]
#     joined = ref.join(m, rsuffix="_pd").dropna(subset=["open_pd"])
#     ohlc_ok = all(np.allclose(joined[c], joined[f"{c}_pd"]) for c in
#                   ["open", "high", "low", "close"])
#     vol_ok = np.allclose(joined["vol"].iloc[1:], joined["vol_pd"].iloc[1:])  # skip
#     # bar 1: SQL diff excludes auction volume, tick-sum path has no 09:29 baseline
#     print(f"[check {day}] bars={len(joined)}  OHLC match: {ohlc_ok}  "
#           f"vol match (ex 1st bar): {vol_ok}")
#     return ohlc_ok and vol_ok

if __name__ == "__main__":
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    raw = fetch_min_bars(sess, CODE, START, END)
    bars = to_session_bars(raw, CODE)
    ok = check_one_day(sess, CODE, START.rsplit(".", 1)[0] + ".03", bars)
    sess.close()
    out = f"{D}/min_bars_{CODE}{'_pilot' if PILOT else ''}.csv"
    bars.to_csv(out, index=False)
    print(bars.head(3).to_string(), f"\n... {len(bars)} bars, "
          f"{bars.ts.dt.normalize().nunique()} days -> {out}")
    print("median nticks/bar:", bars.nticks.median(), "(expect ~120 = 500ms cadence)")
