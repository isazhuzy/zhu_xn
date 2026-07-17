import calendar
import os
import sys

import numpy as np
import pandas as pd
import dolphindb as ddb

sys.path.insert(0, "/Users/zhuisabella/xn/prediction")
from ddb_config import HOST, PORT, USER, PW
from lob_common import TICK, CODES

PILOT = os.environ.get("PILOT") == "1"
PRICE = os.environ.get("PRICE", "last") #or mid
SUF = "_pilot" if PILOT else ""
D = "/Users/zhuisabella/xn/manual"
K_FWD = [1, 5, 10,20] #500ms
NPTS = 4000

if PILOT:
    MONTHS = [(2024, m) for m in range(1, 13)]
else:
    MONTHS = [(y, m) for y in range(2020, 2027) for m in range(1, 13)
              if (2020, 1) <= (y, m) <= (2026, 5)]


def fetch(sess, code, yr, mo):
    """added newest price"""
    last = calendar.monthrange(yr, mo)[1]
    q = f"""
    pt=loadTable("dfs://hft_future_ts","TickPartitioned")
    select m_nDatetime as ts, m_nPrice as px,
           m_nBidPrice as pb, m_nBidVolume as qb,
           m_nAskPrice as pa, m_nAskVolume as qa
    from pt where code_init=`{code[:2]}, code=`{code},
          m_nDatetime>={yr}.{mo:02d}.01T00:00:00,
          m_nDatetime<={yr}.{mo:02d}.{last:02d}T23:59:59,
          minute(m_nDatetime) between 09:30m:15:00m
    """
    return sess.run(q)


def prep(df):
    """clean up, AM/PM sessions"""
    df = df.drop_duplicates("ts").sort_values("ts").copy()
    df = df[(df.pb > 0) & (df.pa > df.pb) & (df.px > 0)] #filter one sided books
    tod = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    df = df[((tod >= 570) & (tod <= 690)) | ((tod >= 780) & (tod <= 900))]
    if df.empty:
        return df
    spr = np.rint((df.pa - df.pb) / TICK).astype(int)
    df = df[(spr >= 1) & (spr <= 50)]
    if df.empty:
        return df
    pm = ((df["ts"].dt.hour * 60 + df["ts"].dt.minute) >= 780).astype("int64")
    df["gid"] = df["ts"].dt.normalize().astype("int64") * 2 + pm #normalize() chops a timestamp to midnight; .astype("int64") turns it into nanoseconds-since-epoch (a unique number per day); *2 + pm makes AM and PM of the same day two different numbers. Result: one integer id per (day, session).
    return df.reset_index(drop=True)


def add_voi(df):
    """VOI"""
    g = df.groupby("gid", sort=False)
    pbi = np.rint(df.pb / TICK).astype(int)
    pai = np.rint(df.pa / TICK).astype(int)
    pb1 = g["pb"].shift(1); pa1 = g["pa"].shift(1)
    pb1i = np.rint(pb1 / TICK); pa1i = np.rint(pa1 / TICK)
    qb1 = g["qb"].shift(1); qa1 = g["qa"].shift(1)
    dvb = np.where(pbi < pb1i, 0.0, np.where(pbi == pb1i, df.qb - qb1, df.qb))
    dva = np.where(pai > pa1i, 0.0, np.where(pai == pa1i, df.qa - qa1, df.qa))
    df["voi"] = np.where(pb1.notna(), dvb - dva, np.nan)
    return df


def add_fwd(df):
    """Future price change per tick:
        dy_k(t) = price(t+k) - price(t)
    shift(-k) inside gid -> the last k ticks of each session get NaN instead of
    peeking into the next session. |dy|>100 ticks = bad print, drop."""
    px = df["px"] if PRICE == "last" else (df.pb + df.pa) / 2 #traded price
    df["p_tk"] = px / TICK
    g = df.groupby("gid", sort=False)["p_tk"]
    for k in K_FWD:
        df[f"dy{k}"] = g.shift(-k) - df["p_tk"]
        df.loc[df[f"dy{k}"].abs() > 100, f"dy{k}"] = np.nan
    return df


if __name__ == "__main__":
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    dycols = [f"dy{k}" for k in K_FWD]
    curves = []
    for code in CODES: 
        parts = []
        for yr, mo in MONTHS:
            df = fetch(sess, code, yr, mo)
            if not len(df):
                continue
            df = prep(df)
            if df.empty:
                continue
            df = add_voi(df)
            df = add_fwd(df)
            v = df.dropna(subset=["voi"] + dycols)
            parts.append(v[["voi"] + dycols].astype("float32"))
            print(f"{code} {yr}-{mo:02d}: {len(v):,} ticks", flush=True)
        if not parts:
            continue
        a = pd.concat(parts, ignore_index=True); del parts
        n = len(a)

        order = np.argsort(a["voi"].to_numpy(), kind="stable")
        voi_s = a["voi"].to_numpy()[order]
        idx = np.unique(np.linspace(0, n - 1, NPTS).astype(np.int64))
        out = pd.DataFrame({"code": code, "rank": idx + 1,
                            "q": (idx + 1) / n,
                            "voi": voi_s[idx], "n_total": n})
        for k in K_FWD:
            cum = np.cumsum(a[f"dy{k}"].to_numpy()[order]) #cumsum
            out[f"cum{k}"] = cum[idx]
            print(f"{code} k={k}: end={cum[-1]:,.0f} ticks, "
                  f"min={cum.min():,.0f} at q={(cum.argmin()+1)/n:.3f} "
                  f"(voi={voi_s[cum.argmin()]:.0f})", flush=True)
        curves.append(out); del a
    sess.close()
    pd.concat(curves, ignore_index=True).to_csv(
        f"{D}/voi_cumsum_curve{SUF}.csv", index=False)
    print(f"saved voi_cumsum_curve{SUF}.csv  (price basis: {PRICE})")
