"""Experiment C v2 — unified time grid on BOTH axes, per-factor price basis.

CHANGES vs window_exp_c.py:
  * JLIST == KLIST == unified TIMES [0.5s,2.5s,10s,30s,60s,120s,300s]  (both axes
    identical -> forward-table and look-back-table columns match exactly).
  * OUTCOME price basis is now PER FACTOR (the note's rule):
        voi, oir  -> dy uses LAST price  (m_nPrice)   -- quote/flow factors, bounce-safe
        mpb       -> dy uses MID  price  ((pb+pa)/2)  -- MANDATORY (bounce cancels signal)
    We compute BOTH dy_last and dy_mid per k and pick the right one per factor.

Everything else identical to v1 (see its header for the full calc): factor_J =
within-gid rolling sum over past J ticks (sum==mean for quantile ranking, J=1 =
instantaneous); buckets = nested quantile tails; cell = mean dy_k over the tail =
ticks/trade gross.

Out: window_exp_c2_all.csv  (long: code,factor,J,k,bucket,mean_dy,n,nrows)
"""
import calendar
import os
import sys

import numpy as np
import pandas as pd
import dolphindb as ddb

sys.path.insert(0, "/Users/zhuisabella/xn/prediction")
from ddb_config import HOST, PORT, USER, PW
from lob_common import TICK, CODES, MULT

D = "/Users/zhuisabella/xn/manual"
PILOT = os.environ.get("PILOT") == "1"
SUF = "_pilot" if PILOT else ""
FACTORS = ["voi", "oir", "mpb"]
PRICE_OF = {"voi": "last", "oir": "last", "mpb": "mid"}   # <-- the note's rule
TIMES = [1, 5, 20, 60, 120, 240, 600]                     # 0.5,2.5,10,30,60,120,300 s
JLIST = TIMES
KLIST = TIMES
BUCKETS = [("neg0.1", 0.001, "lo"), ("neg1", 0.01, "lo"), ("neg5", 0.05, "lo"),
           ("pos5", 0.95, "hi"), ("pos1", 0.99, "hi"), ("pos0.1", 0.999, "hi")]

if PILOT:
    MONTHS = [(2024, m) for m in range(1, 13)]
else:
    MONTHS = [(y, m) for y in range(2020, 2027) for m in range(1, 13)
              if (2020, 1) <= (y, m) <= (2026, 5)]


def fetch(sess, code, yr, mo):
    last = calendar.monthrange(yr, mo)[1]
    q = f"""
    pt=loadTable("dfs://hft_future_ts","TickPartitioned")
    select m_nDatetime as ts, m_nPrice as px,
           m_nBidPrice as pb, m_nBidVolume as qb,
           m_nAskPrice as pa, m_nAskVolume as qa,
           m_iVolume as vol, m_iTurnover as amt
    from pt where code_init=`{code[:2]}, code=`{code},
          m_nDatetime>={yr}.{mo:02d}.01T00:00:00,
          m_nDatetime<={yr}.{mo:02d}.{last:02d}T23:59:59,
          minute(m_nDatetime) between 09:30m:15:00m
    """
    return sess.run(q)


def prep(df):
    df = df.drop_duplicates("ts").sort_values("ts").copy()
    df = df[(df.pb > 0) & (df.pa > df.pb) & (df.px > 0)]
    tod = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    df = df[((tod >= 570) & (tod <= 690)) | ((tod >= 780) & (tod <= 900))]
    if df.empty:
        return df
    spr = np.rint((df.pa - df.pb) / TICK).astype(int)
    df = df[(spr >= 1) & (spr <= 50)]
    if df.empty:
        return df
    pm = ((df["ts"].dt.hour * 60 + df["ts"].dt.minute) >= 780).astype("int64")
    df["gid"] = df["ts"].dt.normalize().astype("int64") * 2 + pm
    return df.reset_index(drop=True)


def add_voi(df):
    g = df.groupby("gid", sort=False)
    pbi = np.rint(df.pb / TICK).astype(int); pai = np.rint(df.pa / TICK).astype(int)
    pb1 = g["pb"].shift(1); pa1 = g["pa"].shift(1)
    pb1i = np.rint(pb1 / TICK); pa1i = np.rint(pa1 / TICK)
    qb1 = g["qb"].shift(1); qa1 = g["qa"].shift(1)
    dvb = np.where(pbi < pb1i, 0.0, np.where(pbi == pb1i, df.qb - qb1, df.qb))
    dva = np.where(pai > pa1i, 0.0, np.where(pai == pa1i, df.qa - qa1, df.qa))
    df["voi"] = np.where(pb1.notna(), dvb - dva, np.nan)
    return df


def add_oir(df):
    df["oir"] = (df.qb - df.qa) / (df.qb + df.qa)
    return df


def add_mpb(df, code):
    g = df.groupby("gid", sort=False)
    df["mid_tk"] = (df.pb + df.pa) / (2 * TICK)
    tp = np.where(df.vol > 0, df.amt / (df.vol * MULT[code[:2]]), np.nan) / TICK
    tp = pd.Series(tp, index=df.index).groupby(df["gid"]).ffill().fillna(df["mid_tk"])
    df["mpb"] = tp - (df["mid_tk"] + g["mid_tk"].shift(1)) / 2
    return df


def add_fwd(df):
    """compute BOTH last-price and mid-price forward returns for every k."""
    df["last_tk"] = df["px"] / TICK
    df["mid_tk2"] = (df.pb + df.pa) / (2 * TICK)
    gl = df.groupby("gid", sort=False)["last_tk"]
    gm = df.groupby("gid", sort=False)["mid_tk2"]
    for k in KLIST:
        for basis, g, base in [("last", gl, "last_tk"), ("mid", gm, "mid_tk2")]:
            dy = g.shift(-k) - df[base]
            dy[dy.abs() > 100] = np.nan
            df[f"dy_{basis}{k}"] = dy.astype("float32")
    return df


def rollsum(df, col, J):
    if J == 1:
        return df[col]
    x = df[col].fillna(0.0)
    c = x.groupby(df["gid"]).cumsum()
    return c - c.groupby(df["gid"]).shift(J)


if __name__ == "__main__":
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    dycols = [f"dy_{b}{k}" for k in KLIST for b in ("last", "mid")]
    rows = []
    for code in CODES:
        parts = []
        for yr, mo in MONTHS:
            df = fetch(sess, code, yr, mo)
            if not len(df):
                continue
            df = prep(df)
            if df.empty:
                continue
            df = add_voi(df); df = add_oir(df); df = add_mpb(df, code); df = add_fwd(df)
            parts.append(df[["gid", "voi", "oir", "mpb"] + dycols].copy())
            print(f"{code} {yr}-{mo:02d}: {len(df):,} ticks", flush=True)
        if not parts:
            continue
        a = pd.concat(parts, ignore_index=True); del parts
        dyarr = {(b, k): a[f"dy_{b}{k}"].to_numpy() for k in KLIST for b in ("last", "mid")}
        for fac in FACTORS:
            basis = PRICE_OF[fac]
            for J in JLIST:
                f = rollsum(a, fac, J).to_numpy()
                for k in KLIST:
                    dy = dyarr[(basis, k)]
                    good = np.isfinite(f) & np.isfinite(dy)
                    fg = f[good]; dyg = dy[good]; nrows = fg.size
                    qs = {q: np.quantile(fg, q) for q in
                          (0.001, 0.01, 0.05, 0.95, 0.99, 0.999)}
                    for label, q, side in BUCKETS:
                        mask = fg <= qs[q] if side == "lo" else fg >= qs[q]
                        m = dyg[mask].mean() if mask.any() else np.nan
                        rows.append(dict(code=code, factor=fac, basis=basis, J=J, k=k,
                                         bucket=label, mean_dy=m, n=int(mask.sum()),
                                         nrows=nrows))
                print(f"{code} {fac}({basis}) J={J}: rows={a.shape[0]:,}", flush=True)
        del a
    sess.close()
    out = pd.DataFrame(rows)
    out.to_csv(f"{D}/window_exp_c2_all{SUF}.csv", index=False)
    print(f"saved window_exp_c2_all{SUF}.csv  ({len(out)} rows)")
