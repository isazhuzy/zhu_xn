"""Experiment C — the look-back x forward INTERACTION grid, for all 3 factors,
all 4 contracts.  "Does lengthening the look-back flip momentum -> reversal?"

================  THE CALCULATION, IN FULL  ================
Per tick t we form a FACTOR (look-back) and an OUTCOME (forward), then for each
(look-back J, forward k) we report the mean outcome of the extreme factor tails.

FACTORS (per-tick, then aggregated over a look-back window):
  voi : Shen Volume-Order-Imbalance = dVbid - dVask (signed lots, a FLOW).
        1st tick of each session has no prior -> NaN.
  oir : Order-Imbalance-Ratio = (qb-qa)/(qb+qa) in [-1,+1] (a LEVEL). No history.
  mpb : Mid-Price-Basis = avg trade px this 500ms - avg of last two mids, in ticks
        (a signed trade-side PRESSURE). vol==0 -> carry last trade fwd; 1st tick NaN.

LOOK-BACK aggregation (J ticks = J*0.5 s):
  factor_J(t) = sum_{i=t-J+1..t} factor(i)   , within gid (day x AM/PM), min J terms.
  ** Because we bucket by QUANTILE (a ranking), and every kept tick sums exactly J
     terms, rolling-SUM and rolling-MEAN give the SAME ranking -> SAME buckets.
     So sum vs mean is irrelevant here; we use the sum.  J=1 = instantaneous factor.

OUTCOME (forward k ticks = k*0.5 s):  MID-TO-MID, mandatory & uniform:
  dy_k(t) = mid(t+k) - mid(t)   in ticks,  mid=(pb+pa)/2 ,  shift(-k) within gid.
  |dy|>100 ticks dropped (bad print).  Mid basis kills the bid-ask bounce that
  would otherwise cancel MPB's signal (the note's PRICE=mid lesson); using it for
  all three makes the factors directly comparable.

EXTREME BUCKETS (nested tails, same 6 as NOTE_cumsum_教学.md 五):
  neg0.1/neg1/neg5 = ticks with factor <= q(0.001/0.01/0.05)   (most net-selling)
  pos5/pos1/pos0.1 = ticks with factor >= q(0.95/0.99/0.999)   (most net-buying)
  reported value = mean(dy_k) over the bucket = ticks/trade, GROSS, sub-spread.
  (This per-trade mean is NOT inflated by window overlap -- each tick contributes
   exactly one dy -- unlike the cumsum y-axis.)

Out: window_exp_c_all.csv  (long: code,factor,J,k,bucket,mean_dy,n,nrows)
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
JLIST = [1, 5, 20, 60, 120]            # look-back: 0.5s,2.5s,10s,30s,60s
KLIST = [1, 20, 60, 120, 240, 600]     # forward : 0.5s,10s,30s,60s,120s,300s
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
    """OUTCOME = mid-to-mid, uniform for all factors (bounce-free)."""
    df["m_tk"] = (df.pb + df.pa) / (2 * TICK)
    g = df.groupby("gid", sort=False)["m_tk"]
    for k in KLIST:
        dy = g.shift(-k) - df["m_tk"]
        dy[dy.abs() > 100] = np.nan
        df[f"dy{k}"] = dy.astype("float32")
    return df


def rollsum(df, col, J):
    """within-gid rolling sum over the past J ticks (== rolling mean for ranking)."""
    if J == 1:
        return df[col]
    x = df[col].fillna(0.0)                 # NaN 1st tick taints only session-open windows
    c = x.groupby(df["gid"]).cumsum()
    return c - c.groupby(df["gid"]).shift(J)  # sum of x[t-J+1..t]; NaN for first J ticks


if __name__ == "__main__":
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    dycols = [f"dy{k}" for k in KLIST]
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
        dyarr = {k: a[f"dy{k}"].to_numpy() for k in KLIST}
        for fac in FACTORS:
            for J in JLIST:
                f = rollsum(a, fac, J).to_numpy()
                for k in KLIST:
                    dy = dyarr[k]
                    good = np.isfinite(f) & np.isfinite(dy)
                    fg = f[good]; dyg = dy[good]; nrows = fg.size
                    qs = {q: np.quantile(fg, q) for q in
                          (0.001, 0.01, 0.05, 0.95, 0.99, 0.999)}
                    for label, q, side in BUCKETS:
                        mask = fg <= qs[q] if side == "lo" else fg >= qs[q]
                        m = dyg[mask].mean() if mask.any() else np.nan
                        rows.append(dict(code=code, factor=fac, J=J, k=k,
                                         bucket=label, mean_dy=m, n=int(mask.sum()),
                                         nrows=nrows))
                print(f"{code} {fac} J={J}: rows={a.shape[0]:,}", flush=True)
        del a
    sess.close()
    out = pd.DataFrame(rows)
    out.to_csv(f"{D}/window_exp_c_all{SUF}.csv", index=False)
    print(f"saved window_exp_c_all{SUF}.csv  ({len(out)} rows)")
