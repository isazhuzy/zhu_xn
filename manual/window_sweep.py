"""Window sweep for the sorted-cumsum extreme-bucket experiment.

Two independent windows, both anchored at tick t:
  * LOOK-BACK  J : the factor is a rolling SUM of per-tick VOI over the past J
                   ticks  (J=1 recovers the instantaneous VOI of the note).
                   VOI is a signed flow (lots), so summing = net order-flow
                   imbalance accumulated over the past J*0.5 s.
  * FORWARD    k : outcome dy_k(t) = P(t+k) - P(t) in ticks (last price).

For every (J, k) we report the single-trade mean forward return (ticks/trade)
of the extreme factor buckets -- the same 6 nested tails as NOTE_cumsum_教学.md
section 五:  most-neg 0.1/1/5 %  and  most-pos 5/1/0.1 %.
Computed directly from quantile masks (exact bucket means, no cumsum-slope
interpolation).  J=1,k=20 must reproduce the note's VOI table -> sanity check.

Env:  PILOT=1 (2024 only) | FACTOR=voi (only voi supported for look-back sum)
Out:  window_sweep_<factor><suf>.csv   (long format: code,J,k,bucket,mean_dy,n)
"""
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
FACTOR = os.environ.get("FACTOR", "voi")
SUF = "_pilot" if PILOT else ""
D = "/Users/zhuisabella/xn/manual"

JLIST = [1, 5, 20, 60, 120]            # look-back ticks: instant, 2.5s,10s,30s,60s
KLIST = [1, 20, 60, 120, 240, 600]     # forward ticks: 0.5s,10s,30s,60s,120s,300s
# nested-tail buckets: (label, q_lo, q_hi) as fraction of the sorted factor
BUCKETS = [
    ("neg0.1", 0.000, 0.001),
    ("neg1",   0.000, 0.010),
    ("neg5",   0.000, 0.050),
    ("pos5",   0.950, 1.000),
    ("pos1",   0.990, 1.000),
    ("pos0.1", 0.999, 1.000),
]

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
           m_nAskPrice as pa, m_nAskVolume as qa
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
    df["p_tk"] = df["px"] / TICK
    g = df.groupby("gid", sort=False)["p_tk"]
    for k in KLIST:
        dy = g.shift(-k) - df["p_tk"]
        dy[dy.abs() > 100] = np.nan
        df[f"dy{k}"] = dy.astype("float32")
    return df


def rollsum(df, col, J):
    """within-gid rolling sum of `col` over the past J ticks (inclusive).
    J=1 -> the column itself.  cumsum-trick, vectorized.
    valid only where J full prior ticks exist in the same gid."""
    if J == 1:
        return df[col]
    x = df[col].fillna(0.0)            # 1st tick/gid VOI is NaN; only taints windows
    c = x.groupby(df["gid"]).cumsum()  #   that straddle the session open (<0.1%)
    clag = c.groupby(df["gid"]).shift(J)
    out = c - clag                     # = sum of x[t-J+1..t]
    return out                          # NaN for the first J ticks of each gid


if __name__ == "__main__":
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
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
            df = add_voi(df)
            df = add_fwd(df)
            keep = ["gid", "voi"] + [f"dy{k}" for k in KLIST]
            parts.append(df[keep].copy())
            print(f"{code} {yr}-{mo:02d}: {len(df):,} ticks", flush=True)
        if not parts:
            continue
        a = pd.concat(parts, ignore_index=True); del parts
        # build look-back factor columns
        for J in JLIST:
            a[f"f{J}"] = rollsum(a, "voi", J).astype("float32")
        dyarr = {k: a[f"dy{k}"].to_numpy() for k in KLIST}
        for J in JLIST:
            f = a[f"f{J}"].to_numpy()
            for k in KLIST:
                dy = dyarr[k]
                good = np.isfinite(f) & np.isfinite(dy)
                fg = f[good]; dyg = dy[good]
                nrows = fg.size
                # quantile thresholds on the factor
                qs = np.quantile(fg, [0.001, 0.01, 0.05, 0.95, 0.99, 0.999])
                thr = dict(zip([0.001, 0.01, 0.05, 0.95, 0.99, 0.999], qs))
                for label, lo, hi in BUCKETS:
                    if hi <= 0.5:                 # negative tail: f <= q(hi)
                        mask = fg <= thr[hi]
                    else:                         # positive tail: f >= q(lo)
                        mask = fg >= thr[lo]
                    m = dyg[mask].mean() if mask.any() else np.nan
                    rows.append(dict(code=code, J=J, k=k, bucket=label,
                                     mean_dy=m, n=int(mask.sum()), nrows=nrows))
                print(f"{code} J={J} k={k}: rows={nrows:,}", flush=True)
        del a
    sess.close()
    out = pd.DataFrame(rows)
    out.to_csv(f"{D}/window_sweep_{FACTOR}{SUF}.csv", index=False)
    print(f"saved window_sweep_{FACTOR}{SUF}.csv  ({len(out)} rows)")
