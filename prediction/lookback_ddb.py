"""lookback_ddb.py — mirror of fig111: fix the forward step, vary the LOOKBACK 20s→20min.
For each factor accumulated over a trailing window L, predict the next 10s return:
  VOI : net flow  = Σ VOI over last L bars
  OIR : avg level = mean OIR over last L bars
  MPB : avg trade basis = mean MPB over last L bars
10s base bars; L ∈ {20s,1min,5min,15min,20min}; single-factor OLS; train …2024-12, OOS 2025-26.
Shows how predictive power depends on how far BACK you aggregate (recent vs stale flow).
Outputs: lookback_grid.csv (code, factor, look_s, r2_is, r2_oos, slope, n)
Run: /Users/zhuisabella/xn/.venv/bin/python lookback_ddb.py   (sandbox OFF)
"""
import calendar
import os
import numpy as np, pandas as pd, dolphindb as ddb
from ddb_config import HOST, PORT, USER, PW
from lob_common import CODES, MULT, TICK, fetch_l1, prep_l1

PILOT = os.environ.get("PILOT") == "1"
SUF = "_pilot" if PILOT else ""
D = "/Users/zhuisabella/xn/prediction"
BAR = 2                                        # 2s bars: forward = next 2s (signal still alive)
LOOK = {1: "2s(即时)", 10: "20s", 30: "1min", 150: "5min", 450: "15min", 600: "20min"}
TREND = (2024, 12)


def snap_factors(df, code):
    g = df.groupby("gid", sort=False)
    pbi = np.rint(df.pb / TICK); pai = np.rint(df.pa / TICK)
    pb1 = g["pb"].shift(1); pa1 = g["pa"].shift(1)
    pb1i = np.rint(pb1 / TICK); pa1i = np.rint(pa1 / TICK)
    qb1 = g["qb"].shift(1); qa1 = g["qa"].shift(1)
    dvb = np.where(pbi < pb1i, 0.0, np.where(pbi == pb1i, df.qb - qb1, df.qb))
    dva = np.where(pai > pa1i, 0.0, np.where(pai == pa1i, df.qa - qa1, df.qa))
    df["voi"] = np.where(pb1.notna().to_numpy(), dvb - dva, 0.0)
    df["oir"] = (df.qb - df.qa) / (df.qb + df.qa)
    tp = np.where(df.vol > 0, df.amt / (df.vol * MULT[code[:2]]), np.nan) / TICK
    tp = pd.Series(tp, index=df.index).groupby(df["gid"]).ffill().fillna(df["mid_tk"])
    df["mpb"] = tp - (df["mid_tk"] + g["mid_tk"].shift(1)) / 2
    return df


def bars(df):
    b = (df["ts"].astype("int64") // 10**9 // BAR).astype("int64")
    return df.assign(bar=b).groupby(["gid", "bar"], sort=True).agg(
        voi=("voi", "sum"), oir=("oir", "mean"), mpb=("mpb", "mean"), mid=("mid_tk", "last")).reset_index()


def r2(mom, beta=None):
    n, Sx, Sy, Sxy, Sxx, Syy = mom
    if n < 500 or (n * Sxx - Sx * Sx) <= 0:
        return np.nan, None
    if beta is None:
        b = (n * Sxy - Sx * Sy) / (n * Sxx - Sx * Sx); a = (Sy - b * Sx) / n; beta = (a, b)
    a, b = beta
    sse = Syy - 2 * a * Sy - 2 * b * Sxy + a * a * n + 2 * a * b * Sx + b * b * Sxx
    sst = Syy - Sy * Sy / n
    return (1 - sse / sst if sst > 0 else np.nan), beta


if __name__ == "__main__":
    months = [(2024, 11), (2025, 1)] if PILOT else [
        (y, m) for y in range(2020, 2027) for m in range(1, 13) if (2020, 1) <= (y, m) <= (2026, 5)]
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    acc = {}                                  # (code,factor,L,phase) -> moments[6]
    for yr, mo in months:
        phase = "IS" if (yr, mo) <= TREND else "OOS"
        last = calendar.monthrange(yr, mo)[1]
        for code in CODES:
            df = fetch_l1(sess, code, f"{yr}.{mo:02d}.01", f"{yr}.{mo:02d}.{last:02d}")
            if not len(df):
                continue
            df = prep_l1(df)
            if df.empty:
                continue
            bb = bars(snap_factors(df, code))
            for gid, g in bb.groupby("gid", sort=False):
                g = g.set_index("bar")
                idx = np.arange(g.index.min(), g.index.max() + 1)
                voi = pd.Series(g["voi"].reindex(idx).fillna(0.0).to_numpy())
                oir = pd.Series(g["oir"].reindex(idx).ffill().to_numpy())
                mpb = pd.Series(g["mpb"].reindex(idx).fillna(0.0).to_numpy())
                mid = g["mid"].reindex(idx).ffill().to_numpy(float)
                y = np.r_[np.diff(mid), np.nan]        # next-bar (10s) return
                for L in LOOK:
                    feats = {"VOI": voi.rolling(L).sum().to_numpy(),
                             "OIR": oir.rolling(L).mean().to_numpy(),
                             "MPB": mpb.rolling(L).mean().to_numpy()}
                    for f, x in feats.items():
                        m = np.isfinite(x) & np.isfinite(y)
                        xv, yv = x[m], y[m]
                        k = (code, f, L, phase)
                        acc[k] = acc.get(k, np.zeros(6)) + [len(xv), xv.sum(), yv.sum(),
                                                            (xv * yv).sum(), (xv * xv).sum(), (yv * yv).sum()]
        print(f"{yr}-{mo:02d} done ({phase})", flush=True)
    sess.close()

    rows = []
    for code in CODES:
        for f in ("VOI", "OIR", "MPB"):
            for L in LOOK:
                tr = acc.get((code, f, L, "IS")); te = acc.get((code, f, L, "OOS"))
                if tr is None:
                    continue
                r2is, beta = r2(tr)
                r2oos, _ = r2(te, beta) if te is not None and beta is not None else (np.nan, None)
                rows.append(dict(code=code, factor=f, look_bars=L, look=LOOK[L],
                                 r2_is=r2is, r2_oos=r2oos, slope=beta[1] if beta else np.nan))
    pd.DataFrame(rows).to_csv(f"{D}/lookback_grid{SUF}.csv", index=False)
    print(f"saved lookback_grid{SUF}.csv")
