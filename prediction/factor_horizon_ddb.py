"""factor_horizon_ddb.py — fig111-style forward study for ALL three instantaneous factors.
For each factor f(t) in {VOI, OIR, MPB} (current 500ms snapshot value), single-factor OLS
predicting the future average mid change y_k = mean(M_{t+1..t+k}) - M_t at k=1,4,20,120
(=0.5,2,10,60s). Train 2020-01..2024-12 freeze, OOS 2025-26. Plus a binned scatter at k=4 (2s).
Outputs: fh_results.csv (code,factor,k,secs,r2_is,r2_oos,slope), fh_scatter.csv (code,factor,x,mean_y,n)
Run: /Users/zhuisabella/xn/.venv/bin/python factor_horizon_ddb.py   (sandbox OFF)
"""
import calendar
import os
import numpy as np, pandas as pd, dolphindb as ddb
from ddb_config import HOST, PORT, USER, PW
from lob_common import CODES, fetch_l1, prep_l1
from lookback_ddb import snap_factors

PILOT = os.environ.get("PILOT") == "1"
SUF = "_pilot" if PILOT else ""
D = "/Users/zhuisabella/xn/prediction"
K = [1, 4, 20, 120]
SECS = {1: 0.5, 4: 2, 20: 10, 120: 60}
TREND = (2024, 12)
BINCFG = {"VOI": (5.0, 150.0), "OIR": (0.05, 1.0), "MPB": (0.1, 3.0)}   # (bin width, cap)


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
    acc = {}                                  # (code,factor,k,phase) -> moments
    scat = {}                                 # (code,factor,bin) -> [sum_y, n]  (k=4)
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
            df = snap_factors(df, code)
            gm = df.groupby("gid", sort=False)["mid_tk"]
            ys = {}
            for k in K:
                yk = gm.transform(lambda s, k=k: s.rolling(k).mean().shift(-k)) - df["mid_tk"]
                ys[k] = yk.where(yk.abs() <= 100).to_numpy(float)
            for fac in ("VOI", "OIR", "MPB"):
                x = df[fac.lower()].to_numpy(float)
                for k in K:
                    y = ys[k]; m = np.isfinite(x) & np.isfinite(y)
                    xv, yv = x[m], y[m]
                    key = (code, fac, k, phase)
                    acc[key] = acc.get(key, np.zeros(6)) + [len(xv), xv.sum(), yv.sum(),
                                                            (xv * yv).sum(), (xv * xv).sum(), (yv * yv).sum()]
                bw, cap = BINCFG[fac]
                y4 = ys[4]; m = np.isfinite(x) & np.isfinite(y4) & (np.abs(x) <= cap)
                b = (np.round(x[m] / bw) * bw)
                for bc, yy in zip(b, y4[m]):
                    s = scat.setdefault((code, fac, round(float(bc), 3)), [0.0, 0]); s[0] += yy; s[1] += 1
        print(f"{yr}-{mo:02d} done ({phase})", flush=True)
    sess.close()

    rows = []
    for code in CODES:
        for fac in ("VOI", "OIR", "MPB"):
            for k in K:
                tr = acc.get((code, fac, k, "IS")); te = acc.get((code, fac, k, "OOS"))
                if tr is None:
                    continue
                r2is, beta = r2(tr)
                r2oos, _ = r2(te, beta) if te is not None and beta is not None else (np.nan, None)
                rows.append(dict(code=code, factor=fac, k=k, secs=SECS[k], r2_is=r2is, r2_oos=r2oos,
                                 slope=beta[1] if beta else np.nan))
    pd.DataFrame(rows).to_csv(f"{D}/fh_results{SUF}.csv", index=False)
    pd.DataFrame([{"code": c, "factor": f, "x": b, "mean_y": v[0] / v[1], "n": v[1]}
                  for (c, f, b), v in scat.items() if v[1] >= 100]).to_csv(f"{D}/fh_scatter{SUF}.csv", index=False)
    print(f"saved fh_results{SUF}.csv fh_scatter{SUF}.csv")
