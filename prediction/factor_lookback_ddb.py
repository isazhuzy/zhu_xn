"""factor_lookback_ddb.py — the matched LOOK-BACK mirror of fig115 (same definitions).
Native 500ms snapshots; factor accumulated over the last L snapshots (VOI:sum, OIR/MPB:mean);
FIXED forward = future 2s averaged mid change y_4 = mean(M_{t+1..t+4}) - M_t. Single-factor
OOS R^2 for VOI/OIR/MPB, look-back L in {0.5s,2s,10s,60s,5min,20min}. Train..2024-12, OOS 2025-26.
Outputs: flb2_results.csv (code,factor,look_s,r2_is,r2_oos)
Run: /Users/zhuisabella/xn/.venv/bin/python factor_lookback_ddb.py   (sandbox OFF)
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
LOOK = {1: "0.5s", 4: "2s", 20: "10s", 120: "60s", 600: "5min", 2400: "20min"}   # snapshots
KF = 4                                                     # fixed forward: 2s averaged
TREND = (2024, 12)


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
    acc = {}
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
            for gid, g in df.groupby("gid", sort=False):
                mid = g["mid_tk"].to_numpy(float)
                y = np.full(len(mid), np.nan)
                if len(mid) > KF:
                    fut = pd.Series(mid).rolling(KF).mean().shift(-KF).to_numpy()
                    y = fut - mid
                y = np.where(np.abs(y) <= 100, y, np.nan)
                voi = pd.Series(g["voi"].to_numpy()); oir = pd.Series(g["oir"].to_numpy()); mpb = pd.Series(g["mpb"].to_numpy())
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
                rows.append(dict(code=code, factor=f, look_snap=L, look=LOOK[L], r2_is=r2is, r2_oos=r2oos))
    pd.DataFrame(rows).to_csv(f"{D}/flb2_results{SUF}.csv", index=False)
    print(f"saved flb2_results{SUF}.csv")
