"""har_ddb.py — Direction C: forecast VOLATILITY, not direction (Corsi 2009 HAR-RV, intraday).
The lesson-in-contrast to everything before: direction dies in seconds, but *variance*
is strongly predictable at minute+ horizons — and it's genuinely useful (position sizing,
options, risk).

Realized variance RV over a window = Σ (500ms mid returns)². Intraday HAR cascade:
  predict next-h RV from trailing RV at short/medium/long scales.
  log RV_next(h) ~ β0 + β_s·logRV_last(1min) + β_m·logRV_last(5min) + β_l·logRV_last(30min)
Targets h ∈ {1min, 5min}. Pooled OLS via moments, frozen on train (…2024-12), OOS 2025-26.
Also reports a naive AR(1) baseline (only trailing-1min term) to show the HAR cascade's lift.
Outputs: har_results.csv (code, horizon, model, r2_is, r2_oos, n), har_coefs.csv
Run: /Users/zhuisabella/xn/.venv/bin/python har_ddb.py   (sandbox OFF)
"""
import calendar
import os
import numpy as np, pandas as pd, dolphindb as ddb
from ddb_config import HOST, PORT, USER, PW
from lob_common import CODES, fetch_l1, prep_l1

PILOT = os.environ.get("PILOT") == "1"
SUF = "_pilot" if PILOT else ""
D = "/Users/zhuisabella/xn/prediction"
BAR = 10                                        # 10s base bar for RV
SHORT, MED, LONG = 6, 30, 180                   # trailing 1min / 5min / 30min (bars)
HOR = {6: "1min", 30: "5min"}                   # forecast horizons (bars ahead)
EPS = 1e-6
TREND = (2024, 12)
MONTHS = ([(2024, 11), (2025, 1)] if PILOT else
          [(y, m) for y in range(2020, 2027) for m in range(1, 13) if (2020, 1) <= (y, m) <= (2026, 5)])


def rv_bars(df):
    df = df.sort_values("ts")
    r = df.groupby("gid", sort=False)["mid_tk"].diff()
    df = df.assign(r2=(r * r))
    b = (df["ts"].astype("int64") // 10**9 // BAR).astype("int64")
    return df.assign(bar=b).groupby(["gid", "bar"], sort=True).agg(rv=("r2", "sum")).reset_index()


def moms(X, y):
    p = X.shape[1]
    return dict(XtX=X.T @ X, Xty=X.T @ y, yty=y @ y, n=len(y), p=p)


def r2_of(m, cols, beta=None):
    XtX, Xty, yty, n = m["XtX"], m["Xty"], m["yty"], m["n"]
    A = XtX[np.ix_(cols, cols)]; b = Xty[cols]
    if n < 500:
        return np.nan, None
    if beta is None:
        beta = np.linalg.solve(A, b)
    sse = yty - 2 * beta @ b + beta @ A @ beta
    sst = yty - Xty[0] ** 2 / n
    return (1 - sse / sst if sst > 0 else np.nan), beta


if __name__ == "__main__":
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    # design cols: [1, logRV_1min, logRV_5min, logRV_30min]  ; per (code,H,phase)
    acc = {}
    for yr, mo in MONTHS:
        phase = "IS" if (yr, mo) <= TREND else "OOS"
        last = calendar.monthrange(yr, mo)[1]
        for code in CODES:
            df = fetch_l1(sess, code, f"{yr}.{mo:02d}.01", f"{yr}.{mo:02d}.{last:02d}")
            if not len(df):
                continue
            df = prep_l1(df)
            if df.empty:
                continue
            rb = rv_bars(df)
            for gid, g in rb.groupby("gid", sort=False):
                g = g.set_index("bar")
                idx = np.arange(g.index.min(), g.index.max() + 1)
                rv = g["rv"].reindex(idx).fillna(0.0).to_numpy(float)
                s = pd.Series(rv)
                lr_s = np.log(s.rolling(SHORT).sum().to_numpy() + EPS)
                lr_m = np.log(s.rolling(MED).sum().to_numpy() + EPS)
                lr_l = np.log(s.rolling(LONG).sum().to_numpy() + EPS)
                for H in HOR:
                    fwd = s.rolling(H).sum().shift(-H).to_numpy()           # next-H RV
                    ly = np.log(fwd + EPS)
                    m = np.isfinite(lr_s) & np.isfinite(lr_m) & np.isfinite(lr_l) & np.isfinite(ly)
                    if m.sum() < 50:
                        continue
                    X = np.column_stack([np.ones(m.sum()), lr_s[m], lr_m[m], lr_l[m]])
                    y = ly[m]
                    k = (code, H, phase)
                    if k not in acc:
                        acc[k] = moms(X, y)
                    else:
                        a = acc[k]; a["XtX"] += X.T @ X; a["Xty"] += X.T @ y
                        a["yty"] += y @ y; a["n"] += len(y)
        print(f"{yr}-{mo:02d} done ({phase})", flush=True)
    sess.close()

    resrows, coefrows = [], []
    for code in CODES:
        for H in HOR:
            tr = acc.get((code, H, "IS")); te = acc.get((code, H, "OOS"))
            if tr is None:
                continue
            for model, cols in [("AR1", [0, 1]), ("HAR", [0, 1, 2, 3])]:
                r2is, beta = r2_of(tr, cols)
                r2oos, _ = r2_of(te, cols, beta) if te is not None and beta is not None else (np.nan, None)
                resrows.append(dict(code=code, horizon=HOR[H], model=model, r2_is=r2is,
                                    r2_oos=r2oos, n_is=tr["n"], n_oos=te["n"] if te else 0))
                if model == "HAR" and beta is not None:
                    for name, bv in zip(["const", "logRV_1min", "logRV_5min", "logRV_30min"], beta):
                        coefrows.append(dict(code=code, horizon=HOR[H], var=name, beta=bv))
    pd.DataFrame(resrows).to_csv(f"{D}/har_results{SUF}.csv", index=False)
    pd.DataFrame(coefrows).to_csv(f"{D}/har_coefs{SUF}.csv", index=False)
    print(f"saved har_results{SUF}.csv har_coefs{SUF}.csv")
