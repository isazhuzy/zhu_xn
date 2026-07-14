"""mpb_return_ddb.py — MPB's STANDALONE impact on future return, windows aligned to the
VOI study (paper 3). MPB(t) = avg trade price − avg mid this 500ms → predict the future
average mid change y_k, k ∈ {1,4,20,120} snapshots = 0.5s,2s,10s,60s. Single-factor OLS,
train 2020-01…2024-12 freeze, OOS 2025-01…2026-05. Also a binned scatter at k=4 (2s).
Outputs: mpb_results.csv (code,k,secs,r2_is,r2_oos,slope,n), mpb_scatter.csv (code,mpb,mean_y,n)
Run: /Users/zhuisabella/xn/.venv/bin/python mpb_return_ddb.py   (sandbox OFF)
"""
import calendar
import os
import numpy as np, pandas as pd, dolphindb as ddb
from ddb_config import HOST, PORT, USER, PW
from lob_common import CODES, MULT, TICK, fetch_l1, prep_l1

PILOT = os.environ.get("PILOT") == "1"
SUF = "_pilot" if PILOT else ""
D = "/Users/zhuisabella/xn/prediction"
K = [1, 4, 20, 120]
SECS = {1: 0.5, 4: 2, 20: 10, 120: 60}
TREND = (2024, 12)
BW, CAP = 0.1, 3.0                                   # MPB scatter bin width / cap (ticks)


def build(df, code):
    g = df.groupby("gid", sort=False)
    tp = np.where(df.vol > 0, df.amt / (df.vol * MULT[code[:2]]), np.nan) / TICK
    tp = pd.Series(tp, index=df.index).groupby(df["gid"]).ffill().fillna(df["mid_tk"])
    df["mpb"] = tp - (df["mid_tk"] + g["mid_tk"].shift(1)) / 2
    gm = df.groupby("gid", sort=False)["mid_tk"]
    for k in K:
        df[f"y{k}"] = gm.transform(lambda s, k=k: s.rolling(k).mean().shift(-k)) - df["mid_tk"]
        df.loc[df[f"y{k}"].abs() > 100, f"y{k}"] = np.nan
    return df


def r2(mom, beta=None):
    n, Sx, Sy, Sxy, Sxx, Syy = mom
    if n < 500:
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
    acc = {}                                # (code,k,phase) -> moments[6]
    scat = {}                               # (code,bin) -> [sum_y, n]   (k=4)
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
            df = build(df, code)
            x = df["mpb"].to_numpy(float)
            for k in K:
                y = df[f"y{k}"].to_numpy(float)
                m = np.isfinite(x) & np.isfinite(y)
                xv, yv = x[m], y[m]
                key = (code, k, phase)
                acc[key] = acc.get(key, np.zeros(6)) + [len(xv), xv.sum(), yv.sum(),
                                                        (xv * yv).sum(), (xv * xv).sum(), (yv * yv).sum()]
            y4 = df["y4"].to_numpy(float)
            m = np.isfinite(x) & np.isfinite(y4) & (np.abs(x) <= CAP)
            b = (np.round(x[m] / BW) * BW).round(2)
            for bc, yy in zip(b, y4[m]):
                s = scat.setdefault((code, float(bc)), [0.0, 0]); s[0] += yy; s[1] += 1
        print(f"{yr}-{mo:02d} done ({phase})", flush=True)
    sess.close()

    rows = []
    for code in CODES:
        for k in K:
            tr = acc.get((code, k, "IS")); te = acc.get((code, k, "OOS"))
            if tr is None:
                continue
            r2is, beta = r2(tr)
            r2oos, _ = r2(te, beta) if te is not None and beta is not None else (np.nan, None)
            rows.append(dict(code=code, k=k, secs=SECS[k], r2_is=r2is, r2_oos=r2oos,
                             slope=beta[1] if beta else np.nan, n_oos=int(te[0]) if te is not None else 0))
    pd.DataFrame(rows).to_csv(f"{D}/mpb_results{SUF}.csv", index=False)
    pd.DataFrame([{"code": c, "mpb": b, "mean_y": v[0] / v[1], "n": v[1]}
                  for (c, b), v in scat.items() if v[1] >= 100]).to_csv(f"{D}/mpb_scatter{SUF}.csv", index=False)
    print(f"saved mpb_results{SUF}.csv mpb_scatter{SUF}.csv")
