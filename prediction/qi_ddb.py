"""qi_ddb.py — Paper 2: Gould & Bonart (2016), *Queue Imbalance as a One-Tick-Ahead
Price Predictor in a Limit Order Book*, Market Microstructure & Liquidity.

Question: does the L1 queue imbalance  I = (qb - qa)/(qb + qa)  predict the DIRECTION
of the NEXT mid-price move?  For every 500ms snapshot we look forward to the first
future mid change within the same session and record its sign; we then estimate
P(up | I) empirically in 40 imbalance bins (+ a separate I==0 slot) and fit a grouped
logistic  p = 1/(1+exp(-(a + b*I))), overall and split by spread state (1/2/3+ ticks).

Outputs: qi_bins[_pilot].csv    (code, sprstate, x=bin center, n, n_up)
         qi_results[_pilot].csv (code, sprstate, a, b, pseudoR2, hitrate, n)
         qi_permonth[_pilot].csv(code, year, month, hitrate, n)
Run: PILOT=1 /Users/zhuisabella/xn/.venv/bin/python qi_ddb.py   (sandbox OFF)
"""
import calendar
import os
import numpy as np, pandas as pd, dolphindb as ddb
from ddb_config import HOST, PORT, USER, PW
from lob_common import CODES, fetch_l1, prep_l1, month_windows

PILOT = os.environ.get("PILOT") == "1"
SUF = "_pilot" if PILOT else ""
D = "/Users/zhuisabella/xn/prediction"
NB = 40                                   # imbalance bins over [-1,1]; slot NB = I==0
NSPR = 3                                  # spread states: 1, 2, 3+ ticks


def fit_logit(x, n, k):
    """grouped logistic p=sigmoid(a+b*x) by Newton; returns a, b, McFadden pseudo-R2."""
    m = n > 0
    x, n, k = x[m], n[m].astype(float), k[m].astype(float)
    a, b = 0.0, 1.0
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-(a + b * x)))
        w = n * p * (1 - p)
        ga, gb = (k - n * p).sum(), ((k - n * p) * x).sum()
        H = np.array([[w.sum(), (w * x).sum()], [(w * x).sum(), (w * x * x).sum()]])
        try:
            da, db = np.linalg.solve(H, [ga, gb])
        except np.linalg.LinAlgError:
            return np.nan, np.nan, np.nan
        a, b = a + da, b + db
        if max(abs(da), abs(db)) < 1e-12:
            break
    p = np.clip(1.0 / (1.0 + np.exp(-(a + b * x))), 1e-12, 1 - 1e-12)
    ll = (k * np.log(p) + (n - k) * np.log(1 - p)).sum()
    p0 = np.clip(k.sum() / n.sum(), 1e-12, 1 - 1e-12)
    ll0 = k.sum() * np.log(p0) + (n - k).sum() * np.log(1 - p0)
    return a, b, 1.0 - ll / ll0


if __name__ == "__main__":
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    cnt = {c: np.zeros((NSPR, NB + 1)) for c in CODES}   # snapshots per (spread, bin)
    up = {c: np.zeros((NSPR, NB + 1)) for c in CODES}    # ... whose next move was UP
    monrows = []
    for yr, mo in month_windows(PILOT):
        last = calendar.monthrange(yr, mo)[1]
        for code in CODES:
            df = fetch_l1(sess, code, f"{yr}.{mo:02d}.01", f"{yr}.{mo:02d}.{last:02d}")
            if not len(df):
                continue
            df = prep_l1(df)
            if df.empty:
                continue
            g = df.groupby("gid", sort=False)
            d1 = g["mid_tk"].shift(-1) - df["mid_tk"]        # mid change t -> t+1
            d1 = d1.where(d1.abs() <= 50)                    # bad-tick guard
            dirn = np.sign(d1).replace(0.0, np.nan)
            df["y"] = dirn.groupby(df["gid"]).bfill()        # sign of NEXT mid move
            df["I"] = (df.qb - df.qa) / (df.qb + df.qa)
            v = df.dropna(subset=["y"])
            binidx = np.clip(((v["I"].to_numpy() + 1) / 2 * NB).astype(int), 0, NB - 1)
            binidx = np.where(v["I"].to_numpy() == 0, NB, binidx)
            sidx = np.clip(v["spr"].to_numpy(), 1, NSPR) - 1
            isup = (v["y"].to_numpy() > 0).astype(float)
            np.add.at(cnt[code], (sidx, binidx), 1.0)
            np.add.at(up[code], (sidx, binidx), isup)
            nz = v[v["I"] != 0]
            hit = float((np.sign(nz["I"]) == nz["y"]).mean()) if len(nz) else np.nan
            monrows.append(dict(code=code, year=yr, month=mo, hitrate=hit, n=len(nz)))
        print(f"{yr}-{mo:02d} done", flush=True)
    sess.close()

    centers = np.r_[-1 + (np.arange(NB) + 0.5) * (2 / NB), 0.0]
    binrows, resrows = [], []
    for code in CODES:
        variants = [("all", cnt[code].sum(0), up[code].sum(0))]
        variants += [(str(s + 1), cnt[code][s], up[code][s]) for s in range(NSPR)]
        for tag, n, k in variants:
            for i in range(NB + 1):
                if n[i] > 0:
                    binrows.append(dict(code=code, sprstate=tag, x=centers[i],
                                        n=int(n[i]), n_up=int(k[i])))
            a, b, r2 = fit_logit(centers, n, k)
            pos, neg = centers > 0, centers < 0
            nhit = k[pos].sum() + (n[neg] - k[neg]).sum()
            ntot = n[pos].sum() + n[neg].sum()
            resrows.append(dict(code=code, sprstate=tag, a=a, b=b, pseudoR2=r2,
                                hitrate=nhit / ntot if ntot else np.nan, n=int(n.sum())))
    pd.DataFrame(binrows).to_csv(f"{D}/qi_bins{SUF}.csv", index=False)
    pd.DataFrame(resrows).to_csv(f"{D}/qi_results{SUF}.csv", index=False)
    pd.DataFrame(monrows).to_csv(f"{D}/qi_permonth{SUF}.csv", index=False)
    print(f"saved qi_bins{SUF}.csv qi_results{SUF}.csv qi_permonth{SUF}.csv")
