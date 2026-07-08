"""xc_icim_ddb.py — focused IC↔IM (small-cap pair) cross-predict, in-sample vs out-of-sample.
Zooms into the pair that dominated the fig103 beta matrix. For each target ∈ {IC, IM}:
  own model :  r_i(t+1) ~ 1 + OFI_i(t)
  pair model:  r_i(t+1) ~ 1 + OFI_i(t) + OFI_other(t)
Report IS R² (train), OOS R² (frozen β on test), the cross-β and its t-stat, and the
IC↔IM return lead-lag split IS/OOS. W ∈ {1,2}s (0.5s aliases; excluded).
Window 2022-07..2026-05; train ≤2024-12, test 2025-01+.
Reuses helpers from crossimpact_ddb. Outputs xc_icim_predict[_pilot].csv, xc_icim_leadlag[_pilot].csv.
Run: PILOT=1 /Users/zhuisabella/xn/.venv/bin/python xc_icim_ddb.py   (sandbox OFF)
"""
import calendar
import os
import numpy as np, pandas as pd, dolphindb as ddb
from ddb_config import HOST, PORT, USER, PW
from lob_common import fetch_l1, prep_l1
from crossimpact_ddb import add_ofi, per_bin

PILOT = os.environ.get("PILOT") == "1"
SUF = "_pilot" if PILOT else ""
D = "/Users/zhuisabella/xn/prediction"
PAIR = ["IC0000", "IM0000"]
WS = [1, 2]
LAGS = [-3, -2, -1, 0, 1, 2, 3]
TREND = (2024, 12)


def align2(perbin, gid):
    pres = {c: perbin[c][perbin[c].gid == gid].set_index("b") for c in PAIR
            if (perbin[c].gid == gid).any()}
    if len(pres) < 2:
        return None
    lo = min(f.index.min() for f in pres.values())
    hi = max(f.index.max() for f in pres.values())
    idx = np.arange(lo, hi + 1)
    out = {}
    for c in PAIR:
        f = pres[c]
        mid = f["mid"].reindex(idx).ffill().to_numpy(float)
        ofi = f["ofi"].reindex(idx).fillna(0.0).to_numpy(float)
        r = np.empty_like(mid); r[:] = np.nan; r[1:] = np.diff(mid)
        rnext = np.roll(r, -1); rnext[-1] = np.nan
        out[c] = (ofi, r, rnext)
    return out


def acc_ll(store, key, x, y):
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    if len(x):
        store[key] = store.get(key, np.zeros(6)) + [len(x), x.sum(), y.sum(),
                                                     (x * y).sum(), (x * x).sum(), (y * y).sum()]


def corr_of(v):
    n, Sx, Sy, Sxy, Sxx, Syy = v
    dx, dy = n * Sxx - Sx * Sx, n * Syy - Sy * Sy
    return ((n * Sxy - Sx * Sy) / np.sqrt(dx * dy) if dx > 0 and dy > 0 and n > 100 else np.nan), int(n)


def fit(XtX, Xty, yty, n, cols, beta=None):
    A = XtX[np.ix_(cols, cols)]; b = Xty[cols]
    if beta is None:
        beta = np.linalg.solve(A, b)
    sse = yty - 2 * beta @ b + beta @ A @ beta
    sst = yty - Xty[0] ** 2 / n
    r2 = 1 - sse / sst if sst > 0 else np.nan
    # t-stat of last coef: se = sqrt(sigma2 * (A^-1)_kk),  sigma2 = sse/(n-p)
    k = len(cols) - 1
    sigma2 = sse / (n - len(cols))
    se = np.sqrt(sigma2 * np.linalg.inv(A)[k, k])
    return r2, beta, beta[k] / se if se > 0 else np.nan


if __name__ == "__main__":
    months = [(2024, 6), (2024, 7)] if PILOT else [
        (y, m) for y in range(2022, 2027) for m in range(1, 13) if (2022, 7) <= (y, m) <= (2026, 5)]
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    ll = {}                          # (W,lag,phase) -> moments   (x=r_IC, y=r_IM shifted)
    pm = {}                          # (target,W,phase) -> dict   design=[1, OFI_target, OFI_other]
    for yr, mo in months:
        phase = "IS" if (yr, mo) <= TREND else "OOS"
        last = calendar.monthrange(yr, mo)[1]
        raw = {}
        for c in PAIR:
            df = fetch_l1(sess, c, f"{yr}.{mo:02d}.01", f"{yr}.{mo:02d}.{last:02d}")
            if len(df):
                df = prep_l1(df)
            raw[c] = add_ofi(df) if len(df) and not df.empty else None
        if any(v is None for v in raw.values()):
            print(f"{yr}-{mo:02d} skip (missing contract)", flush=True); continue
        for W in WS:
            perbin = {c: per_bin(raw[c], W) for c in PAIR}
            gids = pd.unique(pd.concat([perbin[c]["gid"] for c in PAIR]))
            for gid in gids:
                A = align2(perbin, gid)
                if A is None:
                    continue
                ric, rim = A["IC0000"][1], A["IM0000"][1]
                for L in LAGS:
                    if L >= 0:
                        x, y = ric[:len(ric) - L], rim[L:]
                    else:
                        x, y = ric[-L:], rim[:len(rim) + L]
                    acc_ll(ll, (W, L, phase), x, y)
                oic, oim = A["IC0000"][0], A["IM0000"][0]
                n = len(oic)
                for tgt, oo in [("IC0000", (oic, oim)), ("IM0000", (oim, oic))]:
                    y = A[tgt][2]; m = np.isfinite(y)
                    if m.sum() < 50:
                        continue
                    X = np.column_stack([np.ones(n), oo[0], oo[1]])[m]
                    ym = y[m]
                    k = (tgt, W, phase)
                    s = pm.setdefault(k, dict(XtX=np.zeros((3, 3)), Xty=np.zeros(3), yty=0.0, n=0))
                    s["XtX"] += X.T @ X; s["Xty"] += X.T @ ym; s["yty"] += ym @ ym; s["n"] += len(ym)
        print(f"{yr}-{mo:02d} done ({phase})", flush=True)
    sess.close()

    lr = []
    for (W, L, ph), v in sorted(ll.items()):
        c, n = corr_of(v)
        lr.append(dict(W=W, lag=L, phase=ph, corr=c, n=n))
    pd.DataFrame(lr).to_csv(f"{D}/xc_icim_leadlag{SUF}.csv", index=False)

    pr = []
    for tgt in PAIR:
        for W in WS:
            tr = pm.get((tgt, W, "IS")); te = pm.get((tgt, W, "OOS"))
            if not tr:
                continue
            r2o_is, bo, _ = fit(tr["XtX"], tr["Xty"], tr["yty"], tr["n"], [0, 1])
            r2p_is, bp, tstat = fit(tr["XtX"], tr["Xty"], tr["yty"], tr["n"], [0, 1, 2])
            r2o_oos = r2p_oos = np.nan; noos = 0
            if te:
                noos = te["n"]
                r2o_oos, _, _ = fit(te["XtX"], te["Xty"], te["yty"], te["n"], [0, 1], beta=bo)
                r2p_oos, _, _ = fit(te["XtX"], te["Xty"], te["yty"], te["n"], [0, 1, 2], beta=bp)
            pr.append(dict(target=tgt[:2], W=W, model="own", r2_is=r2o_is, r2_oos=r2o_oos,
                           cross_beta=np.nan, cross_t=np.nan, n_is=tr["n"], n_oos=noos))
            pr.append(dict(target=tgt[:2], W=W, model="pair", r2_is=r2p_is, r2_oos=r2p_oos,
                           cross_beta=bp[2], cross_t=tstat, n_is=tr["n"], n_oos=noos))
    pd.DataFrame(pr).to_csv(f"{D}/xc_icim_predict{SUF}.csv", index=False)
    print(f"saved xc_icim_predict{SUF}.csv xc_icim_leadlag{SUF}.csv")
