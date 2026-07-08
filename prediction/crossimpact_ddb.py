"""crossimpact_ddb.py — Direction A: cross-contract lead-lag & OFI cross-impact.
Paper: Cont, Cucuringu & Zhang (2023), *Cross-Impact of Order Flow Imbalance in
Equity Markets*, Quant. Finance 23(10). We ask, on CFFEX index futures IC/IF/IH/IM:
  (1) who LEADS whom  — return cross-correlation corr(r_a(t), r_b(t+ℓ)); peak at ℓ>0 ⇒ a leads b.
  (2) does cross-OFI PREDICT — r_i(t+1) ~ own OFI_i(t) + others' OFI_j(t), frozen-β OOS R².

Sync: the 4 contracts tick asynchronously, so we floor every tick to a wall-clock bin
of width W seconds; per (contract, session, bin) sum tick-OFI and take the last mid.
Stale bins ffill the mid (⇒ r=0) and get OFI=0. This puts all 4 on one clock (Cont's aggregation).
OFI (CKS L1): bid_contrib + ask_contrib from consecutive ticks within a session.

Window 2022-07..2026-05 (all 4 coexist; IM starts 2022-07). Train ≤2024-12, test 2025-01+.
Outputs: xc_leadlag[_pilot].csv  (W, a, b, lag, corr, n)
         xc_predict[_pilot].csv  (target, W, model, r2_train, r2_oos, n_train, n_test)
         xc_betas[_pilot].csv     (target, W, source, beta)   [full model, train]
Run: PILOT=1 /Users/zhuisabella/xn/.venv/bin/python crossimpact_ddb.py   (sandbox OFF)
"""
import calendar
import os
import numpy as np, pandas as pd, dolphindb as ddb
from ddb_config import HOST, PORT, USER, PW
from lob_common import CODES, fetch_l1, prep_l1

PILOT = os.environ.get("PILOT") == "1"
SUF = "_pilot" if PILOT else ""
D = "/Users/zhuisabella/xn/prediction"
WS = [0.5, 1, 2]                  # bin widths (seconds); 0.5s = our 500ms resolution floor
LAGS = [-3, -2, -1, 0, 1, 2, 3]  # lead-lag lags (bins)
TREND = (2024, 12)               # last train month
NC = len(CODES)                  # 4; own index = CODES.index(code)


def add_ofi(df):
    g = df.groupby("gid", sort=False)
    pb1, qb1 = g["pb"].shift(1), g["qb"].shift(1)
    pa1, qa1 = g["pa"].shift(1), g["qa"].shift(1)
    bid = np.where(df.pb >= pb1, df.qb, 0.0) - np.where(df.pb <= pb1, qb1, 0.0)
    ask = np.where(df.pa <= pa1, df.qa, 0.0) - np.where(df.pa >= pa1, qa1, 0.0)
    df["ofi"] = np.where(pb1.notna().to_numpy(), bid - ask, 0.0)
    return df


def per_bin(df, W):
    b = (df["ts"].astype("int64") // 10**6 // int(W * 1000)).astype("int64")  # floor to W-sec bin (ms clock)
    a = df.assign(b=b).groupby(["gid", "b"], sort=True).agg(
        ofi=("ofi", "sum"), mid=("mid_tk", "last")).reset_index()
    return a


def aligned_arrays(perbin, gid, W):
    """return dict code -> (ofi[], r[], rnext[]) on a dense common bin grid for this session."""
    present = {c: perbin[c][perbin[c].gid == gid].set_index("b")
               for c in CODES if (perbin[c].gid == gid).any()}
    if len(present) < 2:
        return None
    lo = min(f.index.min() for f in present.values())
    hi = max(f.index.max() for f in present.values())
    idx = np.arange(lo, hi + 1)
    out = {}
    for c in CODES:
        if c not in present:
            out[c] = (None, None, None)
            continue
        f = present[c]
        mid = f["mid"].reindex(idx).ffill().to_numpy(float)
        ofi = f["ofi"].reindex(idx).fillna(0.0).to_numpy(float)
        r = np.empty_like(mid); r[:] = np.nan
        r[1:] = np.diff(mid)
        rnext = np.roll(r, -1); rnext[-1] = np.nan
        out[c] = (ofi, r, rnext)
    return out


def acc_moment(store, key, x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) == 0:
        return
    store[key] = store.get(key, np.zeros(6)) + [len(x), x.sum(), y.sum(),
                                                 (x * y).sum(), (x * x).sum(), (y * y).sum()]


def corr_from(v):
    n, Sx, Sy, Sxy, Sxx, Syy = v
    dx, dy = n * Sxx - Sx * Sx, n * Syy - Sy * Sy
    if n < 100 or dx <= 0 or dy <= 0:
        return np.nan, int(n)
    return (n * Sxy - Sx * Sy) / np.sqrt(dx * dy), int(n)


def r2_sub(XtX, Xty, yty, n, cols, beta=None):
    A = XtX[np.ix_(cols, cols)]; b = Xty[cols]
    if beta is None:
        if n < 500:
            return np.nan, None
        beta = np.linalg.solve(A, b)
    sse = yty - 2 * beta @ b + beta @ A @ beta
    sst = yty - Xty[0] ** 2 / n
    return (1 - sse / sst if sst > 0 else np.nan), beta


if __name__ == "__main__":
    months = [(2024, 6), (2024, 7)] if PILOT else [
        (y, m) for y in range(2022, 2027) for m in range(1, 13)
        if (2022, 7) <= (y, m) <= (2026, 5)]
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    ll = {}                          # (W,a,b,lag) -> moment[6]
    pm = {}                          # (target, W, phase) -> dict(XtX,Xty,yty,n)  design=[1,ofi_IC,ofi_IF,ofi_IH,ofi_IM]
    beta_tr = {}                     # (target, W) -> full-model train beta
    P = 1 + NC
    for yr, mo in months:
        phase = "train" if (yr, mo) <= TREND else "test"
        if phase == "test" and not beta_tr:
            for (t, W, ph), s in list(pm.items()):
                if ph == "train":
                    _, b = r2_sub(s["XtX"], s["Xty"], s["yty"], s["n"], list(range(P)))
                    if b is not None:
                        beta_tr[(t, W)] = b
        last = calendar.monthrange(yr, mo)[1]
        raw = {}                                      # fetch each contract ONCE per month
        for c in CODES:
            df = fetch_l1(sess, c, f"{yr}.{mo:02d}.01", f"{yr}.{mo:02d}.{last:02d}")
            if len(df):
                df = prep_l1(df)
            raw[c] = add_ofi(df) if len(df) and not df.empty else None
        for W in WS:
            perbin = {c: (per_bin(raw[c], W) if raw[c] is not None else
                          pd.DataFrame(columns=["gid", "b", "ofi", "mid"])) for c in CODES}
            gids = pd.unique(pd.concat([perbin[c]["gid"] for c in CODES]))
            for gid in gids:
                A = aligned_arrays(perbin, gid, W)
                if A is None:
                    continue
                # (1) lead-lag on returns
                for ia in range(NC):
                    ra = A[CODES[ia]][1]
                    if ra is None:
                        continue
                    for ib in range(ia + 1, NC):
                        rb = A[CODES[ib]][1]
                        if rb is None:
                            continue
                        for L in LAGS:
                            if L >= 0:
                                x, y = ra[:len(ra) - L], rb[L:]
                            else:
                                x, y = ra[-L:], rb[:len(rb) + L]
                            acc_moment(ll, (W, CODES[ia], CODES[ib], L), x, y)
                # (2) OFI cross-predictive: design = [1, ofi_IC, ofi_IF, ofi_IH, ofi_IM]
                ofis = [A[c][0] for c in CODES]
                if any(o is None for o in ofis):
                    continue
                n = len(ofis[0])
                X = np.column_stack([np.ones(n)] + ofis)
                for it, tc in enumerate(CODES):
                    y = A[tc][2]                 # r_target(t+1)
                    m = np.isfinite(y)
                    if m.sum() < 50:
                        continue
                    Xm, ym = X[m], y[m]
                    k = (tc, W, phase)
                    s = pm.setdefault(k, dict(XtX=np.zeros((P, P)), Xty=np.zeros(P), yty=0.0, n=0))
                    s["XtX"] += Xm.T @ Xm; s["Xty"] += Xm.T @ ym
                    s["yty"] += ym @ ym; s["n"] += len(ym)
        print(f"{yr}-{mo:02d} done ({phase})", flush=True)
    sess.close()
    if not beta_tr:
        for (t, W, ph), s in pm.items():
            if ph == "train":
                _, b = r2_sub(s["XtX"], s["Xty"], s["yty"], s["n"], list(range(P)))
                if b is not None:
                    beta_tr[(t, W)] = b

    pd.DataFrame([dict(W=W, a=a, b=b, lag=L, corr=corr_from(v)[0], n=corr_from(v)[1])
                  for (W, a, b, L), v in sorted(ll.items())]).to_csv(f"{D}/xc_leadlag{SUF}.csv", index=False)

    prows, brows = [], []
    for tc in CODES:
        it = CODES.index(tc)
        own_cols = [0, 1 + it]
        full_cols = list(range(P))
        for W in WS:
            tr = pm.get((tc, W, "train")); te = pm.get((tc, W, "test"))
            b_full = beta_tr.get((tc, W))
            if tr:
                r2o_tr, _ = r2_sub(tr["XtX"], tr["Xty"], tr["yty"], tr["n"], own_cols)
                r2f_tr, _ = r2_sub(tr["XtX"], tr["Xty"], tr["yty"], tr["n"], full_cols)
            else:
                r2o_tr = r2f_tr = np.nan
            r2o_oos = r2f_oos = np.nan; nte = 0
            if te and tr:
                nte = te["n"]
                _, bo = r2_sub(tr["XtX"], tr["Xty"], tr["yty"], tr["n"], own_cols)
                r2o_oos, _ = r2_sub(te["XtX"], te["Xty"], te["yty"], te["n"], own_cols, beta=bo)
                if b_full is not None:
                    r2f_oos, _ = r2_sub(te["XtX"], te["Xty"], te["yty"], te["n"], full_cols, beta=b_full)
            ntr = tr["n"] if tr else 0
            prows.append(dict(target=tc, W=W, model="own", r2_train=r2o_tr, r2_oos=r2o_oos, n_train=ntr, n_test=nte))
            prows.append(dict(target=tc, W=W, model="full", r2_train=r2f_tr, r2_oos=r2f_oos, n_train=ntr, n_test=nte))
            if b_full is not None:
                for j, sc in enumerate(CODES):
                    brows.append(dict(target=tc, W=W, source=sc, beta=b_full[1 + j], own=(sc == tc)))
    pd.DataFrame(prows).to_csv(f"{D}/xc_predict{SUF}.csv", index=False)
    pd.DataFrame(brows).to_csv(f"{D}/xc_betas{SUF}.csv", index=False)
    print(f"saved xc_leadlag{SUF} xc_predict{SUF} xc_betas{SUF}")
