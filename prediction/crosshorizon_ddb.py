"""crosshorizon_ddb.py — does CROSS-contract prediction survive at LONGER horizons?
Mirror of fig110 but for the cross-OFI model. 10s bars, 4 contracts synchronized.
Input = each contract's OFI accumulated over the last 1min (6 bars). For each target i,
predict its future return over H bars, own-only vs full (all 4 OFIs). OOS R² by horizon
H in {20s,1min,5min,15min,20min}. Hypothesis: single-contract direction dies by 60s, but
the common-factor cross-signal may carry further.
Window 2022-07..2026-05; train <=2024-12, test 2025-26.
Outputs: crosshor_results.csv (target, hor_s, model, r2_oos, r2_is, n)
Run: /Users/zhuisabella/xn/.venv/bin/python crosshorizon_ddb.py   (sandbox OFF)
"""
import calendar
import os
import numpy as np, pandas as pd, dolphindb as ddb
from ddb_config import HOST, PORT, USER, PW
from lob_common import CODES, fetch_l1, prep_l1
from crossimpact_ddb import add_ofi

PILOT = os.environ.get("PILOT") == "1"
SUF = "_pilot" if PILOT else ""
D = "/Users/zhuisabella/xn/prediction"
BAR, LB = 10, 6                                  # 10s bars; accumulate OFI over 6 bars = 1min
HOR = {2: "20s", 6: "1min", 30: "5min", 90: "15min", 120: "20min"}
TREND = (2024, 12)
NC = len(CODES)


def per_bin(df):
    b = (df["ts"].astype("int64") // 10**9 // BAR).astype("int64")
    return df.assign(bar=b).groupby(["gid", "bar"], sort=True).agg(
        ofi=("ofi", "sum"), mid=("mid_tk", "last")).reset_index()


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
    months = [(2024, 11), (2025, 1)] if PILOT else [
        (y, m) for y in range(2022, 2027) for m in range(1, 13) if (2022, 7) <= (y, m) <= (2026, 5)]
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    P = 1 + NC
    acc = {}                                # (target,H,phase) -> dict  design=[1,cOFI_IC,IF,IH,IM]
    beta_tr = {}                            # (target,H,model) -> frozen train beta
    for yr, mo in months:
        phase = "train" if (yr, mo) <= TREND else "test"
        if phase == "test" and not beta_tr:
            for (t, H, ph), s in list(acc.items()):
                if ph == "train" and s["n"] >= 500:
                    _, bf = r2_sub(s["XtX"], s["Xty"], s["yty"], s["n"], list(range(P)))
                    it = CODES.index(t)
                    _, bo = r2_sub(s["XtX"], s["Xty"], s["yty"], s["n"], [0, 1 + it])
                    beta_tr[(t, H, "full")] = bf; beta_tr[(t, H, "own")] = bo
        last = calendar.monthrange(yr, mo)[1]
        pb = {}
        for c in CODES:
            df = fetch_l1(sess, c, f"{yr}.{mo:02d}.01", f"{yr}.{mo:02d}.{last:02d}")
            if len(df):
                df = prep_l1(df)
            pb[c] = per_bin(add_ofi(df)) if len(df) and not df.empty else pd.DataFrame(columns=["gid", "bar", "ofi", "mid"])
        gids = pd.unique(pd.concat([pb[c]["gid"] for c in CODES]))
        for gid in gids:
            present = {c: pb[c][pb[c].gid == gid].set_index("bar") for c in CODES if (pb[c].gid == gid).any()}
            if len(present) < NC:
                continue
            lo = min(f.index.min() for f in present.values()); hi = max(f.index.max() for f in present.values())
            idx = np.arange(lo, hi + 1)
            cof = {}; mids = {}
            for c in CODES:
                f = present[c]
                ofi = f["ofi"].reindex(idx).fillna(0.0)
                cof[c] = ofi.rolling(LB).sum().to_numpy()        # OFI accumulated over last 1min
                mids[c] = f["mid"].reindex(idx).ffill().to_numpy(float)
            X = np.column_stack([np.ones(len(idx))] + [cof[c] for c in CODES])
            for it, tc in enumerate(CODES):
                m0 = mids[tc]
                for H in HOR:
                    y = np.full(len(m0), np.nan); y[:-H] = m0[H:] - m0[:-H]
                    m = np.isfinite(y) & np.isfinite(X).all(axis=1)
                    if m.sum() < 30:
                        continue
                    Xm, ym = X[m], y[m]
                    k = (tc, H, phase)
                    s = acc.setdefault(k, dict(XtX=np.zeros((P, P)), Xty=np.zeros(P), yty=0.0, n=0))
                    s["XtX"] += Xm.T @ Xm; s["Xty"] += Xm.T @ ym; s["yty"] += ym @ ym; s["n"] += len(ym)
        print(f"{yr}-{mo:02d} done ({phase})", flush=True)
    sess.close()
    if not beta_tr:
        for (t, H, ph), s in acc.items():
            if ph == "train" and s["n"] >= 500:
                _, bf = r2_sub(s["XtX"], s["Xty"], s["yty"], s["n"], list(range(P)))
                it = CODES.index(t); _, bo = r2_sub(s["XtX"], s["Xty"], s["yty"], s["n"], [0, 1 + it])
                beta_tr[(t, H, "full")] = bf; beta_tr[(t, H, "own")] = bo

    rows = []
    for tc in CODES:
        it = CODES.index(tc)
        for H in HOR:
            tr = acc.get((tc, H, "train")); te = acc.get((tc, H, "test"))
            for model, cols in [("own", [0, 1 + it]), ("full", list(range(P)))]:
                b = beta_tr.get((tc, H, model))
                r2is = r2_sub(tr["XtX"], tr["Xty"], tr["yty"], tr["n"], cols)[0] if tr else np.nan
                r2oos = r2_sub(te["XtX"], te["Xty"], te["yty"], te["n"], cols, beta=b)[0] if (te and b is not None) else np.nan
                rows.append(dict(target=tc[:2], hor=HOR[H], hor_s={2:20,6:60,30:300,90:900,120:1200}[H],
                                 model=model, r2_is=r2is, r2_oos=r2oos, n_test=te["n"] if te else 0))
    pd.DataFrame(rows).to_csv(f"{D}/crosshor_results{SUF}.csv", index=False)
    print(f"saved crosshor_results{SUF}.csv")
