"""shen_grid_ddb.py — the COMBINED three-factor model on the (look-back L) x (look-forward H)
grid: fig114's 2D view, upgraded from single factors to Shen's model-B style combination.
Snapshot factors VOI/OIR/MPB (lookback_ddb.snap_factors), two normalizations:
  raw  : factor as-is
  shen : factor / spread(t)  at snapshot level (Shen's "经价差归一"), then accumulated.
Accumulate over trailing L bars (VOI:sum, OIR/MPB:mean), regress y_H = M(t+H)-M(t) on
  [1, VOI_L, OIR_L, MPB_L]  (model ALL) — single-factor R² comes free from the same moments.
Base bar 2s; L in {2s,20s,1min,5min}; H in {2s,10s,1min,5min}; train ..2024-12, OOS 2025-26.
Outputs: shen_grid.csv (code,norm,model,look,hor,look_s,hor_s,r2_is,r2_oos,n_oos)
Run: PILOT=1 /Users/zhuisabella/xn/.venv/bin/python shen_grid_ddb.py   (sandbox OFF)
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
BAR = 2                                       # 2s base bars
LOOK = {1: "2s", 10: "20s", 30: "1min", 150: "5min"}
HOR = {1: "2s", 5: "10s", 30: "1min", 150: "5min"}
TREND = (2024, 12)
NORMS = ("raw", "shen")
FACS = ("VOI", "OIR", "MPB")
P = 4                                         # const + 3 factors


def bars2(df):
    """2s bars keeping raw and spread-normalized factor aggregates."""
    for c in ("voi", "oir", "mpb"):
        df[c + "_n"] = df[c] / df["spr"]
    b = (df["ts"].astype("int64") // 10**9 // BAR).astype("int64")
    return df.assign(bar=b).groupby(["gid", "bar"], sort=True).agg(
        voi=("voi", "sum"), oir=("oir", "mean"), mpb=("mpb", "mean"),
        voi_n=("voi_n", "sum"), oir_n=("oir_n", "mean"), mpb_n=("mpb_n", "mean"),
        mid=("mid_tk", "last")).reset_index()


def r2_from(mom, beta=None):
    XtX, Xty, yty, n = mom
    if n < 500:
        return np.nan, None
    if beta is None:
        try:
            beta = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            return np.nan, None
    sse = yty - 2 * beta @ Xty + beta @ XtX @ beta
    sst = yty - Xty[0] ** 2 / n
    return (1 - sse / sst if sst > 0 else np.nan), beta


def sub(mom, idx):
    XtX, Xty, yty, n = mom
    return XtX[np.ix_(idx, idx)], Xty[idx], yty, n


if __name__ == "__main__":
    months = [(2024, 11), (2025, 1)] if PILOT else [
        (y, m) for y in range(2020, 2027) for m in range(1, 13) if (2020, 1) <= (y, m) <= (2026, 5)]
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    acc = {}                                  # (code,norm,L,H,phase) -> [XtX,Xty,yty,n]
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
            bb = bars2(snap_factors(df, code))
            for gid, g in bb.groupby("gid", sort=False):
                g = g.set_index("bar")
                idx = np.arange(g.index.min(), g.index.max() + 1)
                mid = g["mid"].reindex(idx).ffill().to_numpy(float)
                cols = {}
                for norm, s in (("raw", ""), ("shen", "_n")):
                    cols[norm] = {
                        "VOI": pd.Series(g["voi" + s].reindex(idx).fillna(0.0).to_numpy()),
                        "OIR": pd.Series(g["oir" + s].reindex(idx).ffill().to_numpy()),
                        "MPB": pd.Series(g["mpb" + s].reindex(idx).fillna(0.0).to_numpy())}
                Ys = {}
                for H in HOR:
                    y = np.full(len(mid), np.nan); y[:-H] = mid[H:] - mid[:-H]; Ys[H] = y
                for norm in NORMS:
                    for L in LOOK:
                        X3 = np.column_stack([
                            cols[norm]["VOI"].rolling(L).sum().to_numpy(),
                            cols[norm]["OIR"].rolling(L).mean().to_numpy(),
                            cols[norm]["MPB"].rolling(L).mean().to_numpy()])
                        okx = np.isfinite(X3).all(axis=1)
                        for H in HOR:
                            m = okx & np.isfinite(Ys[H])
                            if m.sum() < 100:
                                continue
                            X = np.column_stack([np.ones(int(m.sum())), X3[m]])
                            yv = Ys[H][m]
                            k = (code, norm, L, H, phase)
                            if k not in acc:
                                acc[k] = [np.zeros((P, P)), np.zeros(P), 0.0, 0]
                            a = acc[k]
                            a[0] += X.T @ X; a[1] += X.T @ yv
                            a[2] += yv @ yv; a[3] += len(yv)
        print(f"{yr}-{mo:02d} done ({phase})", flush=True)
    sess.close()

    MODELS = {"ALL": [0, 1, 2, 3], "VOI": [0, 1], "OIR": [0, 2], "MPB": [0, 3]}
    rows = []
    for code in CODES:
        for norm in NORMS:
            for L in LOOK:
                for H in HOR:
                    tr = acc.get((code, norm, L, H, "IS"))
                    te = acc.get((code, norm, L, H, "OOS"))
                    if tr is None:
                        continue
                    for mdl, ix in MODELS.items():
                        r2is, beta = r2_from(sub(tr, ix))
                        r2oos = np.nan
                        if te is not None and beta is not None:
                            r2oos, _ = r2_from(sub(te, ix), beta)
                        rows.append(dict(code=code, norm=norm, model=mdl,
                                         look=LOOK[L], hor=HOR[H],
                                         look_s=L * BAR, hor_s=H * BAR,
                                         r2_is=r2is, r2_oos=r2oos,
                                         n_oos=te[3] if te is not None else 0))
    pd.DataFrame(rows).to_csv(f"{D}/shen_grid{SUF}.csv", index=False)
    print(f"saved shen_grid{SUF}.csv")
