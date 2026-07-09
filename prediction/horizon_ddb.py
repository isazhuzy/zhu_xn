"""horizon_ddb.py — extend the two axes: how far BACK we look (accumulated flow) and how
far FORWARD we predict (up to 15 min). Answers "does order-flow predictability survive
past the 60 s where VOI died?" Expected: direction predictability → ~0 at minutes, but
accumulated (longer-window) flow may keep a thin tail.

Base bar = 10 s (20 snapshots). Predictor = net OFI accumulated over the last L bars.
Target   = mid change over the next H bars (ticks). Grid over (L, H); pooled OLS, frozen
on train (…2024-12), scored out-of-sample (2025-01…2026-05).
  lookback L ∈ {1, 6, 30} bars   = 10 s, 1 min, 5 min   (accumulated flow window)
  horizon  H ∈ {1, 6, 30, 90} bars = 10 s, 1 min, 5 min, 15 min
Outputs: horizon_grid.csv (code, look_s, hor_s, r2_is, r2_oos, slope, n)
Run: /Users/zhuisabella/xn/.venv/bin/python horizon_ddb.py   (sandbox OFF)
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
BAR = 10                                       # base bar seconds
LOOK = {1: "10s", 6: "1min", 30: "5min"}       # accumulated-flow lookback (bars)
HOR = {1: "10s", 6: "1min", 30: "5min", 90: "15min"}   # forward horizon (bars)
TREND = (2024, 12)
MONTHS = ([(2024, 11), (2025, 1)] if PILOT else
          [(y, m) for y in range(2020, 2027) for m in range(1, 13) if (2020, 1) <= (y, m) <= (2026, 5)])


def bars(df):
    b = (df["ts"].astype("int64") // 10**9 // BAR).astype("int64")
    return df.assign(bar=b).groupby(["gid", "bar"], sort=True).agg(
        ofi=("ofi", "sum"), mid=("mid_tk", "last")).reset_index()


def r2(mom, beta=None):
    n, Sx, Sy, Sxy, Sxx, Syy = mom
    if n < 500:
        return np.nan, None
    if beta is None:
        b = (n * Sxy - Sx * Sy) / (n * Sxx - Sx * Sx)
        a = (Sy - b * Sx) / n
        beta = (a, b)
    a, b = beta
    sse = Syy - 2 * a * Sy - 2 * b * Sxy + a * a * n + 2 * a * b * Sx + b * b * Sxx
    sst = Syy - Sy * Sy / n
    return (1 - sse / sst if sst > 0 else np.nan), beta


if __name__ == "__main__":
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    acc = {}                          # (code,L,H,phase) -> moments[6]
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
            bb = bars(add_ofi(df))
            for gid, g in bb.groupby("gid", sort=False):
                g = g.set_index("bar")
                idx = np.arange(g.index.min(), g.index.max() + 1)
                ofi = g["ofi"].reindex(idx).fillna(0.0).to_numpy(float)
                mid = g["mid"].reindex(idx).ffill().to_numpy(float)
                cum = {L: pd.Series(ofi).rolling(L).sum().to_numpy() for L in LOOK}
                for H in HOR:
                    y = np.full(len(mid), np.nan)
                    y[:-H] = mid[H:] - mid[:-H]
                    for L in LOOK:
                        x = cum[L]
                        m = np.isfinite(x) & np.isfinite(y)
                        if m.sum() == 0:
                            continue
                        xv, yv = x[m], y[m]
                        k = (code, L, H, phase)
                        acc[k] = acc.get(k, np.zeros(6)) + [len(xv), xv.sum(), yv.sum(),
                                                            (xv * yv).sum(), (xv * xv).sum(), (yv * yv).sum()]
        print(f"{yr}-{mo:02d} done ({phase})", flush=True)
    sess.close()

    rows = []
    for code in CODES:
        for L in LOOK:
            for H in HOR:
                tr = acc.get((code, L, H, "IS")); te = acc.get((code, L, H, "OOS"))
                if tr is None:
                    continue
                r2is, beta = r2(tr)
                r2oos, _ = r2(te, beta) if te is not None and beta is not None else (np.nan, None)
                rows.append(dict(code=code, look_bars=L, look=LOOK[L], hor_bars=H, hor=HOR[H],
                                 r2_is=r2is, r2_oos=r2oos, slope=beta[1] if beta else np.nan,
                                 n_oos=int(te[0]) if te is not None else 0))
    pd.DataFrame(rows).to_csv(f"{D}/horizon_grid{SUF}.csv", index=False)
    print(f"saved horizon_grid{SUF}.csv")
