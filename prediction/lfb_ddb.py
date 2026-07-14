"""lfb_ddb.py — the synthesized 2D view: OOS R^2 over (look-BACK L) x (look-FORWARD H).
Input X_L(t) = factor accumulated over the last L bars (VOI:sum, OIR/MPB:mean).
Target y_H(t) = M(t+H) - M(t). Regress y_H ~ X_L, single factor, OOS 2025-26.
Grid: L in {2s,20s,1min,5min}, H in {2s,10s,1min,5min}, for VOI/OIR/MPB.
Base bar = 2s. Output lfb_grid.csv (code,factor,look_s,hor_s,r2_oos).
Run: /Users/zhuisabella/xn/.venv/bin/python lfb_ddb.py   (sandbox OFF)
"""
import calendar
import os
import numpy as np, pandas as pd, dolphindb as ddb
from ddb_config import HOST, PORT, USER, PW
from lob_common import CODES, MULT, TICK, fetch_l1, prep_l1
from lookback_ddb import snap_factors, bars, r2

PILOT = os.environ.get("PILOT") == "1"
SUF = "_pilot" if PILOT else ""
D = "/Users/zhuisabella/xn/prediction"
LOOK = {1: "2s", 10: "20s", 30: "1min", 150: "5min"}          # look-back (bars, 2s each)
HOR = {1: "2s", 5: "10s", 30: "1min", 150: "5min"}            # look-forward (bars)
TREND = (2024, 12)


if __name__ == "__main__":
    months = [(2024, 11), (2025, 1)] if PILOT else [
        (y, m) for y in range(2020, 2027) for m in range(1, 13) if (2020, 1) <= (y, m) <= (2026, 5)]
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    acc = {}                                  # (code,factor,L,H,phase) -> moments[6]
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
            bb = bars(snap_factors(df, code))
            for gid, g in bb.groupby("gid", sort=False):
                g = g.set_index("bar")
                idx = np.arange(g.index.min(), g.index.max() + 1)
                voi = pd.Series(g["voi"].reindex(idx).fillna(0.0).to_numpy())
                oir = pd.Series(g["oir"].reindex(idx).ffill().to_numpy())
                mpb = pd.Series(g["mpb"].reindex(idx).fillna(0.0).to_numpy())
                mid = g["mid"].reindex(idx).ffill().to_numpy(float)
                Xs = {"VOI": {L: voi.rolling(L).sum().to_numpy() for L in LOOK},
                      "OIR": {L: oir.rolling(L).mean().to_numpy() for L in LOOK},
                      "MPB": {L: mpb.rolling(L).mean().to_numpy() for L in LOOK}}
                Ys = {}
                for H in HOR:
                    y = np.full(len(mid), np.nan); y[:-H] = mid[H:] - mid[:-H]; Ys[H] = y
                for fac in Xs:
                    for L in LOOK:
                        x = Xs[fac][L]
                        for H in HOR:
                            y = Ys[H]; m = np.isfinite(x) & np.isfinite(y)
                            xv, yv = x[m], y[m]
                            k = (code, fac, L, H, phase)
                            acc[k] = acc.get(k, np.zeros(6)) + [len(xv), xv.sum(), yv.sum(),
                                                                (xv * yv).sum(), (xv * xv).sum(), (yv * yv).sum()]
        print(f"{yr}-{mo:02d} done ({phase})", flush=True)
    sess.close()

    rows = []
    for code in CODES:
        for fac in ("VOI", "OIR", "MPB"):
            for L in LOOK:
                for H in HOR:
                    tr = acc.get((code, fac, L, H, "IS")); te = acc.get((code, fac, L, H, "OOS"))
                    if tr is None:
                        continue
                    _, beta = r2(tr)
                    r2oos, _ = r2(te, beta) if te is not None and beta is not None else (np.nan, None)
                    rows.append(dict(code=code, factor=fac, look=LOOK[L], hor=HOR[H], r2_oos=r2oos))
    pd.DataFrame(rows).to_csv(f"{D}/lfb_grid{SUF}.csv", index=False)
    print(f"saved lfb_grid{SUF}.csv")
