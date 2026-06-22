"""
reversal_regression.py — threshold-free view of the open's first two minutes.

For each product, build per-day open closes at 9:30/9:31/9:32/9:33 and study the
predictive structure of the early moves (all in basis points of the mid):

  m1 = 9:30->9:31 return     m2 = 9:31->9:32 return     m3 = 9:32->9:33 return

  REV-A: regress m2 on m1   (slope<0 => the open move reverses next minute)
  REV-B: regress m3 on m2   (does the reversal persist into minute 2?)
  LAG-2: regress m3 on m1   (does the original open move still matter 2 min out?)

Reports OLS slope (beta), its t-stat, R^2, sign-agreement, and the mean/|mean|
size of each minute's move. Runs on the small /tmp/open_ticks.csv extract.
"""
import numpy as np
import pandas as pd
from datetime import time as dtime
from first_two_minutes import open_bars, PRODUCTS, EXCLUDE_MONTHS

SRC = "/tmp/open_ticks.csv"


def closes_wide(bars):
    """day x {9:30,9:31,9:32,9:33} mid close -> early returns in bp."""
    w = bars.pivot_table(index="day", columns="tod", values="close")
    w = w[~w.index.month.isin(EXCLUDE_MONTHS)]
    c30, c31 = w[dtime(9, 30)], w[dtime(9, 31)]
    c32, c33 = w[dtime(9, 32)], w[dtime(9, 33)]
    out = pd.DataFrame({
        "m1": (c31 / c30 - 1) * 1e4,   # 9:30->9:31
        "m2": (c32 / c31 - 1) * 1e4,   # 9:31->9:32
        "m3": (c33 / c32 - 1) * 1e4,   # 9:32->9:33
    }).dropna()
    return out


def ols(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    b1, b0 = np.polyfit(x, y, 1)
    yhat = b0 + b1 * x
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot else np.nan
    se = np.sqrt(ss_res / (n - 2)) / np.sqrt(((x - x.mean()) ** 2).sum())
    t = b1 / se if se else np.nan
    corr = np.corrcoef(x, y)[0, 1]
    return dict(n=n, beta=round(b1, 3), t=round(t, 2), r2=round(r2, 3),
                corr=round(corr, 3))


if __name__ == "__main__":
    df = pd.read_csv(SRC, dtype={"code": "string"}, parse_dates=["m_nDatetime"])

    print("=== Open-move magnitude (bp of mid, March excluded) ===")
    mags = []
    data = {}
    for c in PRODUCTS:
        r = closes_wide(open_bars(df, c)); data[c] = r
        mags.append({"product": c, "n": len(r),
                     "mean_m1": round(r.m1.mean(), 2), "absmean_m1": round(r.m1.abs().mean(), 2),
                     "absmean_m2": round(r.m2.abs().mean(), 2), "absmean_m3": round(r.m3.abs().mean(), 2)})
    print(pd.DataFrame(mags).to_string(index=False))

    print("\n=== Predictive regressions (slope beta in bp-return per bp-move) ===")
    print("REV-A: m2~m1 | REV-B: m3~m2 | LAG2: m3~m1   (beta<0 => reversal)\n")
    rows = []
    for c in PRODUCTS:
        r = data[c]
        for name, (xc, yc) in {"REV-A m2~m1": ("m1", "m2"),
                               "REV-B m3~m2": ("m2", "m3"),
                               "LAG2  m3~m1": ("m1", "m3")}.items():
            rows.append({"product": c, "reg": name, **ols(r[xc], r[yc])})
    print(pd.DataFrame(rows).to_string(index=False))
