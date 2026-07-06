"""voi_scatter_ddb.py — the PREDICTIVE relationship, visualized (paper #3).
Answers "上半秒 VOI 预测下半秒" directly and simply:
  x = VOI_t   (net queue change over the PREVIOUS 500ms snapshot)
  y = y1_t = mid_{t+1} - mid_t   (mid change over the NEXT 500ms), in ticks
Over the OUT-OF-SAMPLE window 2025-01..2026-05 only (so this is honest predictive info,
not in-sample fit). Two outputs per contract:
  1) single-factor OLS y1~VOI: exact pooled slope + R2 from accumulated moments
     (this is the "1 factor, no lags, next 0.5s" number — the barest predictive test)
  2) binned scatter: mean y1 in each VOI bin -> the visible predictive curve
     = pure signal value in ticks, BEFORE any trading cost (i.e. non-tradeable info)
Run: /Users/zhuisabella/xn/.venv/bin/python voi_scatter_ddb.py   (sandbox OFF)
"""
import calendar
import numpy as np, pandas as pd, dolphindb as ddb
from ddb_config import HOST, PORT, USER, PW
from lob_common import CODES, TICK, fetch_l1, prep_l1

D = "/Users/zhuisabella/xn/prediction"
MONTHS = [(y, m) for y in (2025, 2026) for m in range(1, 13) if (2025, 1) <= (y, m) <= (2026, 5)]
BW, CAP = 4, 160                          # VOI bin width / clip range (contracts)


def voi_and_y1(df):
    g = df.groupby("gid", sort=False)
    pbi = np.rint(df.pb / TICK); pai = np.rint(df.pa / TICK)
    pb1 = g["pb"].shift(1); pa1 = g["pa"].shift(1)
    pb1i = np.rint(pb1 / TICK); pa1i = np.rint(pa1 / TICK)
    qb1 = g["qb"].shift(1); qa1 = g["qa"].shift(1)
    dvb = np.where(pbi < pb1i, 0.0, np.where(pbi == pb1i, df.qb - qb1, df.qb))
    dva = np.where(pai > pa1i, 0.0, np.where(pai == pa1i, df.qa - qa1, df.qa))
    voi = np.where(pb1.notna().to_numpy(), dvb - dva, np.nan)
    y1 = (g["mid_tk"].shift(-1) - df["mid_tk"])
    y1 = y1.where(y1.abs() <= 50).to_numpy()
    return voi, y1


if __name__ == "__main__":
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    mom = {c: np.zeros(6) for c in CODES}          # n,Sx,Sy,Sxy,Sxx,Syy
    scat = {c: {} for c in CODES}                  # bin -> [sum_y1, count]
    for yr, mo in MONTHS:
        last = calendar.monthrange(yr, mo)[1]
        for code in CODES:
            df = fetch_l1(sess, code, f"{yr}.{mo:02d}.01", f"{yr}.{mo:02d}.{last:02d}")
            if not len(df):
                continue
            df = prep_l1(df)
            if df.empty:
                continue
            x, y = voi_and_y1(df)
            m = np.isfinite(x) & np.isfinite(y)
            x, y = x[m], y[m]
            mom[code] += [len(x), x.sum(), y.sum(), (x * y).sum(), (x * x).sum(), (y * y).sum()]
            xin = x[np.abs(x) <= CAP]; yin = y[np.abs(x) <= CAP]
            b = (np.round(xin / BW) * BW).astype(int)
            for bc, yy in zip(b, yin):
                s = scat[code].setdefault(int(bc), [0.0, 0])
                s[0] += yy; s[1] += 1
        print(f"{yr}-{mo:02d} done", flush=True)
    sess.close()

    rows = []
    for code in CODES:
        n, Sx, Sy, Sxy, Sxx, Syy = mom[code]
        slope = (n * Sxy - Sx * Sy) / (n * Sxx - Sx * Sx)
        r2 = (n * Sxy - Sx * Sy) ** 2 / ((n * Sxx - Sx * Sx) * (n * Syy - Sy * Sy))
        rows.append(dict(code=code, n=int(n), slope_tick_per_lot=slope, r2_single=r2))
    pd.DataFrame(rows).to_csv(f"{D}/voi_single.csv", index=False)
    pd.DataFrame([{"code": c, "voi": b, "mean_y1": v[0] / v[1], "n": v[1]}
                  for c in CODES for b, v in scat[c].items() if v[1] >= 100]
                 ).to_csv(f"{D}/voi_scatter.csv", index=False)
    print("saved voi_single.csv voi_scatter.csv")
