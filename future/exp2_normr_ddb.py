"""exp2_normr_ddb.py — EXP2 volume burst, NORM ratio only, extended thresholds up to r=5.
norm spike = (V_t1/t1)/(V_t2/t2)（每tick平均量比，基线=1）；事件 = spike>r。
看未来 x=20 tick 的方向 signed=sign(mid[i]-mid[i-t1])*(mid[i+x]-mid[i]) 及反转计数。
价格去坏tick（despike，源头 bid/ask>0 在 im_followthrough.fetch 没有，这里仅 despike；
 与 future/ 既有口径一致）。IM 中证1000, 2022-07..2026-05.
Output exp2_normr.csv: t1,t2,r,x, n_peak, g_sum,g_ss, rhits(signed<0)
Run: /Users/zhuisabella/xn/.venv/bin/python exp2_normr_ddb.py   (sandbox OFF)
"""
import calendar
import numpy as np
import pandas as pd
import dolphindb as ddb
from ddb_config import HOST, PORT, USER, PW
from im_followthrough_ddb import fetch, session_blocks, WINDOWS

CODE = "IM0000"
T1S = [5, 10, 20, 30]
RATIOS = [2, 4, 8]
PAIRS = [(t1, t1 * r) for t1 in T1S for r in RATIOS]
RS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
X = 20
CAP = 4.0
OUT = "/Users/zhuisabella/xn/future/exp2_normr.csv"


def despike(mid):
    m = mid.astype(float).copy()
    for _ in range(2):
        if len(m) < 3:
            break
        dp = m[1:-1] - m[:-2]; dn = m[1:-1] - m[2:]
        spk = (np.abs(dp) > CAP) & (np.abs(dn) > CAP) & (np.sign(dp) == np.sign(dn))
        idx = np.where(spk)[0] + 1
        if not idx.size:
            break
        m[idx] = 0.5 * (m[idx - 1] + m[idx + 1])
    return m


def exp2(mid, vol, acc):
    mid = despike(mid)
    L = len(mid)
    cs = np.concatenate([[0.0], np.cumsum(vol)])
    def trailsum(t):
        s = np.full(L, np.nan)
        s[t - 1:] = cs[t:L + 1] - cs[0:L - t + 1]
        return s
    for (t1, t2) in PAIRS:
        v1, v2 = trailsum(t1), trailsum(t2)
        with np.errstate(divide="ignore", invalid="ignore"):
            spike = (v1 / t1) / (v2 / t2)
        lo, hi = t2, L - X
        if hi <= lo:
            continue
        i = np.arange(lo, hi)
        signed = np.sign(mid[i] - mid[i - t1]) * (mid[i + X] - mid[i])
        spi = spike[i]
        for r in RS:
            m = spi > r
            if not m.any():
                continue
            g = signed[m]
            a = acc.setdefault((t1, t2, r), [0, 0.0, 0.0, 0])
            a[0] += int(g.size); a[1] += g.sum(); a[2] += (g * g).sum()
            a[3] += int((g < 0).sum())


if __name__ == "__main__":
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    acc = {}
    for yr, mo in WINDOWS:
        last = calendar.monthrange(yr, mo)[1]
        start, end = f"{yr}.{mo:02d}.01", f"{yr}.{mo:02d}.{last:02d}"
        df = fetch(sess, CODE, start, end)
        if not len(df):
            continue
        for mid, vol in session_blocks(df):
            if len(mid) > max(p[1] for p in PAIRS) + X:
                exp2(mid, vol, acc)
        print(f"{yr}-{mo:02d} done", flush=True)
    sess.close()
    rows = [{"t1": t1, "t2": t2, "r": r, "x": X, "n_peak": a[0],
             "g_sum": a[1], "g_ss": a[2], "rhits": a[3]}
            for (t1, t2, r), a in acc.items()]
    pd.DataFrame(rows).sort_values(["t1", "t2", "r"]).to_csv(OUT, index=False)
    print(f"saved {OUT} ({len(rows)})")
