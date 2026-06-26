"""exp2_x_ddb.py — EXP2 burst, sweep the FORWARD window x while varying windows/ratio,
under BOTH ratio calc methods (user 2026-06). 价格已去坏tick。
  method 'norm'  = (V_t1/t1)/(V_t2/t2)   每tick平均量(量/时间)之比, peak = spike>r (1.5/2/3)
  method 'total' = V_t1/V_t2             原始总量占比, peak = spike>r (0.3/0.5/0.7)
窗口网格: t1∈{5,10,20,30} × 比率 t2/t1∈{2,4,8} → 12 对 (覆盖变t1t2 & 变比率)。
峰值后看未来 x∈{2,5,10,20,40,80,120}: absmove / signed / fvol。
Output exp2_x.csv: method,t1,t2,r,x,n_peak, a_sum,a_ss, g_sum,g_ss, v_sum,v_ss
Run: /Users/zhuisabella/xn/.venv/bin/python exp2_x_ddb.py   (sandbox OFF)
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
PAIRS = [(t1, t1 * r) for t1 in T1S for r in RATIOS]      # 12 (t1,t2) pairs
XS = [2, 5, 10, 20, 40, 80, 120]
METHODS = [("norm", [1.5, 2.0, 3.0]), ("total", [0.3, 0.5, 0.7])]
CAP = 4.0
OUT = "/Users/zhuisabella/xn/future/exp2_x.csv"


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
            sp = {"norm": (v1 / t1) / (v2 / t2), "total": v1 / v2}
        for x in XS:
            lo, hi = t2, L - x
            if hi <= lo:
                continue
            i = np.arange(lo, hi)
            absmove = np.abs(mid[i + x] - mid[i])
            signed = np.sign(mid[i] - mid[i - t1]) * (mid[i + x] - mid[i])
            fvol = (cs[i + x + 1] - cs[i + 1]) / x
            for method, RS in METHODS:
                spi = sp[method][i]
                for r in (["ALL"] + RS):
                    m = np.ones(len(i), bool) if r == "ALL" else spi > r
                    if not m.any():
                        continue
                    am, gm, vm = absmove[m], signed[m], fvol[m]
                    a = acc.setdefault((method, t1, t2, r, x),
                                       [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0])
                    a[0] += int(am.size)
                    a[1] += am.sum(); a[2] += (am * am).sum()
                    a[3] += gm.sum(); a[4] += (gm * gm).sum()
                    a[5] += vm.sum(); a[6] += (vm * vm).sum()
                    a[7] += int((gm < 0).sum())          # 反转命中数（signed<0）


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
            if len(mid) > max(p[1] for p in PAIRS) + max(XS):
                exp2(mid, vol, acc)
        print(f"{yr}-{mo:02d} done", flush=True)
    sess.close()
    rows = [{"method": me, "t1": t1, "t2": t2, "r": r, "x": x, "n_peak": a[0],
             "a_sum": a[1], "a_ss": a[2], "g_sum": a[3], "g_ss": a[4],
             "v_sum": a[5], "v_ss": a[6], "rhits": a[7]}
            for (me, t1, t2, r, x), a in acc.items()]
    pd.DataFrame(rows).sort_values(["method", "t1", "t2", "x"]).to_csv(OUT, index=False)
    print(f"saved {OUT}  ({len(rows)} rows)")
