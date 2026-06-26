"""exp2_sweep_ddb.py — EXP2 clean sweeps: fix t1 sweep t2, and fix t2 sweep t1,
each under TWO ratio calc methods (user 2026-06):
  method 'norm'  = (V_t1/t1)/(V_t2/t2)   每tick平均量之比, baseline 1, peak = spike>r (r=1.5/2/3)
  method 'total' = V_t1/V_t2             原始总量占比 ∈(0,1], peak = spike>r (r=0.3/0.5/0.7)
(V_t2/V_t1 is just the reciprocal of 'total' — same events, not recomputed.)
t2 includes t1; windows inside (day,session) blocks; i>=t2 drops opening auction.
Peak-then-forward (x∈{10,20,60} ticks):
  absmove=|mid[i+x]-mid[i]|  signed=sign(mid[i]-mid[i-t1])*(mid[i+x]-mid[i])  fvol=fwd vol/tick
Output exp2_sweep.csv: method,t1,t2,r,h,n_peak, a_sum,a_ss, g_sum,g_ss, v_sum,v_ss
Run: /Users/zhuisabella/xn/.venv/bin/python exp2_sweep_ddb.py   (sandbox OFF)
"""
import calendar
import numpy as np
import pandas as pd
import dolphindb as ddb
from ddb_config import HOST, PORT, USER, PW
from im_followthrough_ddb import fetch, session_blocks, WINDOWS

CODE = "IM0000"
PAIRS_A = [(10, t2) for t2 in [20, 30, 40, 60, 90, 120, 180]]   # fix t1=10, sweep t2
PAIRS_B = [(t1, 120) for t1 in [5, 10, 20, 30, 40, 60]]         # fix t2=120, sweep t1
PAIRS = sorted(set(PAIRS_A + PAIRS_B))
HS2 = [10, 20, 60]
METHODS = [("norm", [1.5, 2.0, 3.0]), ("total", [0.3, 0.5, 0.7])]
OUT = "/Users/zhuisabella/xn/future/exp2_sweep.csv"


def exp2(mid, vol, acc):
    L = len(mid)
    cs = np.concatenate([[0.0], np.cumsum(vol)])
    def trailsum(t):
        s = np.full(L, np.nan)
        s[t - 1:] = cs[t:L + 1] - cs[0:L - t + 1]
        return s
    for (t1, t2) in PAIRS:
        v1, v2 = trailsum(t1), trailsum(t2)
        with np.errstate(divide="ignore", invalid="ignore"):
            sp_norm = (v1 / t1) / (v2 / t2)
            sp_total = v1 / v2
        spikes = {"norm": sp_norm, "total": sp_total}
        for h in HS2:
            lo, hi = t2, L - h
            if hi <= lo:
                continue
            i = np.arange(lo, hi)
            absmove = np.abs(mid[i + h] - mid[i])
            signed = np.sign(mid[i] - mid[i - t1]) * (mid[i + h] - mid[i])
            fvol = (cs[i + h + 1] - cs[i + 1]) / h
            for method, RS in METHODS:
                sp = spikes[method][i]
                for r in (["ALL"] + RS):
                    m = np.ones(len(i), bool) if r == "ALL" else sp > r
                    if not m.any():
                        continue
                    am, gm, vm = absmove[m], signed[m], fvol[m]
                    a = acc.setdefault((method, t1, t2, r, h),
                                       [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    a[0] += int(am.size)
                    a[1] += am.sum(); a[2] += (am * am).sum()
                    a[3] += gm.sum(); a[4] += (gm * gm).sum()
                    a[5] += vm.sum(); a[6] += (vm * vm).sum()


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
            if len(mid) > max(p[1] for p in PAIRS) + max(HS2):
                exp2(mid, vol, acc)
        print(f"{yr}-{mo:02d} done", flush=True)
    sess.close()
    rows = [{"method": me, "t1": t1, "t2": t2, "r": r, "h": h, "n_peak": a[0],
             "a_sum": a[1], "a_ss": a[2], "g_sum": a[3], "g_ss": a[4],
             "v_sum": a[5], "v_ss": a[6]}
            for (me, t1, t2, r, h), a in acc.items()]
    pd.DataFrame(rows).sort_values(["method", "t1", "t2", "h"]).to_csv(OUT, index=False)
    print(f"saved {OUT}  ({len(rows)} rows)")
