"""exp2_burst_ddb.py — EXP2 volume burst, ratio = V_t2 / V_t1  (user 2026-06).
spike = V_t2 / V_t1，其中 V_t1=过去 t1 笔成交量总和，V_t2=过去 t2 笔总和，t2 窗口【包含】t1。
=> spike ∈ [1, ∞)，基线（均匀成交）= t2/t1。 越接近 1 → 长窗的量几乎全集中在最近 t1 笔
   = 一次【放量爆发】。事件(峰值) = spike < r，r 取接近 1 的值（越小越极端）。
ALL = 全样本基线。

峰值之后看未来 x 笔：
  absmove = |mid[i+x]-mid[i]|              (波动率：平均绝对位移)
  signed  = sign(mid[i]-mid[i-t1])*(mid[i+x]-mid[i])   (>0延续 <0反转)
  fvol    = 未来 x 笔的每 tick 平均成交量   (用来识别“安静期假爆发”：真爆发 fvol 高)
Reuses fetch/session_blocks/WINDOWS from im_followthrough_ddb (IM only).
Output exp2_burst.csv: code,t1,t2,r,h,n_peak, a_sum,a_ss, g_sum,g_ss, v_sum,v_ss
Run: /Users/zhuisabella/xn/.venv/bin/python exp2_burst_ddb.py   (sandbox OFF)
"""
import calendar
import numpy as np
import pandas as pd
import dolphindb as ddb
from ddb_config import HOST, PORT, USER, PW
from im_followthrough_ddb import fetch, session_blocks, WINDOWS

CODE = "IM0000"
# baseline of V_t2/V_t1 = t2/t1 → keep it ≈5 so all pairs are comparable.
#   scale family  (ratio=5, vary window size): (4,20)(8,40)(12,60)(20,100)(30,150)
#   ratio family  (t1=10, ratio 4/5/6 "5上下"): (10,40)(10,50)(10,60)
PAIRS = [(4, 20), (8, 40), (12, 60), (20, 100), (30, 150), (10, 40), (10, 50), (10, 60)]
RS = [1.5, 2.0, 3.0, 4.0]                 # burst thresholds (baseline=5): peak = V_t2/V_t1 < r
HS2 = [10, 20, 60]
OUT = "/Users/zhuisabella/xn/future/exp2_burst.csv"


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
            ratio = v2 / v1                          # V_t2 / V_t1 ∈ [1, ∞)
        for h in HS2:
            lo, hi = t2, L - h
            if hi <= lo:
                continue
            i = np.arange(lo, hi)
            rr = ratio[i]
            absmove = np.abs(mid[i + h] - mid[i])
            pdir = np.sign(mid[i] - mid[i - t1])
            signed = pdir * (mid[i + h] - mid[i])
            fvol = (cs[i + h + 1] - cs[i + 1]) / h
            for r in (["ALL"] + RS):
                m = np.ones(len(i), bool) if r == "ALL" else rr < r   # burst = ratio < r
                if not m.any():
                    continue
                am, gm, vm = absmove[m], signed[m], fvol[m]
                a = acc.setdefault((t1, t2, r, h), [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
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
    rows = [{"code": CODE, "t1": t1, "t2": t2, "r": r, "h": h, "n_peak": a[0],
             "a_sum": a[1], "a_ss": a[2], "g_sum": a[3], "g_ss": a[4],
             "v_sum": a[5], "v_ss": a[6]}
            for (t1, t2, r, h), a in acc.items()]
    pd.DataFrame(rows).sort_values(["t1", "t2", "h"]).to_csv(OUT, index=False)
    print(f"saved {OUT}  ({len(rows)} rows)")
