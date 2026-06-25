"""exp2_ratio_ddb.py — EXP2 volume peak, REDEFINED (user request 2026-06):
峰值 = 原始总量之比  spike = V_t1 / V_t2，其中 V_t1=过去 t1 笔成交量总和，
V_t2=过去 t2 笔总和，且 t2 窗口【包含】t1（都是到 i 为止的尾窗，t2>t1）。
=> spike ∈ (0,1] = “过去 t2 的总量里，集中在最近 t1 的比例”。 (NOT per-tick averaged.)
峰值阈值 r 用集中度比例：r∈{0.3,0.5,0.7}；ALL=全样本基线。

峰值之后看未来 x 笔：
  absmove = |mid[i+x]-mid[i]|              (波动率：平均绝对位移)
  signed  = sign(mid[i]-mid[i-t1])*(mid[i+x]-mid[i])   (>0延续 <0反转)
  fvol    = 未来 x 笔的每 tick 平均成交量
Reuses fetch/session_blocks/WINDOWS from im_followthrough_ddb (IM only).
Output exp2_ratio.csv: code,t1,t2,r,h,n_peak, a_sum,a_ss, g_sum,g_ss, v_sum,v_ss
Run: /Users/zhuisabella/xn/.venv/bin/python exp2_ratio_ddb.py   (sandbox OFF)
"""
import calendar
import numpy as np
import pandas as pd
import dolphindb as ddb
from ddb_config import HOST, PORT, USER, PW
from im_followthrough_ddb import fetch, session_blocks, WINDOWS

CODE = "IM0000"
PAIRS = [(5, 30), (5, 60), (5, 120), (10, 60), (10, 120), (20, 120)]   # (t1,t2), t2⊇t1
RS = [0.3, 0.5, 0.7]                      # concentration thresholds (fraction of V_t2)
HS2 = [10, 20, 60]                        # forward horizons (ticks)
OUT = "/Users/zhuisabella/xn/future/exp2_ratio.csv"


def exp2(mid, vol, acc):
    L = len(mid)
    cs = np.concatenate([[0.0], np.cumsum(vol)])     # cs[k]=sum vol[:k]
    def trailsum(t):                                 # raw sum over last t ticks (incl i)
        s = np.full(L, np.nan)
        s[t - 1:] = cs[t:L + 1] - cs[0:L - t + 1]
        return s
    for (t1, t2) in PAIRS:
        v1, v2 = trailsum(t1), trailsum(t2)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = v1 / v2                          # raw totals, t2 includes t1
        for h in HS2:
            lo, hi = t2, L - h                       # i>=t2 drops opening-auction tick
            if hi <= lo:
                continue
            i = np.arange(lo, hi)
            rr = ratio[i]
            absmove = np.abs(mid[i + h] - mid[i])
            pdir = np.sign(mid[i] - mid[i - t1])
            signed = pdir * (mid[i + h] - mid[i])
            fvol = (cs[i + h + 1] - cs[i + 1]) / h
            for r in (["ALL"] + RS):
                m = np.ones(len(i), bool) if r == "ALL" else rr > r
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
