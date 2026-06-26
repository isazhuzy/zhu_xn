"""exp1_nk_clean_ddb.py — same dense n×k sweep as exp1_nk, but DE-SPIKE bad ticks first.
坏 tick = 单点尖峰：mid[i] 同时远离左右邻居 >CAP 点、且两侧同向（会回弹）→ 用邻居均值替换。
真正的多 tick 大 move 不受影响（它不是单点尖峰）。两遍清洗以处理连续 glitch。
Output exp1_nk_clean.csv (same columns as exp1_nk.csv).
Run: /Users/zhuisabella/xn/.venv/bin/python exp1_nk_clean_ddb.py   (sandbox OFF)
"""
import calendar
import numpy as np
import pandas as pd
import dolphindb as ddb
from ddb_config import HOST, PORT, USER, PW
from im_followthrough_ddb import fetch, session_blocks, WINDOWS

TICK = 0.2
CODE = "IM0000"
X = 10
NS = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 30]
KTICKS = [0, 1, 2, 3, 4, 6, 8, 10, 13, 16, 20, 30, 45, 60, 90]
KS = [kt * TICK for kt in KTICKS]
CAP = 4.0                                   # 单 tick 尖峰阈值（指数点）= 20 价格tick
OUT = "/Users/zhuisabella/xn/future/exp1_nk_clean.csv"
_stat = [0, 0]                              # [n_spikes_removed, n_ticks_total]


def despike(mid):
    m = mid.astype(float).copy()
    for _ in range(2):                      # two passes for back-to-back glitches
        if len(m) < 3:
            break
        dp = m[1:-1] - m[:-2]               # mid[i]-mid[i-1]
        dn = m[1:-1] - m[2:]                # mid[i]-mid[i+1]
        spike = (np.abs(dp) > CAP) & (np.abs(dn) > CAP) & (np.sign(dp) == np.sign(dn))
        idx = np.where(spike)[0] + 1
        if not idx.size:
            break
        m[idx] = 0.5 * (m[idx - 1] + m[idx + 1])
        _stat[0] += int(idx.size)
    return m


def exp1(mid, acc):
    L = len(mid)
    for n in NS:
        lo, hi = n, L - X
        if hi <= lo:
            continue
        i = np.arange(lo, hi)
        back = mid[i] - mid[i - n]
        fwd = mid[i + X] - mid[i]
        signed = np.sign(back) * fwd
        ab = np.abs(back)
        for kt, k in zip(KTICKS, KS):
            m = ab > k if k > 0 else np.ones(len(i), bool)
            s = signed[m]
            if not s.size:
                continue
            a = acc.setdefault((n, kt), [0.0, 0.0, 0, 0])
            a[0] += s.sum(); a[1] += (s * s).sum(); a[2] += s.size
            a[3] += int((s > 0).sum())


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
            if len(mid) > max(NS) + X:
                _stat[1] += len(mid)
                exp1(despike(mid), acc)
        print(f"{yr}-{mo:02d} done", flush=True)
    sess.close()
    rows = [{"code": CODE, "n": n, "k_ticks": kt, "k_pts": round(kt * TICK, 2), "x": X,
             "s_sum": a[0], "s_ss": a[1], "s_n": a[2], "hits": a[3]}
            for (n, kt), a in acc.items()]
    pd.DataFrame(rows).sort_values(["n", "k_ticks"]).to_csv(OUT, index=False)
    frac = 100.0 * _stat[0] / max(_stat[1], 1)
    print(f"saved {OUT}  ({len(rows)} rows);  去尖峰 {_stat[0]:,} / {_stat[1]:,} 个 tick = {frac:.4f}%")
