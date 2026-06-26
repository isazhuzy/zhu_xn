"""exp1_nkx_ddb.py — EXP1 price pulse, FULL 3-D grid n × k × x (de-spiked bad ticks).
用于看不同 (n,k) 组合下、结果随向前窗口 x 的变化。
脉冲 = mid[i]−mid[i−n]，|脉冲|>k 触发；纵轴量 = sign(脉冲)×(mid[i+x]−mid[i]) 的均值。
坏 tick 清洗同 exp1_nk_clean（单点尖峰 >4 点且回弹 → 邻居均值）。
Output exp1_nkx.csv: n,k_ticks,k_pts,x, s_sum,s_ss,s_n,hits
Run: /Users/zhuisabella/xn/.venv/bin/python exp1_nkx_ddb.py   (sandbox OFF)
"""
import calendar
import numpy as np
import pandas as pd
import dolphindb as ddb
from ddb_config import HOST, PORT, USER, PW
from im_followthrough_ddb import fetch, session_blocks, WINDOWS

TICK = 0.2
CODE = "IM0000"
NS = [1, 2, 3, 5, 8, 13, 20, 25, 30]
KTICKS = [0, 1, 2, 4, 8, 16, 30, 45, 90]
KS = [kt * TICK for kt in KTICKS]
XS = [2, 5, 10, 20, 40, 80, 120]
CAP = 4.0
OUT = "/Users/zhuisabella/xn/future/exp1_nkx.csv"


def despike(mid):
    m = mid.astype(float).copy()
    for _ in range(2):
        if len(m) < 3:
            break
        dp = m[1:-1] - m[:-2]; dn = m[1:-1] - m[2:]
        spike = (np.abs(dp) > CAP) & (np.abs(dn) > CAP) & (np.sign(dp) == np.sign(dn))
        idx = np.where(spike)[0] + 1
        if not idx.size:
            break
        m[idx] = 0.5 * (m[idx - 1] + m[idx + 1])
    return m


def exp1(mid, acc):
    L = len(mid)
    for n in NS:
        for x in XS:
            lo, hi = n, L - x
            if hi <= lo:
                continue
            i = np.arange(lo, hi)
            back = mid[i] - mid[i - n]
            fwd = mid[i + x] - mid[i]
            signed = np.sign(back) * fwd
            ab = np.abs(back)
            for kt, k in zip(KTICKS, KS):
                m = ab > k if k > 0 else np.ones(len(i), bool)
                s = signed[m]
                if not s.size:
                    continue
                a = acc.setdefault((n, kt, x), [0.0, 0.0, 0, 0])
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
            if len(mid) > max(NS) + max(XS):
                exp1(despike(mid), acc)
        print(f"{yr}-{mo:02d} done", flush=True)
    sess.close()
    rows = [{"n": n, "k_ticks": kt, "k_pts": round(kt * TICK, 2), "x": x,
             "s_sum": a[0], "s_ss": a[1], "s_n": a[2], "hits": a[3]}
            for (n, kt, x), a in acc.items()]
    pd.DataFrame(rows).sort_values(["n", "k_ticks", "x"]).to_csv(OUT, index=False)
    print(f"saved {OUT}  ({len(rows)} rows)")
