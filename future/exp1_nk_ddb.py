"""exp1_nk_ddb.py — EXP1 price pulse, dense n×k grid at a FIXED forward horizon x
(user 2026-06: 不再扫 x；只看 n 和阈值 k 对 x 笔之后收益的影响。n≤30, k≤90 价格tick).
脉冲 = mid[i]−mid[i−n]，|脉冲|>k 触发；纵轴 = sign(脉冲)×(mid[i+x]−mid[i]) 的均值。
Reuses fetch/session_blocks/WINDOWS from im_followthrough_ddb (IM only).
Output exp1_nk.csv: code,n,k_ticks,k_pts,x, s_sum,s_ss,s_n,hits
Run: /Users/zhuisabella/xn/.venv/bin/python exp1_nk_ddb.py   (sandbox OFF)
"""
import calendar
import numpy as np
import pandas as pd
import dolphindb as ddb
from ddb_config import HOST, PORT, USER, PW
from im_followthrough_ddb import fetch, session_blocks, WINDOWS

TICK = 0.2
CODE = "IM0000"
X = 10                                                # FIXED forward horizon (ticks ≈ 5s)
NS = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 30]    # lookback (≤30)
KTICKS = [0, 1, 2, 3, 4, 6, 8, 10, 13, 16, 20, 30, 45, 60, 90]   # threshold price-ticks (≤90)
KS = [kt * TICK for kt in KTICKS]
OUT = "/Users/zhuisabella/xn/future/exp1_nk.csv"


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
                exp1(mid, acc)
        print(f"{yr}-{mo:02d} done", flush=True)
    sess.close()
    rows = [{"code": CODE, "n": n, "k_ticks": kt, "k_pts": round(kt * TICK, 2), "x": X,
             "s_sum": a[0], "s_ss": a[1], "s_n": a[2], "hits": a[3]}
            for (n, kt), a in acc.items()]
    pd.DataFrame(rows).sort_values(["n", "k_ticks"]).to_csv(OUT, index=False)
    print(f"saved {OUT}  ({len(rows)} rows)")
