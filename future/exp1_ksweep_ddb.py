"""exp1_ksweep_ddb.py — EXP1 price pulse with EXTENDED grids (user request 2026-06):
  - threshold k swept UP TO 20 price-ticks  (1 price-tick = 0.2 index pt → 0 .. 4.0 pt)
  - lookback n kept SHORT, furthest = 30 ticks back
  - forward horizon x unchanged
Signed forward return = sign(pulse) * (mid[i+x] - mid[i]); >0 trend, <0 reversal.
Reuses fetch / session_blocks / WINDOWS from im_followthrough_ddb (IM only).

Output exp1_ksweep.csv: code,n,k_ticks,k_pts,h, s_sum,s_ss,s_n,hits
  mean = s_sum/s_n ;  se = sqrt(s_ss/s_n - mean^2)/sqrt(s_n) ;  t = mean/se ;  hit% = hits/s_n
Run: /Users/zhuisabella/xn/.venv/bin/python exp1_ksweep_ddb.py   (sandbox OFF)
"""
import calendar
import numpy as np
import pandas as pd
import dolphindb as ddb
from ddb_config import HOST, PORT, USER, PW
from im_followthrough_ddb import fetch, session_blocks, WINDOWS

TICK = 0.2                                   # IM minimum price increment (index pts)
CODE = "IM0000"
NS = [1, 2, 3, 5, 10, 15, 20, 30]            # lookback ticks (short; furthest = 30)
KTICKS = [0, 1, 2, 3, 4, 6, 8, 12, 16, 20]   # threshold in PRICE-TICKS (0 = baseline)
KS = [kt * TICK for kt in KTICKS]            # same threshold in index points
HS = [5, 10, 20, 40, 80, 120]                # forward horizons (ticks)
OUT = "/Users/zhuisabella/xn/future/exp1_ksweep.csv"


def exp1(mid, acc):
    L = len(mid)
    for n in NS:
        for h in HS:
            lo, hi = n, L - h
            if hi <= lo:
                continue
            i = np.arange(lo, hi)
            back = mid[i] - mid[i - n]
            fwd = mid[i + h] - mid[i]
            signed = np.sign(back) * fwd
            ab = np.abs(back)
            for kt, k in zip(KTICKS, KS):
                m = ab > k if k > 0 else np.ones(len(i), bool)
                s = signed[m]
                if not s.size:
                    continue
                a = acc.setdefault((n, kt, h), [0.0, 0.0, 0, 0])
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
            if len(mid) > max(NS) + max(HS):
                exp1(mid, acc)
        print(f"{yr}-{mo:02d} done", flush=True)
    sess.close()
    rows = [{"code": CODE, "n": n, "k_ticks": kt, "k_pts": round(kt * TICK, 2), "h": h,
             "s_sum": a[0], "s_ss": a[1], "s_n": a[2], "hits": a[3]}
            for (n, kt, h), a in acc.items()]
    pd.DataFrame(rows).sort_values(["n", "k_ticks", "h"]).to_csv(OUT, index=False)
    print(f"saved {OUT}  ({len(rows)} rows)")
