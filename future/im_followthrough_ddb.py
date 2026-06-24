"""im_followthrough_ddb.py — IM only: does a sudden PRICE move / VOLUME spike lead to a trend?

Two independent tick-level "trigger -> follow-through" studies on IM0000 (中证1000):

  EXP1 PRICE impulse:  at each tick i look back n ticks, back = mid[i] - mid[i-n].
        When |back| exceeds threshold k (index points), TRIGGER in direction
        dir = sign(back). Then measure the forward move signed by that direction
        over horizon h:  signed = dir * (mid[i+h] - mid[i]).
        mean(signed) > 0  => the impulse CONTINUES (a move leads to a trend);
        mean(signed) < 0  => it REVERSES.  k=0.0 row = baseline (every tick).
        Sweep n (lookback), k (threshold), h (forward horizon).

  EXP2 VOLUME spike ("peak"):  short trailing window t1 vs long trailing window t2
        (t2 > t1). spike = (V_t1/t1) / (V_t2/t2)  = recent per-tick volume relative
        to its longer baseline.  When spike > r => a volume PEAK. We then study the
        effect on the NEXT h ticks:
          - abs forward move |mid[i+h]-mid[i]|            (does a peak precede volatility?)
          - signed forward move, signed by the short-window price dir
            sign(mid[i]-mid[i-t1])                        (continuation vs reversal after a peak)
          - mean forward per-tick volume                 (does a peak cluster / persist?)
        r="ALL" row = baseline over every tick, so peak rows can be read relative to it.
        Sweep (t1,t2), r (peak threshold), h.

Windows never cross a session boundary (AM 09:30-11:30 / PM 13:00-15:00) or a day:
everything is computed inside contiguous (day, session) tick blocks. The first tick of
each session (opening-auction volume burst) is excluded from EXP2 via i>=t2.

Outputs (sum/ss/n so a downstream plot can form means + t-stats):
  price_trend.csv   : code,n,k,h, s_sum,s_ss,s_n, hits          (signed fwd return)
  volume_peak.csv   : code,t1,t2,r,h, n_peak,
                      a_sum,a_ss (abs move), g_sum,g_ss (signed move), v_sum,v_ss (fwd vol)

Run: /Users/zhuisabella/xn/.venv/bin/python im_followthrough_ddb.py   (sandbox OFF)
"""
import calendar
import numpy as np
import pandas as pd
import dolphindb as ddb
from ddb_config import HOST, PORT, USER, PW

CODES = ["IM0000"]                       # IM 中证1000 only
WINDOWS = [(y, m) for y in range(2022, 2027) for m in range(1, 13)
           if (2022, 7) <= (y, m) <= (2026, 5)]   # IM exists from 2022-07

# ---- EXP1 price-impulse grid ----------------------------------------------
NS = [5, 10, 20, 40]                     # lookback ticks
KS = [0.0, 0.2, 0.4, 0.8, 1.6]           # |move| threshold, index points (0.0=baseline)
HS = [5, 10, 20, 40, 80, 120]            # forward horizons (ticks)

# ---- EXP2 volume-peak grid ------------------------------------------------
PAIRS = [(5, 30), (5, 60), (5, 120), (10, 60), (10, 120), (20, 120)]   # (t1,t2), t2>t1
RS = [1.5, 2.0, 3.0]                      # spike thresholds; "ALL" baseline added too
HS2 = [10, 20, 60]                        # forward horizons (ticks)

OUT_PRICE = "/Users/zhuisabella/xn/future/price_trend.csv"
OUT_VOL = "/Users/zhuisabella/xn/future/volume_peak.csv"


def fetch(sess, code, start, end):
    q = f"""
    pt=loadTable("dfs://hft_future_ts","TickPartitioned")
    select m_nDatetime as ts, (m_nBidPrice+m_nAskPrice)/2.0 as mid, m_iVolume as vol
    from pt where code_init=`{code[:2]}, m_nDatetime>={start}T00:00:00,
          m_nDatetime<={end}T23:59:59, code=`{code},
          minute(m_nDatetime) between 09:30m:15:00m
    """
    return sess.run(q)


def session_blocks(df):
    """Yield contiguous (day, session) mid/vol arrays, intraday windows only."""
    df = df.drop_duplicates("ts").sort_values("ts").copy()
    tod = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    df = df[((tod >= 570) & (tod <= 690)) | ((tod >= 780) & (tod <= 900))]
    if df.empty:
        return
    df["day"] = df["ts"].dt.normalize()
    df["session"] = np.where(tod[df.index] <= 690, "AM", "PM")
    for _, grp in df.groupby(["day", "session"], sort=False):
        yield grp["mid"].to_numpy(float), grp["vol"].to_numpy(float)


def exp1(mid, acc):
    """Price impulse -> signed forward follow-through."""
    L = len(mid)
    for n in NS:
        for h in HS:
            lo, hi = n, L - h            # i in [n, L-h)
            if hi <= lo:
                continue
            i = np.arange(lo, hi)
            back = mid[i] - mid[i - n]
            fwd = mid[i + h] - mid[i]
            signed = np.sign(back) * fwd
            ab = np.abs(back)
            for k in KS:
                m = ab > k if k > 0 else np.ones(len(i), bool)
                s = signed[m]
                if not s.size:
                    continue
                a = acc.setdefault((n, k, h), [0.0, 0.0, 0, 0])
                a[0] += s.sum(); a[1] += (s * s).sum(); a[2] += s.size
                a[3] += int((s > 0).sum())


def exp2(mid, vol, acc):
    """Volume peak -> forward |move|, signed move, forward volume."""
    L = len(mid)
    cs = np.concatenate([[0.0], np.cumsum(vol)])     # cs[k] = sum vol[:k]
    def trail(t):                                    # trailing-t sum ending at i (incl.)
        s = np.full(L, np.nan)
        s[t - 1:] = cs[t:L + 1] - cs[0:L - t + 1]
        return s
    for (t1, t2) in PAIRS:
        v1, v2 = trail(t1), trail(t2)
        spike = (v1 / t1) / (v2 / t2)
        for h in HS2:
            lo, hi = t2, L - h           # i>=t2 drops opening-auction tick from windows
            if hi <= lo:
                continue
            i = np.arange(lo, hi)
            sp = spike[i]
            absmove = np.abs(mid[i + h] - mid[i])
            pdir = np.sign(mid[i] - mid[i - t1])
            signed = pdir * (mid[i + h] - mid[i])
            fvol = (cs[i + h + 1] - cs[i + 1]) / h
            for r in (["ALL"] + RS):
                m = np.ones(len(i), bool) if r == "ALL" else sp > r
                if not m.any():
                    continue
                am, gm, vm = absmove[m], signed[m], fvol[m]
                a = acc.setdefault((t1, t2, r, h),
                                   [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                a[0] += int(am.size)
                a[1] += am.sum(); a[2] += (am * am).sum()
                a[3] += gm.sum(); a[4] += (gm * gm).sum()
                a[5] += vm.sum(); a[6] += (vm * vm).sum()


def tstat(s, ss, n):
    if n < 2:
        return float("nan"), float("nan")
    mean = s / n
    var = max(ss / n - mean * mean, 0.0)
    se = (var / n) ** 0.5
    return mean, (mean / se if se > 0 else float("nan"))


if __name__ == "__main__":
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    acc1, acc2 = {}, {}
    for yr, mo in WINDOWS:
        last = calendar.monthrange(yr, mo)[1]
        start, end = f"{yr}.{mo:02d}.01", f"{yr}.{mo:02d}.{last:02d}"
        for code in CODES:
            df = fetch(sess, code, start, end)
            if not len(df):
                continue
            for mid, vol in session_blocks(df):
                if len(mid) > max(NS) + max(HS):
                    exp1(mid, acc1)
                if len(mid) > max(p[1] for p in PAIRS) + max(HS2):
                    exp2(mid, vol, acc2)
        print(f"{yr}-{mo:02d} done", flush=True)
    sess.close()

    code = CODES[0]
    r1 = [{"code": code, "n": n, "k": k, "h": h,
           "s_sum": a[0], "s_ss": a[1], "s_n": a[2], "hits": a[3]}
          for (n, k, h), a in acc1.items()]
    pd.DataFrame(r1).sort_values(["n", "k", "h"]).to_csv(OUT_PRICE, index=False)
    r2 = [{"code": code, "t1": t1, "t2": t2, "r": r, "h": h, "n_peak": a[0],
           "a_sum": a[1], "a_ss": a[2], "g_sum": a[3], "g_ss": a[4],
           "v_sum": a[5], "v_ss": a[6]}
          for (t1, t2, r, h), a in acc2.items()]
    pd.DataFrame(r2).sort_values(["t1", "t2", "h"]).to_csv(OUT_VOL, index=False)
    print(f"saved {OUT_PRICE}  ({len(r1)} rows)")
    print(f"saved {OUT_VOL}  ({len(r2)} rows)")

    # ---- compact stdout summary ------------------------------------------
    print("\n=== EXP1 price impulse: mean signed fwd return (t) [hit%] ===")
    print("  k=threshold pts; >0 trend/continuation, <0 reversal.  n=lookback, h=horizon")
    for (n, k, h), a in sorted(acc1.items()):
        mean, t = tstat(a[0], a[1], a[2])
        hp = 100.0 * a[3] / a[2] if a[2] else float("nan")
        print(f"  n={n:>2} k={k:>3} h={h:>3}: {mean:+.4f} (t={t:+5.1f}) "
              f"[{hp:4.1f}%] n={a[2]}")
    print("\n=== EXP2 volume peak: after-peak forward stats ===")
    print("  absmove=|fwd move| (volatility), signed=dir-of-short-win fwd, fvol=fwd vol/tick")
    for (t1, t2, r, h), a in sorted(acc2.items(), key=lambda x: (x[0][0], x[0][1], x[0][3], str(x[0][2]))):
        am, _ = tstat(a[1], a[2], a[0])
        gm, gt = tstat(a[3], a[4], a[0])
        vm, _ = tstat(a[5], a[6], a[0])
        print(f"  t1={t1:>2} t2={t2:>3} r={str(r):>3} h={h:>3}: "
              f"absmove={am:.3f} signed={gm:+.4f}(t={gt:+5.1f}) fvol={vm:.2f} n={a[0]}")
