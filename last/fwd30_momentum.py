"""fwd30_momentum.py — minute-level momentum exercise on tick2min bars (compute).

Rule: at each minute t, sign = direction of minute t's mid change
(mid_close(t) - mid_close(t-1)); hold that direction for the next H=30 minutes;
strategy return = sign * (mid_close(t+H) - mid_close(t)) / mid_close(t), in bps.
Report the average strategy return per minute-of-day t (mean, se, t across days).

Design notes:
  - mid_close (quote midpoint), not trade close -> no bid-ask-bounce fake reversal.
  - prev/next prices found by TIMESTAMP lookup (ts±Δ), not row shift -> the
    forward window never crosses lunch/overnight (those timestamps don't exist),
    and missing or NaN-mid minutes (limit-up) give NaN instead of misalignment.
  - per minute-of-day, each day contributes ONE sample -> non-overlapping,
    so the per-minute t-stat is clean. The pooled overall t is NOT (overlapping
    windows across adjacent minutes) and is printed as indicative only.
Plot with fwd30_momentum_plot.py (system python3).
Run: PILOT=1 /Users/zhuisabella/xn/.venv/bin/python fwd30_momentum.py  (sandbox OFF)
"""
import os
import sys
import numpy as np
import pandas as pd
import dolphindb as ddb

sys.path.insert(0, "/Users/zhuisabella/xn/prediction")
sys.path.insert(0, "/Users/zhuisabella/xn/last")
from ddb_config import HOST, PORT, USER, PW
from tick2min_ddb import fetch_min_bars, to_session_bars

PILOT = os.environ.get("PILOT") == "1"
SUF = "_pilot" if PILOT else ""
D = "/Users/zhuisabella/xn/last"
CODE = os.environ.get("CODE", "IF0000")
H = int(os.environ.get("H", "30"))                    # holding horizon in minutes
START, END = ("2024.06.01", "2024.06.30") if PILOT else ("2024.01.01", "2024.12.31")

sess = ddb.session(HOST, PORT); sess.login(USER, PW)
b = to_session_bars(fetch_min_bars(sess, CODE, START, END), CODE)
sess.close()

# signal & forward return by TIMESTAMP lookup, not row shift: a missing or
# NaN-mid minute (e.g. fully limit-up) yields NaN instead of a misaligned row,
# and session breaks need no special casing (12:59 / 11:30+ simply don't exist)
day = b.ts.dt.normalize()
px = b.set_index("ts")["mid_close"]
mid = b["mid_close"].to_numpy()
prev = px.reindex(b.ts - pd.Timedelta(minutes=1)).to_numpy()
nxt = px.reindex(b.ts + pd.Timedelta(minutes=H)).to_numpy()
sig = np.sign(mid - prev)                             # minute t's own direction
sig[sig == 0] = np.nan                                # flat minute -> no trade
fwd = (nxt - mid) / mid * 1e4                         # next-H-min move, bps
b["strat"] = sig * fwd

# average per minute-of-day: one sample per day per cell -> clean se/t
b["hm"] = b.ts.dt.strftime("%H:%M")
s = (b.dropna(subset=["strat"]).groupby("hm")["strat"]
       .agg(mean="mean", sd="std", n="count").reset_index())
s["se"] = s.sd / np.sqrt(s.n)
s["t"] = s["mean"] / s["se"]
s.insert(0, "code", CODE); s.insert(1, "H", H)
s.to_csv(f"{D}/fwd{H}_momentum_{CODE}{SUF}.csv", index=False)

pooled = b["strat"].dropna()
print(f"{CODE} {START}..{END}  H={H}min  trades={len(pooled)}  days={day.nunique()}")
print(f"pooled mean {pooled.mean():+.2f} bps (indicative t="
      f"{pooled.mean() / pooled.std() * np.sqrt(len(pooled)):.1f}, overlap-inflated)")
print(f"minutes with |t|>=2: {(s.t.abs() >= 2).sum()} / {len(s)}")
print(f"saved fwd{H}_momentum_{CODE}{SUF}.csv")
