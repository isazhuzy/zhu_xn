"""fwd30_momentum.py — minute-level momentum exercise on tick2min bars.

Rule: at each minute t, sign = direction of minute t's mid change
(mid_close(t) - mid_close(t-1)); hold that direction for the next H=30 minutes;
strategy return = sign * (mid_close(t+H) - mid_close(t)) / mid_close(t), in bps.
Report the average strategy return per minute-of-day t (mean, se, t across days).

Design notes:
  - mid_close (quote midpoint), not trade close -> no bid-ask-bounce fake reversal.
  - shift(-H) within (day, session) groups -> the forward window never crosses
    lunch or overnight; last H minutes of each session drop out as NaN.
  - per minute-of-day, each day contributes ONE sample -> non-overlapping,
    so the per-minute t-stat is clean. The pooled overall t is NOT (overlapping
    windows across adjacent minutes) and is printed as indicative only.
Run: PILOT=1 /Users/zhuisabella/xn/.venv/bin/python fwd30_momentum.py  (sandbox OFF)
"""
import os
import sys
import numpy as np
import pandas as pd
import dolphindb as ddb
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

sys.path.insert(0, "/Users/zhuisabella/xn/prediction")
sys.path.insert(0, "/Users/zhuisabella/xn/last")
from ddb_config import HOST, PORT, USER, PW
from tick2min_ddb import fetch_min_bars, to_session_bars

_av = {f.name for f in fm.fontManager.ttflist}
for _f in ["Arial Unicode MS", "PingFang HK", "Heiti TC", "STHeiti", "Songti SC"]:
    if _f in _av:
        matplotlib.rcParams["font.sans-serif"] = [_f]; break
matplotlib.rcParams["axes.unicode_minus"] = False

PILOT = os.environ.get("PILOT") == "1"
SUF = "_pilot" if PILOT else ""
D = "/Users/zhuisabella/xn/last"
CODE = os.environ.get("CODE", "IF0000")
H = int(os.environ.get("H", "30"))                    # holding horizon in minutes
START, END = ("2024.06.01", "2024.06.30") if PILOT else ("2024.01.01", "2024.12.31")

sess = ddb.session(HOST, PORT); sess.login(USER, PW)
b = to_session_bars(fetch_min_bars(sess, CODE, START, END), CODE)
sess.close()

# signal & forward return, both inside (day, session) so nothing crosses a break
day = b.ts.dt.normalize()
pm = (b.ts.dt.hour >= 13).astype(int)
g = b.groupby([day, pm])["mid_close"]
sig = np.sign(g.diff())                               # minute t's own direction
sig = sig.replace(0, np.nan)                          # flat minute -> no trade
fwd = (g.shift(-H) - b["mid_close"]) / b["mid_close"] * 1e4   # next-H-min move, bps
b["strat"] = sig * fwd

# average per minute-of-day: one sample per day per cell -> clean se/t
b["hm"] = b.ts.dt.strftime("%H:%M")
s = (b.dropna(subset=["strat"]).groupby("hm")["strat"]
       .agg(mean="mean", sd="std", n="count").reset_index())
s["se"] = s.sd / np.sqrt(s.n)
s["t"] = s["mean"] / s["se"]
s.to_csv(f"{D}/fwd{H}_momentum_{CODE}{SUF}.csv", index=False)

pooled = b["strat"].dropna()
print(f"{CODE} {START}..{END}  H={H}min  trades={len(pooled)}  days={day.nunique()}")
print(f"pooled mean {pooled.mean():+.2f} bps (indicative t={pooled.mean()/pooled.std()*np.sqrt(len(pooled)):.1f}, overlap-inflated)")
print(f"minutes with |t|>=2: {(s.t.abs() >= 2).sum()} / {len(s)}")

fig, ax = plt.subplots(figsize=(12, 4.5))
x = np.arange(len(s))
ax.axhline(0, color="0.5", lw=.7)
ax.fill_between(x, s["mean"] - 2 * s.se, s["mean"] + 2 * s.se,
                color="#4C72B0", alpha=.25, label="±2·se")
ax.plot(x, s["mean"], lw=1.4, color="#4C72B0", label="均值")
tick = [i for i, h in enumerate(s.hm) if h.endswith(("00", "30"))]
ax.set_xticks(tick); ax.set_xticklabels(s.hm.iloc[tick], fontsize=8)
ax.set_xlabel("日内分钟 t（信号 = 第 t 分钟方向）")
ax.set_ylabel(f"动量策略收益（bps，持有 {H} 分钟）")
ax.set_title(f"{CODE} 分钟动量：sign(第t分钟涨跌) × 未来{H}分钟收益，按日内分钟平均"
             f"（{START}–{END}）", fontsize=11, fontweight="bold")
ax.legend(); ax.grid(True, alpha=.3)
fig.tight_layout(); fig.savefig(f"{D}/fig_fwd{H}_momentum_{CODE}{SUF}.png", dpi=135)
print(f"saved fwd{H}_momentum_{CODE}{SUF}.csv + fig")
