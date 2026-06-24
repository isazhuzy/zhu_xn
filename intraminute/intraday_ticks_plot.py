"""intraday_ticks_plot.py — fig60: avg ticks per minute across the trading day, per contract.
Line graph, x = time of day (lunch break shown as a gap). Reads intraday_ticks.csv.
Run: python3 intraday_ticks_plot.py
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
_av = {f.name for f in fm.fontManager.ttflist}
for _f in ["Arial Unicode MS", "PingFang HK", "Heiti TC", "STHeiti", "Songti SC"]:
    if _f in _av:
        matplotlib.rcParams["font.sans-serif"] = [_f]; break
matplotlib.rcParams["axes.unicode_minus"] = False

D = "/Users/zhuisabella/xn/intraminute"
NAME = {"IC0000": "IC 中证500", "IF0000": "IF 沪深300", "IH0000": "IH 上证50", "IM0000": "IM 中证1000"}
COL = {"IC0000": "#c0392b", "IF0000": "#e08a3c", "IH0000": "#27ae60", "IM0000": "#4C72B0"}
d = pd.read_csv(f"{D}/intraday_ticks.csv")

fig, ax = plt.subplots(figsize=(12, 6))
for code in NAME:
    s = d[d.code == code].sort_values("tod")
    # drop the session-close minutes 11:30 & 15:00 (single closing print, ~1 tick)
    am = s[(s.tod >= 570) & (s.tod <= 689)]; pm = s[(s.tod >= 780) & (s.tod <= 899)]
    ax.plot(am["tod"], am["avg_ticks"], color=COL[code], lw=1.8, label=NAME[code])
    ax.plot(pm["tod"], pm["avg_ticks"], color=COL[code], lw=1.8)
ax.axhline(120, color="0.55", lw=1.0, ls="--"); ax.text(900, 121, "上限120", ha="right", fontsize=8, color="0.5")
ax.axvspan(690, 780, color="0.92", zorder=0); ax.text(735, ax.get_ylim()[0], "午休", ha="center", va="bottom", fontsize=9, color="0.5")
xt = [570, 600, 630, 660, 690, 780, 810, 840, 870, 900]
ax.set_xticks(xt); ax.set_xticklabels([f"{t//60:02d}:{t%60:02d}" for t in xt], fontsize=9)
ax.set_xlim(565, 905)
ax.set_xlabel("时刻（日内）", fontsize=11)
ax.set_ylabel("每分钟平均 tick 数（跨全部交易日）", fontsize=11)
ax.set_title("图60　各股指期货·日内每分钟平均 tick 数（2020-01..2026-05）\n"
             "典型U型：开盘/收盘活跃，午盘前后清淡（11:30、15:00 收盘单笔已剔除）", fontsize=12.5, fontweight="bold")
ax.legend(fontsize=10, loc="upper center", ncol=4); ax.grid(True, alpha=0.25)
fig.tight_layout(); fig.savefig(f"{D}/figs/fig60_日内每分钟tick数.png", dpi=140); plt.close(fig)
print("saved fig60")
for code in NAME:
    s = d[d.code == code].sort_values("tod")
    print(f"  {NAME[code]}: 09:30={s[s.tod==570]['avg_ticks'].iloc[0]:.0f}  "
          f"11:29={s[s.tod==689]['avg_ticks'].iloc[0]:.0f}  "
          f"13:00={s[s.tod==780]['avg_ticks'].iloc[0]:.0f}  "
          f"14:59={s[s.tod==899]['avg_ticks'].iloc[0]:.0f}  "
          f"最低={s['avg_ticks'].min():.0f}@{int(s.loc[s.avg_ticks.idxmin(),'tod'])//60:02d}:{int(s.loc[s.avg_ticks.idxmin(),'tod'])%60:02d}")
