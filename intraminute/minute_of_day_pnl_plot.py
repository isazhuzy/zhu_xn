"""minute_of_day_pnl_plot.py — momentum P&L profile across minute-of-day (one month).
AM and PM panels; one line per contract; mark session-open minutes. Run: python3 ..."""
import glob, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

_av = {f.name for f in fm.fontManager.ttflist}
for _f in ["Arial Unicode MS", "PingFang HK", "Heiti TC", "STHeiti", "Songti SC"]:
    if _f in _av:
        matplotlib.rcParams["font.sans-serif"] = [_f]; break
matplotlib.rcParams["axes.unicode_minus"] = False

OUT = "/Users/zhuisabella/xn/intraminute/figs"
NAME = {"IC0000": "IC 中证500", "IF0000": "IF 沪深300", "IH0000": "IH 上证50", "IM0000": "IM 中证1000"}
COL = {"IC0000": "#c0392b", "IF0000": "#e08a3c", "IH0000": "#27ae60", "IM0000": "#4C72B0"}
f = sorted(glob.glob("/Users/zhuisabella/xn/intraminute/minute_of_day_pnl_*.csv"))[-1]
tag = re.search(r"(\d{4}_\d{2})", f).group(1)
df = pd.read_csv(f)

fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), sharey=True)
for ax, (lo, hi, ttl) in zip(axes, [(570, 690, "上午 09:30–11:30"), (780, 900, "下午 13:00–15:00")]):
    for code in NAME:
        s = df[(df.code == code) & (df.tod >= lo) & (df.tod <= hi)].sort_values("tod")
        ax.plot(s["tod"], s["mean"], color=COL[code], lw=1.2, marker="o", ms=2.5, label=NAME[code])
    ax.axhline(0, color="0.4", lw=.8)
    ax.axvspan(lo, lo + 5, color="gold", alpha=0.18)            # first 5 min of session
    ax.set_title(ttl, fontsize=11, fontweight="bold")
    ticks = list(range(lo, hi + 1, 15))
    ax.set_xticks(ticks); ax.set_xticklabels([f"{t//60:02d}:{t%60:02d}" for t in ticks], fontsize=8)
    ax.grid(True, alpha=0.25)
axes[0].set_ylabel("动量 P&L 均值（指数点/分钟）", fontsize=10)
axes[0].legend(fontsize=8, loc="best")
fig.suptitle(f"图30　各分钟 动量 P&L = sign(上一分钟)×(收−开) · 月内日均（{tag.replace('_','-')}；金=开盘前5分钟）",
             fontsize=13, fontweight="bold")
fig.text(0.5, 0.005, "单月约20个交易日;每点=该minute-of-day的日均动量盈亏。约240个分钟做多重检验,个别t≈3属预期内的偶然——"
         "需跨月复现才算真。正=动量延续可赚,负=反转(反向可赚)。", ha="center", fontsize=8, color="0.4")
fig.tight_layout(rect=(0, 0.03, 1, 0.96))
fig.savefig(f"{OUT}/fig30_分钟P&L_{tag}.png", dpi=130); plt.close(fig)
print(f"saved fig30 ({tag})")

# how many minutes would we EXPECT |t|>2 by chance vs observed
for code in NAME:
    s = df[df.code == code]
    obs = (s["t"].abs() > 2).sum()
    print(f"{code}: minutes |t|>2 observed = {obs} / {len(s)}  (chance ~{0.046*len(s):.0f})")
