"""fig59_avg_ticks.py — average ticks per minute, per contract (bar chart).
Reads ticks_per_minute_hist.csv (2020-01..2026-05). Run: python3 fig59_avg_ticks.py
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
NAME = {"IC0000": "IC\n中证500", "IF0000": "IF\n沪深300", "IH0000": "IH\n上证50", "IM0000": "IM\n中证1000"}
COL = {"IC0000": "#c0392b", "IF0000": "#e08a3c", "IH0000": "#27ae60", "IM0000": "#4C72B0"}
h = pd.read_csv(f"{D}/ticks_per_minute_hist.csv")

codes = list(NAME); x = np.arange(len(codes))
means, meds = [], []
for c in codes:
    s = h[h.code == c].sort_values("cnt")
    cnt = s["cnt"].to_numpy().astype(float); n = s["n_minutes"].to_numpy().astype(float)
    tot = n.sum()
    means.append((cnt * n).sum() / tot)
    meds.append(cnt[np.searchsorted(np.cumsum(n), .5 * tot)])

fig, ax = plt.subplots(figsize=(8.5, 5.6))
ax.axhline(120, color="0.4", lw=1.3, ls="--")
ax.text(len(codes) - 0.5, 121, "500ms 理论上限 = 120", ha="right", fontsize=9, color="0.4")
bars = ax.bar(x, means, width=0.6, color=[COL[c] for c in codes], alpha=0.85)
for i, c in enumerate(codes):
    ax.text(i, means[i] + 1.5, f"{means[i]:.0f}", ha="center", fontsize=14, fontweight="bold")
    ax.scatter([i], [meds[i]], marker="_", s=900, color="k", lw=2, zorder=5)
    ax.text(i, meds[i] + 1.5, f"中位 {meds[i]:.0f}", ha="center", fontsize=8, color="0.25")
ax.set_xticks(x); ax.set_xticklabels([NAME[c] for c in codes], fontsize=10)
ax.set_ylim(0, 130)
ax.set_ylabel("每分钟平均 tick 数（去重时间戳）", fontsize=11)
ax.set_title("图59　各股指期货·每分钟平均 tick 数（2020-01..2026-05；会话内）\n"
             "柱=均值，黑横线=中位数；理论上限 120（500ms/档）", fontsize=12.5, fontweight="bold")
ax.grid(True, axis="y", alpha=0.25)
fig.tight_layout(); fig.savefig(f"{D}/figs/fig59_每分钟平均tick数.png", dpi=140); plt.close(fig)
print("saved fig59")
for c, m, md in zip(codes, means, meds):
    print(f"  {c}: 均值 {m:.1f}  中位 {md:.0f}")
