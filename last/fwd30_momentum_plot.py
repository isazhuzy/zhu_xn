"""fwd30_momentum_plot.py — plot the per-minute-of-day momentum curve.
Reads fwd<H>_momentum_<CODE><SUF>.csv from fwd30_momentum.py.
Run: python3 fwd30_momentum_plot.py   (env: CODE, H, SUF=_pilot)
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
_av = {f.name for f in fm.fontManager.ttflist}
for _f in ["Arial Unicode MS", "PingFang HK", "Heiti TC", "STHeiti", "Songti SC"]:
    if _f in _av:
        matplotlib.rcParams["font.sans-serif"] = [_f]; break
matplotlib.rcParams["axes.unicode_minus"] = False

D = "/Users/zhuisabella/xn/last"
CODE = os.environ.get("CODE", "IF0000")
H = int(os.environ.get("H", "30"))
SUF = os.environ.get("SUF", "")
s = pd.read_csv(f"{D}/fwd{H}_momentum_{CODE}{SUF}.csv")

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
ax.set_title(f"{CODE} 分钟动量：sign(第t分钟涨跌) × 未来{H}分钟收益，按日内分钟平均",
             fontsize=11, fontweight="bold")
ax.legend(); ax.grid(True, alpha=.3)
fig.tight_layout(); fig.savefig(f"{D}/fig_fwd{H}_momentum_{CODE}{SUF}.png", dpi=135)
print(f"saved fig_fwd{H}_momentum_{CODE}{SUF}.png")
