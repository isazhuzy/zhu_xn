"""voi_scatter_plot.py — fig86: the predictive relationship, one glance.
Left: binned mean(next-0.5s mid change) vs (prev-0.5s VOI) per contract — the pure,
non-tradeable signal (slope = ticks of future move per net contract of flow).
Right: the three-rung R² ladder (1 factor / +lags / +factors) from voi_single + voi_results.
Run: python3 voi_scatter_plot.py
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
D = "/Users/zhuisabella/xn/prediction"
NAME = {"IC0000": "IC 中证500", "IF0000": "IF 沪深300", "IH0000": "IH 上证50", "IM0000": "IM 中证1000"}
COL = {"IC0000": "#c0392b", "IF0000": "#e08a3c", "IH0000": "#27ae60", "IM0000": "#4C72B0"}
sc = pd.read_csv(f"{D}/voi_scatter.csv"); sg = pd.read_csv(f"{D}/voi_single.csv")
res = pd.read_csv(f"{D}/voi_results.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.8))

for code in NAME:
    s = sc[sc.code == code].sort_values("voi")
    g = sg[sg.code == code].iloc[0]
    ax1.plot(s.voi, s.mean_y1, marker="o", ms=3.5, lw=1.5, color=COL[code],
             label=f"{NAME[code]}  斜率={g.slope_tick_per_lot:.4f} R²={g.r2_single:.3f}")
    xs = np.linspace(s.voi.min(), s.voi.max(), 40)
    ax1.plot(xs, xs * g.slope_tick_per_lot, color=COL[code], lw=0.7, ls="--", alpha=.5)
ax1.axhline(0, color="0.6", lw=.6); ax1.axvline(0, color="0.6", lw=.6)
ax1.set_xlabel("VOI = 前一个500ms内队伍净流入（t−1→t，手）", fontsize=10)
ax1.set_ylabel("下一个500ms中间价变动 mid(t+1)−mid(t) (tick)", fontsize=10)
ax1.set_title("用 [t−1→t] 的队伍变化 预测 [t→t+1] 的价格：纯信号，未扣任何成本\n（样本外 2025-26；点=分箱均值）", fontweight="bold", fontsize=10.5)
ax1.legend(fontsize=8); ax1.grid(True, alpha=.25)

rung = {}
for code in NAME:
    a1 = res[(res.code == code) & (res.k == 1) & (res.model == "A")].iloc[0]
    b1 = res[(res.code == code) & (res.k == 1) & (res.model == "B")].iloc[0]
    s1 = sg[sg.code == code].iloc[0]
    rung[code] = [s1.r2_single, a1.r2_test_oos, b1.r2_test_oos]
labels = ["① VOI（队伍变化）\n仅当前值",
          "② VOI + 过去5个\n滞后值（模型A）",
          "③ VOI+OIR+MPB\n各带滞后÷价差（模型B）"]
xs = np.arange(3); w = 0.2
for i, code in enumerate(NAME):
    ax2.bar(xs + (i - 1.5) * w, rung[code], w, color=COL[code], label=NAME[code])
ax2.set_xticks(xs); ax2.set_xticklabels(labels, fontsize=8.5)
ax2.set_ylabel("样本外预测 R²（下一个500ms）", fontsize=10)
ax2.set_title("加入异质因子后预测 R² 逐级抬高\nVOI=队伍变化 · OIR=队伍水平 · MPB=成交方向（2025-26 样本外）",
              fontweight="bold", fontsize=10)
ax2.legend(fontsize=8.5); ax2.grid(True, axis="y", alpha=.3)

fig.suptitle("图86　论文3的预测关系（用前一个500ms的队伍变化预测下一个500ms的中间价变动）",
             fontsize=12, fontweight="bold")
fig.text(0.5, 0.015,
         "R²（决定系数）= 模型解释了价格波动的百分之几：R²=0 什么都没解释，R²=1 完美预测；"
         "R²=0.087 即解释了 8.7% 的波动，其余为噪声。高频收益预测里 R² 几个百分点已属强信号。",
         ha="center", fontsize=8.5, color="0.35")
fig.tight_layout(rect=(0, 0.04, 1, 0.93)); fig.savefig(f"{D}/fig86_VOI预测散点.png", dpi=135)
print("saved fig86")
