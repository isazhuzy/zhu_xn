"""factor_horizon_plot.py — fig115: fig111-style forward study for VOI, OIR, MPB together.
Row 1: binned scatter factor(t) -> future 2s mid change (VOI/OIR/MPB), 4 contracts.
Row 2: single-factor OOS R^2 vs forward horizon (0.5s..60s), 4 contracts.
Run: python3 factor_horizon_plot.py   (SUF=_pilot)
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
D = "/Users/zhuisabella/xn/prediction"
SUF = os.environ.get("SUF", "")
NM = {"IC0000": "IC 中证500", "IF0000": "IF 沪深300", "IH0000": "IH 上证50", "IM0000": "IM 中证1000"}
COL = {"IC0000": "#c0392b", "IF0000": "#e08a3c", "IH0000": "#27ae60", "IM0000": "#4C72B0"}
XL = {"VOI": "VOI(t) 队伍净流入", "OIR": "OIR(t) 队列失衡", "MPB": "MPB(t) 成交价基差"}
res = pd.read_csv(f"{D}/fh_results{SUF}.csv"); sc = pd.read_csv(f"{D}/fh_scatter{SUF}.csv")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for j, fac in enumerate(["VOI", "OIR", "MPB"]):
    ax = axes[0][j]
    for code in NM:
        s = sc[(sc.code == code) & (sc.factor == fac)].sort_values("x")
        ax.plot(s.x, s.mean_y, marker="o", ms=3, lw=1.4, color=COL[code], label=NM[code])
    ax.axhline(0, color="0.6", lw=.6); ax.axvline(0, color="0.6", lw=.6)
    ax.set_xlabel(XL[fac], fontsize=9.5); ax.set_ylabel("未来2s平均中间价变动 (tick)", fontsize=9)
    ax.set_title(f"{fac} → 未来收益（分箱均值）", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7.5); ax.grid(True, alpha=.25)
    ax2 = axes[1][j]
    for code in NM:
        s = res[(res.code == code) & (res.factor == fac)].sort_values("secs")
        ax2.plot(s.secs, s.r2_oos, marker="o", ms=5, lw=1.8, color=COL[code], label=NM[code])
    ax2.axhline(0, color="k", lw=.8)
    ax2.set_xscale("log"); ax2.set_xticks([0.5, 2, 10, 60]); ax2.set_xticklabels(["0.5s", "2s", "10s", "60s"])
    ax2.set_xlabel("预测期（前看）", fontsize=9.5); ax2.set_ylabel(f"{fac} 单因子 OOS R²", fontsize=9.5)
    ax2.set_title(f"{fac} 预测力随前看衰减", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=7.5); ax2.grid(True, alpha=.25)
fig.suptitle("图115　三个即时因子的预测力（fig111 的 VOI/OIR/MPB 全套）—— 上：因子→未来2s散点；下：OOS R² 随前看衰减",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(f"{D}/fig115_三因子预测力{SUF}.png", dpi=135)
print("saved fig115")
