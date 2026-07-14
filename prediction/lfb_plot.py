"""lfb_plot.py — fig114: the synthesized 2D view. OOS R^2 heatmap over
(look-BACK L, rows) x (look-FORWARD H, cols), one panel per factor, averaged over the
4 contracts. Shows the signal lives only in the top-left corner (instant input x immediate future).
Run: python3 lfb_plot.py   (SUF=_pilot for pilot)
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
g = pd.read_csv(f"{D}/lfb_grid{SUF}.csv")
LOOK = ["2s", "20s", "1min", "5min"]; HOR = ["2s", "10s", "1min", "5min"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
for ax, fac in zip(axes, ["VOI", "OIR", "MPB"]):
    m = g[g.factor == fac].groupby(["look", "hor"]).r2_oos.mean().reset_index()
    M = m.pivot(index="look", columns="hor", values="r2_oos").reindex(index=LOOK, columns=HOR).to_numpy()
    vmax = np.nanmax(np.abs(g[g.factor == fac].r2_oos))
    im = ax.imshow(M * 100, cmap="RdBu_r", vmin=-vmax * 100, vmax=vmax * 100, aspect="auto")
    ax.set_xticks(range(len(HOR))); ax.set_xticklabels(HOR)
    ax.set_yticks(range(len(LOOK))); ax.set_yticklabels(LOOK)
    ax.set_xlabel("往前看 H（预测未来）", fontsize=10); ax.set_ylabel("往回看 L（累积窗口）", fontsize=10)
    ax.set_title(fac, fontsize=13, fontweight="bold")
    for i in range(len(LOOK)):
        for j in range(len(HOR)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i,j]*100:.2f}", ha="center", va="center", fontsize=9,
                        color="black" if abs(M[i, j]) < vmax * .6 else "white")
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04, label="OOS R² (%)")
fig.suptitle("图114　二维综合：回看 L × 前看 H 的样本外 R²（%，四合约均值）—— 信号只在左上角（即时输入×即时未来）",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{D}/fig114_二维回看前看{SUF}.png", dpi=135)
print("saved fig114")
