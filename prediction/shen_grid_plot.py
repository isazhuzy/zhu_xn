"""shen_grid_plot.py — fig118: Shen-combined three-factor model on the (look-back x
look-forward) grid. Top row: OOS R² heatmaps averaged over the 4 contracts, panels =
VOI / OIR / MPB single vs ALL combined (spread-normalized), same color scale — the
combination gain is visible directly. Bottom row: the ALL model per contract.
Run: python3 shen_grid_plot.py   (SUF=_pilot for pilot)
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
LOOK = ["2s", "20s", "1min", "5min"]; HOR = ["2s", "10s", "1min", "5min"]

g = pd.read_csv(f"{D}/shen_grid{SUF}.csv")
g = g[g.norm == "shen"]


def mat(sub):
    p = sub.pivot_table(index="look", columns="hor", values="r2_oos")
    return p.reindex(index=LOOK, columns=HOR).to_numpy() * 100      # in %


def draw(ax, M, title, vmax):
    im = ax.imshow(M, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                        color="black")
    ax.set_xticks(range(len(HOR)), HOR); ax.set_yticks(range(len(LOOK)), LOOK)
    ax.set_xlabel("前看 H"); ax.set_title(title, fontsize=11)
    return im


fig, axes = plt.subplots(2, 4, figsize=(16, 8))
avg = {m: mat(g[g.model == m].groupby(["look", "hor"], as_index=False)["r2_oos"].mean())
       for m in ("VOI", "OIR", "MPB", "ALL")}
vmax = max(np.nanmax(np.abs(v)) for v in avg.values())
for ax, m in zip(axes[0], ("VOI", "OIR", "MPB", "ALL")):
    ttl = f"{m}（单因子/s）" if m != "ALL" else "ALL 三因子组合（Shen式）"
    im = draw(ax, avg[m], ttl + "\n四合约平均", vmax)
axes[0, 0].set_ylabel("回看 L")

sub = g[g.model == "ALL"]
pm = {c: mat(sub[sub.code == c]) for c in NM}
vmax2 = max(np.nanmax(np.abs(v)) for v in pm.values())
for ax, c in zip(axes[1], NM):
    draw(ax, pm[c], f"ALL组合 · {NM[c]}", vmax2)
axes[1, 0].set_ylabel("回看 L")

fig.suptitle("fig118 — Shen式三因子组合：样本外 R²(%) 在 回看L×前看H 网格上（2s基础bar，训练≤2024-12，OOS 2025-26）",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{D}/fig118_三因子组合网格{SUF}.png", dpi=130)
print(f"saved fig118_三因子组合网格{SUF}.png")
