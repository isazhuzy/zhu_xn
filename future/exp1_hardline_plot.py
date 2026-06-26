"""exp1_hardline_plot.py — the "hardline" view: mean sign-flip contour + hit-rate map.
Run: python3 exp1_hardline_plot.py   (reads exp1_nk_clean.csv, x=10; writes 1 fig)
 左：均值热力图 + 均值=0 等高线（hardline，趋势↔反转分界）。
 右：命中率 hit% 热力图 + 50% 等高线（"单次大概率反转"的线——几乎不存在）。
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib import font_manager as fm

_av = {f.name for f in fm.fontManager.ttflist}
for _f in ["Arial Unicode MS", "PingFang HK", "Heiti TC", "STHeiti", "Songti SC"]:
    if _f in _av:
        matplotlib.rcParams["font.sans-serif"] = [_f]; break
matplotlib.rcParams["axes.unicode_minus"] = False

D = "/Users/zhuisabella/xn/future"
FIG = f"{D}/figs"
d = pd.read_csv(f"{D}/exp1_nk_clean.csv")
X = int(d["x"].iloc[0])
d["mean"] = d["s_sum"] / d["s_n"]
d["hit"] = 100 * d["hits"] / d["s_n"]
NS = sorted(d["n"].unique()); KT = sorted(d["k_ticks"].unique())
M = d.pivot(index="n", columns="k_ticks", values="mean").reindex(index=NS, columns=KT)
H = d.pivot(index="n", columns="k_ticks", values="hit").reindex(index=NS, columns=KT)
N = d.pivot(index="n", columns="k_ticks", values="s_n").reindex(index=NS, columns=KT)
xx, yy = np.meshgrid(range(len(KT)), range(len(NS)))

fig, (axA, axB) = plt.subplots(1, 2, figsize=(17, 7.2))

# -------- 左：均值 + hardline (mean=0) --------
imA = axA.imshow(M.values, cmap="seismic", vmin=-0.5, vmax=0.5, aspect="auto", origin="upper")
cs = axA.contour(xx, yy, M.values, levels=[0], colors="black", linewidths=3)
axA.clabel(cs, fmt={0: "hardline 均值=0"}, fontsize=10, inline=True)
for i in range(len(NS)):
    for j in range(len(KT)):
        v = M.values[i, j]
        if np.isfinite(v):
            faint = N.values[i, j] < 50000
            axA.text(j, i, (f"{v:+.2f}" if abs(v) < 10 else f"{v:+.0f}"),
                     ha="center", va="center", fontsize=6,
                     color=("#777" if faint else ("white" if abs(min(max(v, -.5), .5)) > 0.6 * .5 else "black")))
axA.text(0.04, 0.06, "趋势 (+)\n左/上区", transform=axA.transAxes, fontsize=11,
         color="#7a1010", ha="left", va="bottom", weight="bold")
axA.text(0.97, 0.94, "反转 (−)\n右/下区", transform=axA.transAxes, fontsize=11,
         color="#0d1f55", ha="right", va="top", weight="bold")
axA.set_title("① 均值（指数点）＋ hardline\n越过黑线 = 期望结果翻成反转", fontsize=11.5)
fig.colorbar(imA, ax=axA, fraction=0.046, pad=0.03, extend="both")

# -------- 右：命中率 hit% + 50% 线 --------
imB = axB.imshow(H.values, cmap="RdBu", norm=TwoSlopeNorm(vmin=40, vcenter=50, vmax=53),
                 aspect="auto", origin="upper")
cs2 = axB.contour(xx, yy, H.values, levels=[50], colors="black", linewidths=3)
axB.clabel(cs2, fmt={50: "50% 线"}, fontsize=10, inline=True)
for i in range(len(NS)):
    for j in range(len(KT)):
        v = H.values[i, j]
        if np.isfinite(v):
            axB.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.3,
                     color="white" if (v > 51.5 or v < 44) else "black")
axB.text(0.5, 0.5, "几乎整张图 < 50%\n（多数脉冲其实继续，不是反转）",
         transform=axB.transAxes, fontsize=11, color="#222", ha="center", va="center",
         bbox=dict(boxstyle="round", fc="#ffffffcc", ec="#888"))
axB.set_title("② 命中率 hit%（单次真反转的比例）\n50%=抛硬币；>50% 才算'大概率反转'", fontsize=11.5)
fig.colorbar(imB, ax=axB, fraction=0.046, pad=0.03, extend="both")

for ax in (axA, axB):
    ax.set_xticks(range(len(KT))); ax.set_xticklabels(KT, fontsize=8)
    ax.set_yticks(range(len(NS))); ax.set_yticklabels(NS, fontsize=8)
    ax.set_xlabel("k　脉冲幅度阈值（价格 tick；1=0.2 点）", fontsize=10)
    ax.set_ylabel("n　回看窗口（tick；越小越新鲜）", fontsize=10)

fig.suptitle(f"实验1：脉冲越大→反转越强？ hardline 在哪？（固定向前 x={X} tick≈5秒）"
             "　IM 中证1000 · 2022-07..2026-05（已过滤坏tick）", fontsize=13, y=1.0)
fig.text(0.5, -0.02,
         "计算方法：脉冲 = mid[i]−mid[i−n]，|脉冲|>k 触发。①均值 = sign(脉冲)×(mid[i+x]−mid[i]) 的平均；"
         "②命中率 = 该平均符号方向上'确实反转'的比例。关键：① 翻负的硬线存在（≈k=6~8 tick），但 ② 命中率几乎从不过 50% → "
         "反转是'幅度/赔率'边际，不是'胜率'边际。",
         ha="center", va="top", fontsize=8.6, color="#333")
fig.tight_layout()
fig.savefig(f"{FIG}/fig_pulse_hardline.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"saved {FIG}/fig_pulse_hardline.png")
