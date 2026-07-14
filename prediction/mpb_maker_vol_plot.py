"""mpb_maker_vol_plot.py — two figures for round 3.
fig111 (MPB → return, windows aligned to VOI): left = MPB(t) vs next-2s mid change (binned
  scatter), the slope IS MPB's standalone edge; right = OOS R² vs horizon (0.5/2/10/60s).
fig112 (RV-aware maker): markout & risk by predicted-vol tercile — high vol pays more per
  fill (wider spread) but is far riskier; pulling it lifts the risk-adjusted edge.
Run: python3 mpb_maker_vol_plot.py   (SUF=_pilot for pilot files)
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

# ---- fig111: MPB → return ----
res = pd.read_csv(f"{D}/mpb_results{SUF}.csv"); sc = pd.read_csv(f"{D}/mpb_scatter{SUF}.csv")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4))
for code in NM:
    s = sc[sc.code == code].sort_values("mpb")
    ax1.plot(s.mpb, s.mean_y, marker="o", ms=3.5, lw=1.5, color=COL[code], label=NM[code])
ax1.axhline(0, color="0.6", lw=.6); ax1.axvline(0, color="0.6", lw=.6)
ax1.set_xlabel("MPB(t) = 成交价基差（tick）", fontsize=10)
ax1.set_ylabel("未来2s平均中间价变动（tick）", fontsize=10)
ax1.set_title("MPB → 未来收益：主动买(MPB>0)→价涨\n（分箱均值，窗口对齐VOI）", fontsize=11, fontweight="bold")
ax1.legend(fontsize=8.5); ax1.grid(True, alpha=.25)
for code in NM:
    s = res[res.code == code].sort_values("secs")
    ax2.plot(s.secs, s.r2_oos, marker="o", ms=5, lw=1.8, color=COL[code], label=NM[code])
ax2.axhline(0, color="k", lw=.8)
ax2.set_xscale("log"); ax2.set_xticks([0.5, 2, 10, 60]); ax2.set_xticklabels(["0.5s", "2s", "10s", "60s"])
ax2.set_xlabel("预测期", fontsize=10); ax2.set_ylabel("MPB 单因子 OOS R²", fontsize=10)
ax2.set_title("MPB 单因子预测力：短端有、随期限衰减\n（样本外 2025-26）", fontsize=11, fontweight="bold")
ax2.legend(fontsize=8.5); ax2.grid(True, alpha=.25)
fig.suptitle("图111　MPB(成交价基差) 对未来收益的独立影响 —— 窗口对齐 VOI 研究", fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{D}/fig111_MPB对收益{SUF}.png", dpi=135); plt.close(fig)
print("saved fig111")

# ---- fig112: RV-aware maker ----
try:
    v = pd.read_csv(f"{D}/xc_maker_vol_results{SUF}.csv")
    reg = ["low", "mid", "high"]
    labs = {"low": "低波动", "mid": "中波动", "high": "高波动"}
    m = v[v.regime.isin(reg)].set_index("regime").reindex(reg)
    fig, (bx1, bx2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    x = np.arange(3); C = ["#4C72B0", "#e08a3c", "#c0392b"]
    bx1.bar(x - 0.2, m.markout_tk, 0.4, color=C, label="每笔 markout(毛)")
    bx1.bar(x + 0.2, m.std_tk, 0.4, color=C, alpha=.4, hatch="//", label="风险 std")
    bx1.set_xticks(x); bx1.set_xticklabels([labs[r] for r in reg], fontsize=10)
    bx1.set_ylabel("tick", fontsize=10)
    bx1.set_title("按预测波动分档：高波动毛利最高(宽价差)\n但风险(std)也最大", fontsize=11, fontweight="bold")
    bx1.legend(fontsize=9); bx1.grid(True, axis="y", alpha=.3)
    bx2.bar(x, m.risk_adj, 0.5, color=C)
    for i, r in enumerate(reg):
        bx2.text(i, m.risk_adj.iloc[i] + .01, f"{m.risk_adj.iloc[i]:.2f}", ha="center", fontsize=10, fontweight="bold")
    al = v[v.regime.str.startswith("ALL")].risk_adj.iloc[0]
    pl = v[v.regime.str.startswith("PULL")].risk_adj.iloc[0]
    bx2.axhline(al, color="0.5", ls="--", lw=1.2, label=f"全接 {al:.2f}")
    bx2.axhline(pl, color="#27ae60", ls="-", lw=1.6, label=f"撤高波动 {pl:.2f}")
    bx2.set_xticks(x); bx2.set_xticklabels([labs[r] for r in reg], fontsize=10)
    bx2.set_ylabel("风险调整后 markout（mean/std）", fontsize=10)
    bx2.set_title("风险调整后：高波动最差；撤掉它 → 风险调整↑\n预测波动的价值在风控,不在毛利", fontsize=11, fontweight="bold")
    bx2.legend(fontsize=9); bx2.grid(True, axis="y", alpha=.3)
    fig.suptitle("图112　RV感知做市：预测到高波动就撤单 —— 毛利降但风险降更多,风险调整后更优", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{D}/fig112_RV感知做市{SUF}.png", dpi=135); plt.close(fig)
    print("saved fig112")
except FileNotFoundError:
    print("xc_maker_vol_results not ready yet")
