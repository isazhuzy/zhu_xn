"""exp2_cross_plot.py — 成交量爆发后【反转占比】vs 爆发强度，跨合约。
Run: python3 exp2_cross_plot.py   (reads exp2_cross.csv; writes figs/fig_cross_revpct_vol.png)
"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
av = {f.name for f in fm.fontManager.ttflist}
for f in ["Arial Unicode MS", "PingFang HK", "Heiti TC", "STHeiti"]:
    if f in av:
        matplotlib.rcParams["font.sans-serif"] = [f]; break
matplotlib.rcParams["axes.unicode_minus"] = False

D = "/Users/zhuisabella/xn/experiment"
d = pd.read_csv(f"{D}/exp2_cross.csv")
NAME = {"IC0000": "IC 中证500", "IF0000": "IF 沪深300", "IH0000": "IH 上证50", "IM0000": "IM 中证1000"}
COL = {"IC0000": "#c0392b", "IF0000": "#e08a3c", "IH0000": "#27ae60", "IM0000": "#4C72B0"}
PAIRS = sorted(set(zip(d.t1, d.t2)))
RS = sorted(d.r.unique())

fig, axes = plt.subplots(1, len(PAIRS), figsize=(7 * len(PAIRS), 6), sharey=True)
if len(PAIRS) == 1:
    axes = [axes]
for ax, (t1, t2) in zip(axes, PAIRS):
    ax.axhline(50, color="k", lw=1.2, ls="--")
    ax.text(0.05, 50.2, "50% = 抛硬币", fontsize=10, color="#555")
    for code in NAME:
        sub = d[(d.code == code) & (d.t1 == t1) & (d.t2 == t2)]
        rp = sub.groupby("r").apply(lambda g: 100 * g["rhits"].sum() / g["n_peak"].sum()).reindex(RS)
        ax.plot(range(len(RS)), rp.values, color=COL[code], lw=2.4, marker="o", ms=7, label=NAME[code])
    ax.set_xticks(range(len(RS))); ax.set_xticklabels([f"{r:g}×" for r in RS], fontsize=10)
    ax.set_xlabel("爆发强度 r（最近成交强度 = r × 平时）", fontsize=11)
    ax.set_title(f"t1={t1}, t2={t2}（放量持续 {t1*0.5:g}s）", fontsize=11)
axes[0].set_ylabel("放量爆发后【反转占比】%（signed<0）", fontsize=11)
axes[0].set_ylim(44, 58)
axes[0].legend(fontsize=10, title="合约")
fig.suptitle("放量爆发之后到底有多大比例反转？—— 看是否真的 >50%（频率，不是均值）　x=20(10s)，池化",
             fontsize=12.5, y=1.0)
fig.tight_layout()
fig.savefig(f"{D}/figs/fig_cross_revpct_vol.png", dpi=130, bbox_inches="tight")
print(f"saved {D}/figs/fig_cross_revpct_vol.png")

print("\n放量后反转占比% (池化):")
for (t1, t2) in PAIRS:
    print(f"-- t1={t1} t2={t2} --")
    for code in NAME:
        sub = d[(d.code == code) & (d.t1 == t1) & (d.t2 == t2)]
        rp = sub.groupby("r").apply(lambda g: 100 * g["rhits"].sum() / g["n_peak"].sum())
        print(f"  {code[:2]}: " + "  ".join(f"r{r:g}:{rp[r]:.1f}%" for r in RS))
