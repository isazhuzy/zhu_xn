"""intraminute_shifted_plot.py — fig13: whole-minute (shifted grid) reversal vs phase N.
Overlays the old minute-close basket (dashed) to show the shrinking-hold confound."""
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

OUT = "/Users/zhuisabella/xn/ticker/figs_conclusion"
NAME = {"IC0000": "IC 中证500", "IF0000": "IF 沪深300",
        "IH0000": "IH 上证50", "IM0000": "IM 中证1000"}
COL = {"IC0000": "#c0392b", "IF0000": "#e08a3c", "IH0000": "#27ae60", "IM0000": "#4C72B0"}
new = pd.read_csv("/Users/zhuisabella/xn/ticker/open_breakdown/intraminute_shifted.csv")
old = pd.read_csv("/Users/zhuisabella/xn/ticker/open_breakdown/intraminute_wholeday.csv")
N_LIST = sorted(new["N"].unique())

fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
for code in NAME:
    s = new[new.group == code].sort_values("N")
    axes[0].plot(s["N"], s["mean_bp"], "-o", color=COL[code], ms=4, lw=1.1, alpha=.65, label=NAME[code])
    axes[1].plot(s["N"], s["win"] * 100, "-o", color=COL[code], ms=4, lw=1.1, alpha=.65, label=NAME[code])
bn = new[new.group == "组合(4合约等权)"].sort_values("N")
bo = old[old.group == "组合(4合约等权)"].sort_values("N")
axes[0].plot(bn["N"], bn["mean_bp"], "-s", color="k", lw=2.6, ms=7, label="组合·满分钟持有(本次)", zorder=5)
axes[0].plot(bo["N"], bo["mean_bp"], "--D", color="0.45", lw=1.8, ms=5, label="组合·持有到该分钟收盘(旧版)")
for _, r in bn.iterrows():
    axes[0].annotate(f"{r['t']:.1f}", (r["N"], r["mean_bp"]), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=7, fontweight="bold")
axes[1].plot(bn["N"], bn["win"] * 100, "-s", color="k", lw=2.6, ms=7, label="组合(满分钟)")
axes[0].axhline(0, color="0.5", lw=.8); axes[1].axhline(50, color="0.5", lw=1, ls=":")
axes[0].set_ylabel("反转每笔平均收益（bp，毛）")
axes[1].set_ylabel("反转胜率（%）")
axes[1].set_xlabel("进场相位 N（在该分钟第N个tick进场；持有满一分钟到下一分钟第N tick）")
axes[1].set_xticks(N_LIST)
axes[0].text(0.015, 0.05, "满分钟持有下：反转只在 N=1–2 略正；相位偏移即转为轻微延续(N≈8–14最负)",
             transform=axes[0].transAxes, fontsize=9.5, color="#c0392b", fontweight="bold")
for ax in axes:
    ax.grid(alpha=.3); ax.legend(fontsize=7.5, ncol=2)
fig.suptitle("整分钟(平移网格)反转 vs 进场相位N（38天）：去掉「持有随N变短」的混淆",
             fontsize=13, fontweight="bold")
fig.text(0.5, 0.01, "实线=持有满一分钟(本次,N=纯相位);虚线=持有到该分钟收盘(持有随N缩短)。"
         "黑线上数字为t(按38日聚合)。所有幅度<买卖价差。毛收益。",
         ha="center", fontsize=8.5, color="0.35")
fig.tight_layout(rect=(0, 0.03, 1, 0.96))
fig.savefig(f"{OUT}/fig13_整分钟平移对比.png", dpi=150)
print("saved", f"{OUT}/fig13_整分钟平移对比.png")
