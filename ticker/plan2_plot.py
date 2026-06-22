"""plan2_plot.py — fig14 from plan2_threshold.csv (python3.14).
mean_bp vs dead-band k, full + two day-halves, with n collapsing → no prime threshold."""
import numpy as np
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
NAME = {"IC0000": "IC 中证500", "IF0000": "IF 沪深300", "IH0000": "IH 上证50", "IM0000": "IM 中证1000"}
tab = pd.read_csv("/Users/zhuisabella/xn/ticker/open_breakdown/plan2_threshold.csv")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, code in zip(axes.ravel(), NAME):
    s = tab[tab.code == code]
    full = s[s.half == "full"].sort_values("k")
    h1 = s[s.half == "H1"].sort_values("k")
    h2 = s[s.half == "H2"].sort_values("k")
    # grey shading where full-sample n < 50 (unreliable zone)
    small = full[full["n"] < 50]
    if len(small):
        ax.axvspan(small["k"].min(), 40, color="0.85", alpha=.6, zorder=0)
        ax.text(small["k"].min() + 0.5, ax.get_ylim()[1], "n<50 不可靠区",
                fontsize=8, color="0.4", va="top")
    ax.plot(h1["k"], h1["mean_bp"], "--", color="#2c6fbb", lw=1.3, label="前一半交易日 H1")
    ax.plot(h2["k"], h2["mean_bp"], "--", color="#e08a3c", lw=1.3, label="后一半交易日 H2")
    ax.plot(full["k"], full["mean_bp"], "-", color="k", lw=2.4, label="全样本")
    ax.axhline(0, color="0.5", lw=.8)
    kbest = int(full.loc[full["mean_bp"].idxmax(), "k"])
    rb = full[full.k == kbest].iloc[0]
    ax.axvline(kbest, color="#c0392b", lw=1, ls=":")
    ax.annotate(f"k*={kbest}\nmean={rb['mean_bp']:.1f}bp\nn={int(rb['n'])}, t={rb['t']}",
                (kbest, rb["mean_bp"]), fontsize=8, color="#c0392b", ha="right",
                va="top", xytext=(-6, -4), textcoords="offset points")
    ax.set_title(NAME[code], fontsize=11)
    ax.set_xlabel("阈值 k（价格跳;|上一分钟涨跌|>k×0.2 才交易）")
    ax.set_ylabel("每笔平均收益（bp，毛）")
    ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper left")
fig.suptitle("图14　开盘反转的「最优阈值」是过拟合（开盘09:30–59，2个月，最大化每笔均值）\n"
             "k* 全落在边界(37–40)、n 仅 6–116、t≈1;且 H1/H2 两半在高 k 处发散 → 无稳定阈值",
             fontsize=13, fontweight="bold")
fig.text(0.5, 0.005, "低 k(样本大)三条线≈0且一致;高 k 处均值看似暴涨,只是筛到极少数(n<50)大反转、纯噪音。"
         "毛收益。", ha="center", fontsize=8.5, color="0.35")
fig.tight_layout(rect=(0, 0.03, 1, 0.95))
fig.savefig(f"{OUT}/fig14_最优阈值过拟合.png", dpi=150)
print("saved", f"{OUT}/fig14_最优阈值过拟合.png")
