"""fullminute_extended_plot.py — fig13 from fullminute_extended.csv (python3.14)."""
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
tab = pd.read_csv("/Users/zhuisabella/xn/ticker/open_breakdown/fullminute_extended.csv")
periods = list(dict.fromkeys(tab["period"]))
N_LIST = sorted(tab["N"].unique())
cmap = plt.get_cmap("turbo")
colors = {p: cmap(i / max(len(periods) - 1, 1)) for i, p in enumerate(periods)}

fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
for p in periods:
    s = tab[tab.period == p].sort_values("N")
    lab = f"{p} ({s['contracts'].iloc[0]}, {int(s['days'].iloc[0])}d)"
    a1.plot(s["N"], s["mean_bp"], "-o", color=colors[p], ms=5, lw=1.8, label=lab)
    a2.plot(s["N"], s["win"] * 100, "-o", color=colors[p], ms=5, lw=1.8, label=lab)
a1.axhline(0, color="k", lw=.8); a2.axhline(50, color="0.5", lw=1, ls=":")
a1.set_ylabel("反转每笔平均收益（bp，毛）")
a2.set_ylabel("反转胜率（%）")
a2.set_xlabel("进场延迟 N（整体平移N tick；持有恒为整一分钟；N=1≈分钟边界反转）")
a2.set_xticks(N_LIST)
for ax in (a1, a2):
    ax.grid(alpha=.3); ax.legend(fontsize=8, ncol=2)
fig.suptitle("图13　全分钟持有反转(整体平移N tick) × 选择性扩展样本(2016–2025)：结论是否跨年份成立",
             fontsize=13, fontweight="bold")
fig.text(0.5, 0.01, "信号=锚点前一整分钟涨跌、反向开仓、持有整一分钟。各时段为跨年抽样窗口；"
         "早期窗口无IM(中证1000，2022年中上市)。t按交易日聚合，幅度均<价差。毛收益。",
         ha="center", fontsize=8.5, color="0.35")
fig.tight_layout(rect=(0, 0.03, 1, 0.96))
fig.savefig(f"{OUT}/fig13_全分钟扩展样本.png", dpi=150)
print("saved", f"{OUT}/fig13_全分钟扩展样本.png")
