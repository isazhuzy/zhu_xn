"""replication_plot.py — fig13 from replication_sample.csv (python3.14).
跨年复现:固定 N=16(2023两月最优进场延迟),各年1–2月 开盘反转「每笔平均收益率(%)」。
2016 剔除(受限期数据异常:IC +14.6 / IF −14.3 bp级,显然脏)。"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
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
tab = pd.read_csv("/Users/zhuisabella/xn/ticker/open_breakdown/replication_sample.csv",
                  dtype={"year": str})
tab = tab[(tab.year != "2016") & (tab.N == 16)]
tab["pct"] = tab["mean_bp"] / 100.0
years = sorted(tab.year.unique())
x = np.arange(len(years))

fig, ax = plt.subplots(figsize=(11, 6.2))
for code in NAME:
    s = tab[tab.code == code].set_index("year").reindex(years)
    lw = 2.6 if code == "IH0000" else 1.4
    ax.plot(x, s["pct"], "-o", color=COL[code], ms=6, lw=lw, label=NAME[code])
    sig = (s["t"].abs() >= 2).fillna(False)
    ax.scatter(x[sig.values], s["pct"][sig.values], s=150, facecolors="none",
               edgecolors=COL[code], linewidths=2.2, zorder=5)
ax.axhline(0, color="k", lw=.9)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.4f}%"))
ax.set_xticks(x, years)
ax.set_xlabel("年份（各年 1–2 月窗口）")
ax.set_ylabel("每笔平均收益率（%）")
ax.set_title("每分钟反转跨年复现:固定 N=16(2023两月最优进场延迟)，各年每笔平均收益率\n"
             "符号随年份翻转:2019 显著为负(亏)、2022≈0、2023–25 转正 → 非稳定、随体制漂移",
             fontsize=12.5, fontweight="bold")
ax.grid(axis="y", alpha=.3)
ax.legend(fontsize=9, loc="upper left")
fig.text(0.5, 0.01, "空心圈=该点 |t|≥2(显著)。N=16 是用 2023 两月挑出的「最优」进场延迟,但 2019 上显著为负——"
         "样本内最优不外推。IM 中证1000 2022年中才上市。2016 剔除。纵轴=每笔平均收益率(%)，毛口径。",
         ha="center", fontsize=8.2, color="0.35")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(f"{OUT}/fig13_跨年复现.png", dpi=150)
print("saved", f"{OUT}/fig13_跨年复现.png")
