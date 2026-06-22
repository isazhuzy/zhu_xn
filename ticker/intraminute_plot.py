"""intraminute_plot.py — fig11 from intraminute_wholeday.csv (python3.14).
进场延迟 N 与开盘反转「每笔平均收益率(%)」,2个月,各合约(无组合)。决定 N 进场是否重要。"""
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
tab = pd.read_csv("/Users/zhuisabella/xn/ticker/open_breakdown/intraminute_wholeday.csv")
N_LIST = sorted(tab["N"].unique())

fig, ax = plt.subplots(figsize=(10.5, 6.2))
for code in NAME:
    s = tab[tab.group == code].sort_values("N")
    lw = 2.6 if code == "IH0000" else 1.4
    ax.plot(s["N"], s["mean_bp"] / 100.0, "-o", color=COL[code], ms=5, lw=lw, label=NAME[code])
ax.axhline(0, color="0.5", lw=.8)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.4f}%"))
ax.set_xlabel("进场延迟 N（在该分钟第 N 个 tick 进场；持有整一分钟到下一分钟第 N tick）")
ax.set_ylabel("每笔平均收益率（%）")
ax.set_xticks(N_LIST)
ax.set_title("图11　进场延迟 N 对开盘反转收益率的影响（2个月 2023.01–02，各合约）\n"
             "曲线对 N 基本平 → 第几个 tick 进场几乎不影响收益;只有 IH 上证50 略为正",
             fontsize=12.5, fontweight="bold")
ax.grid(alpha=.3)
ax.legend(fontsize=9, ncol=2)
fig.text(0.5, 0.01, "信号=上一分钟涨跌、反向开仓、持有整一分钟。纵轴为每笔平均收益率(%)，毛口径(未扣价差/手续费)。",
         ha="center", fontsize=8.5, color="0.35")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(f"{OUT}/fig11_全日逐分钟tick扫描.png", dpi=150)
print("saved", f"{OUT}/fig11_全日逐分钟tick扫描.png")
