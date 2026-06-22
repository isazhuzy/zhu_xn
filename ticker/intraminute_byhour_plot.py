"""intraminute_byhour_plot.py — fig12 from intraminute_byhour.csv (python3.14).
Prev-minute reversal (full-minute hold, N=10), per contract × 30-min time bucket."""
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
ORDER = ["IC0000", "IF0000", "IH0000", "IM0000"]
NAME = {"IC0000": "IC 中证500", "IF0000": "IF 沪深300", "IH0000": "IH 上证50", "IM0000": "IM 中证1000"}
tab = pd.read_csv("/Users/zhuisabella/xn/ticker/open_breakdown/intraminute_byhour.csv")
M = tab.pivot(index="code", columns="bucket", values="mean_bp").reindex(ORDER)
buckets = list(M.columns)

fig, ax = plt.subplots(figsize=(11, 4.2))
vmax = 0.5
im = ax.imshow(M.values, cmap="RdYlGn_r", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(len(buckets)), buckets)
ax.set_yticks(range(len(ORDER)), [NAME[c] for c in ORDER])
ax.set_xlabel("日内时段（30分钟）"); ax.set_ylabel("合约")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8)
cb = fig.colorbar(im, ax=ax, pad=.01)
cb.set_label("反转每笔平均收益（bp）　红=正(反转盈利) / 绿=负(延续/亏损)")
ax.set_title("图12　反转(上一分钟信号,整分钟持有,N=10)的日内时段分布（2个月，各合约）\n"
             "IH 上证50 的反转集中在早盘(09:30–10:30,红);IC/IM 开盘反而偏延续(绿)",
             fontsize=12, fontweight="bold", pad=14)
fig.text(0.5, 0.02, "信号=上一分钟涨跌、反向开仓、持有整一分钟。仅作时段画像;跨年并不稳定(见 fig13)。"
         "幅度均<买卖价差。毛收益。", ha="center", fontsize=8.5, color="0.35")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(f"{OUT}/fig12_动量日内分布.png", dpi=150)
print("saved", f"{OUT}/fig12_动量日内分布.png")
