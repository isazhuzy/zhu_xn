"""minute_of_day_persist_plot.py — cross-month mean momentum P&L per minute-of-day,
with minutes that are all-K-same-sign marked. AM/PM panels, 4 contracts. Run: python3 ..."""
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

OUT = "/Users/zhuisabella/xn/intraminute/figs"
NAME = {"IC0000": "IC 中证500", "IF0000": "IF 沪深300", "IH0000": "IH 上证50", "IM0000": "IM 中证1000"}
COL = {"IC0000": "#c0392b", "IF0000": "#e08a3c", "IH0000": "#27ae60", "IM0000": "#4C72B0"}
df = pd.read_csv("/Users/zhuisabella/xn/intraminute/minute_of_day_pnl_multi.csv")
K = df.ym.nunique()

fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), sharey=True)
for ax, (lo, hi, ttl) in zip(axes, [(570, 690, "上午 09:30–11:30"), (780, 900, "下午 13:00–15:00")]):
    for code in NAME:
        p = df[df.code == code].pivot_table(index="tod", columns="ym", values="mean").dropna()
        p = p[(p.index >= lo) & (p.index <= hi)]
        cm = p.mean(axis=1)
        allsame = ((np.sign(p.values) > 0).mean(axis=1) == 1) | ((np.sign(p.values) < 0).mean(axis=1) == 1)
        ax.plot(p.index, cm, color=COL[code], lw=1.0, alpha=0.8, label=NAME[code])
        ax.scatter(p.index[allsame], cm[allsame], color=COL[code], s=34, zorder=5, edgecolor="k", linewidth=0.4)
    ax.axhline(0, color="0.4", lw=.8)
    ax.axvspan(lo, lo + 5, color="gold", alpha=0.18)
    ax.set_title(ttl, fontsize=11, fontweight="bold")
    ticks = list(range(lo, hi + 1, 15))
    ax.set_xticks(ticks); ax.set_xticklabels([f"{t//60:02d}:{t%60:02d}" for t in ticks], fontsize=8)
    ax.grid(True, alpha=0.25)
axes[0].set_ylabel(f"{K}个月 动量P&L 月均（指数点/分钟）", fontsize=10)
axes[0].legend(fontsize=8, loc="best")
fig.suptitle(f"图31　各分钟 动量P&L · {K}个平静月 月均（圈点={K}个月全部同号;金=开盘前5分钟）",
             fontsize=13, fontweight="bold")
fig.text(0.5, 0.005, "圈点=该分钟在全部6个月都同号(持续);多数为负=动量亏→反转(fade)赚。圈点数(8–10/合约)≈随机预期(~7),"
         "整体持续性弱;少数分钟跨合约同现(10:06、09:43)是仅有的线索。", ha="center", fontsize=8, color="0.4")
fig.tight_layout(rect=(0, 0.03, 1, 0.96))
fig.savefig(f"{OUT}/fig31_分钟P&L持续性.png", dpi=130); plt.close(fig)
print("saved fig31")
