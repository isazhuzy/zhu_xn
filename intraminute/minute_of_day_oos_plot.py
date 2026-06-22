"""minute_of_day_oos_plot.py — per-contract in-sample vs out-of-sample minute P&L.
One panel per contract: x = IS mean P&L per minute, y = OOS mean P&L. If per-minute edges
were real, points line up on the diagonal (positive corr). Distinctive IS minutes circled.
Run: python3 minute_of_day_oos_plot.py"""
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
df = pd.read_csv("/Users/zhuisabella/xn/intraminute/minute_of_day_pnl_isoos.csv")

fig, axes = plt.subplots(2, 2, figsize=(13, 12))
for ax, code in zip(axes.ravel(), NAME):
    c = df[df.code == code]
    IS = c[c["sample"] == "IS"].pivot_table(index="tod", columns="ym", values="mean").dropna()
    OOS = c[c["sample"] == "OOS"].pivot_table(index="tod", columns="ym", values="mean").dropna()
    common = IS.index.intersection(OOS.index)
    IS, OOS = IS.loc[common], OOS.loc[common]
    ism, oosm = IS.mean(axis=1).values, OOS.mean(axis=1).values
    ist = IS.mean(axis=1).values / (IS.std(axis=1).values / np.sqrt(IS.shape[1]))
    allsame = ((np.sign(IS.values) > 0).mean(1) == 1) | ((np.sign(IS.values) < 0).mean(1) == 1)
    sel = allsame & (np.abs(ist) > 3)
    r = np.corrcoef(ism, oosm)[0, 1]
    ax.axhline(0, color="0.6", lw=.7); ax.axvline(0, color="0.6", lw=.7)
    lim = max(np.abs(ism).max(), np.abs(oosm).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], color="0.5", ls="--", lw=.8)        # y=x: perfect carry-over
    ax.scatter(ism, oosm, s=12, color="0.6", alpha=0.5)
    ax.scatter(ism[sel], oosm[sel], s=55, color=COL[code], edgecolor="k", linewidth=0.5, zorder=5,
               label=f"挑出的显著分钟 ({sel.sum()})")
    for i in np.where(sel)[0]:
        t = int(common[i]); ax.annotate(f"{t//60:02d}:{t%60:02d}", (ism[i], oosm[i]), fontsize=6.5,
                                        xytext=(3, 3), textcoords="offset points")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_title(f"{NAME[code]}　样本内↔样本外 相关 r={r:+.2f}", fontsize=12, fontweight="bold")
    ax.set_xlabel("样本内 月均P&L (指数点)", fontsize=9)
    ax.set_ylabel("样本外 月均P&L (指数点)", fontsize=9)
    ax.legend(fontsize=8, loc="upper left"); ax.grid(True, alpha=0.2)
fig.suptitle("图32　各合约 分钟P&L 样本内 vs 样本外（IS=2023-05..2025-06 6月; OOS=2023-10..2025-10 6月）",
             fontsize=14, fontweight="bold")
fig.text(0.5, 0.005, "每点=一个minute-of-day。若分钟级edge真实,点应沿对角线(r>0);实际r≈0=云团→样本内挑出的"
         "显著分钟样本外随机化(甚至反号),即过拟合噪声。", ha="center", fontsize=8.5, color="0.4")
fig.tight_layout(rect=(0, 0.025, 1, 0.97))
fig.savefig(f"{OUT}/fig32_分钟P&L_样本内外.png", dpi=130); plt.close(fig)
print("saved fig32")
for code in NAME:
    c = df[df.code == code]
    IS = c[c["sample"] == "IS"].pivot_table(index="tod", columns="ym", values="mean").dropna()
    OOS = c[c["sample"] == "OOS"].pivot_table(index="tod", columns="ym", values="mean").dropna()
    common = IS.index.intersection(OOS.index)
    r = np.corrcoef(IS.loc[common].mean(1), OOS.loc[common].mean(1))[0, 1]
    print(f"{code}: IS↔OOS per-minute corr = {r:+.3f}")
