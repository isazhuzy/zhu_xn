"""plot the VOI-sorted cumulative price-change curve.

Reads xn/manual/voi_cumsum_curve[_pilot].csv (from voi_cumsum.py).
x = VOI rank percentile (0..100), y = cumulative sum of future price change (ticks). 
Run: SUF=_pilot /Users/zhuisabella/xn/.venv/bin/python voi_cumsum_plot.py
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

D = "/Users/zhuisabella/xn/last"
SUF = os.environ.get("SUF", "")
NAME = {"IC0000": "IC 中证500", "IF0000": "IF 沪深300",
        "IH0000": "IH 上证50", "IM0000": "IM 中证1000"}
KCOL = {1: "0.6", 20: "#4C72B0", 120: "#c0392b"}      # horizon -> colour
KLAB = {1: "k=1 (0.5秒)", 20: "k=20 (10秒)", 120: "k=120 (60秒)"}

cv = pd.read_csv(f"/Users/zhuisabella/xn/manual/voi_cumsum_curve{SUF}.csv")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, code in zip(axes.ravel(), NAME):
    s = cv[cv.code == code].sort_values("rank")
    if s.empty:
        ax.set_axis_off(); continue
    x = s.q * 100
    for k in (1, 20, 120):
        ax.plot(x, s[f"cum{k}"], color=KCOL[k], lw=1.8, label=KLAB[k])
    # shade the VOI==0 zone (huge mass of "nothing happened" ticks)
    z = s[s.voi == 0]
    if len(z):
        ax.axvspan(z.q.min() * 100, z.q.max() * 100, color="0.85", alpha=.5,
                   label="VOI=0 区间")
    ax.axhline(0, color="k", lw=.6)
    # mark the bottom of the check mark for the main horizon k=20
    imin = s["cum20"].idxmin()
    ax.plot(s.q[imin] * 100, s.cum20[imin], "v", color="#4C72B0", ms=7)
    ax.annotate(f"最低点 VOI≈{s.voi[imin]:.0f}", (s.q[imin] * 100, s.cum20[imin]),
                textcoords="offset points", xytext=(6, -12), fontsize=8, color="0.3")
    # top axis: what VOI value sits at each rank percentile
    top = ax.secondary_xaxis("top")
    qs = [1, 10, 30, 50, 70, 90, 99]
    vals = np.interp(qs, s.q * 100, s.voi)
    top.set_xticks(qs); top.set_xticklabels([f"{v:.0f}" for v in vals], fontsize=7.5)
    top.set_xlabel("该分位处的 VOI 值（手）", fontsize=8, color="0.35")
    ax.set_title(f"{NAME[code]}   n={s.n_total.iloc[0]:,}", fontsize=10.5,
                 fontweight="bold")
    ax.set_xlabel("VOI 排序分位（%）", fontsize=9)
    ax.set_ylabel("未来价格变动的累计和（tick）", fontsize=9)
    ax.legend(fontsize=8, loc="upper center"); ax.grid(True, alpha=.25)
fig.suptitle("VOI 从小到大排序 → 未来k个快照价格变动的累计和",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(f"{D}/fig_voi_cumsum{SUF}.png", dpi=135); plt.close(fig)
print(f"saved fig_voi_cumsum{SUF}.png")
