"""
Reads xn/manual/mpb_cumsum_curve[_pilot].csv (made by xn/manual/voi_cumsum.py
with FACTOR=mpb). 

Run: SUF=_pilot /Users/zhuisabella/xn/.venv/bin/python mpb_cumsum_plot.py
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
XRAW = os.environ.get("XRAW") == "1"
NAME = {"IC0000": "IC 中证500", "IF0000": "IF 沪深300",
        "IH0000": "IH 上证50", "IM0000": "IM 中证1000"}

cv = pd.read_csv(f"/Users/zhuisabella/xn/manual/mpb_cumsum_curve{SUF}.csv")
ks = sorted(int(c[3:]) for c in cv.columns if c.startswith("cum"))
pal = ["0.6", "#27ae60", "#e08a3c", "#4C72B0", "#c0392b", "#8e44ad"]
kcol = {k: pal[i % len(pal)] for i, k in enumerate(ks)}
kmain = ks[-1]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, code in zip(axes.ravel(), NAME):
    s = cv[cv.code == code].sort_values("rank")
    if s.empty:
        ax.set_axis_off(); continue
    x = s.mpb if XRAW else s.q * 100
    for k in ks:
        ax.plot(x, s[f"cum{k}"], color=kcol[k], lw=1.8,
                label=f"k={k} ({k*0.5:g}秒)")
    ax.axhline(0, color="k", lw=.6)
    imin = s[f"cum{kmain}"].idxmin()
    ax.plot(x[imin], s[f"cum{kmain}"][imin], "v", color=kcol[kmain], ms=7)
    if XRAW:
        # fat tails (bad prints reach ±60 ticks): linear within ±1 tick, log beyond
        ax.set_xscale("symlog", linthresh=1)
        ax.axvline(0, color="0.7", lw=.8)
        ax.set_xlabel("MPB（tick）｜±1内线性，两侧对数", fontsize=9)
    else:
        top = ax.secondary_xaxis("top")
        qs = [1, 10, 30, 50, 70, 90, 99]
        vals = np.interp(qs, s.q * 100, s.mpb)
        top.set_xticks(qs)
        top.set_xticklabels([f"{v:.2f}" for v in vals], fontsize=7.5)
        top.set_xlabel("该分位处的 MPB 值（tick）", fontsize=8, color="0.35")
        ax.set_xlabel("MPB 排序分位（%）", fontsize=9)
    ax.set_title(f"{NAME[code]}   n={s.n_total.iloc[0]:,}", fontsize=10.5,
                 fontweight="bold")
    ax.set_ylabel("未来价格变动cumsum（tick）", fontsize=9)
    ax.legend(fontsize=8, loc="upper center"); ax.grid(True, alpha=.25)
fig.suptitle("MPB 排序 → 未来k个快照价格变动的累积和",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.94))
raw = "_raw" if XRAW else ""
fig.savefig(f"{D}/fig_mpb_cumsum{raw}{SUF}.png", dpi=135); plt.close(fig)
print(f"saved fig_mpb_cumsum{raw}{SUF}.png  (k = {ks})")
