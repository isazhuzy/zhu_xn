"""minute_curves_raw_multi_plot.py — draw fig20-style RAW curves for every month CSV in
raw_months/, plus one comparison grid across all windows.
Run with system python3:  python3 minute_curves_raw_multi_plot.py
"""
import os
import glob
import re
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

INDIR = "/Users/zhuisabella/xn/intraminute/raw_months"
OUT = "/Users/zhuisabella/xn/intraminute/figs"
ODIR = f"{OUT}/raw_months"; os.makedirs(ODIR, exist_ok=True)
NAME = {"IC0000": "IC 中证500", "IF0000": "IF 沪深300", "IH0000": "IH 上证50", "IM0000": "IM 中证1000"}
COL = {"IC0000": "#c0392b", "IF0000": "#e08a3c", "IH0000": "#27ae60", "IM0000": "#4C72B0"}
COVER = 0.20      # solid line while n >= 20% of peak coverage
STABLE = 0.50     # scale y-axis to where n >= 50% (ignore the noisy thinning end)

files = sorted(glob.glob(f"{INDIR}/minute_curves_raw_*.csv"))
months = [re.search(r"(\d{4})_(\d{2})", f).group(0).replace("_", "-") for f in files]


def draw(ax, tab):
    ymax = 0.0
    for code in NAME:
        s = tab[tab.code == code].sort_values("tick")
        if s.empty:
            continue
        x, y, n = s["tick"].to_numpy(), s["mean"].to_numpy(), s["n"].to_numpy()
        cut = int(x[n >= COVER * n.max()].max())           # last drawn (solid) tick
        stab = int(x[n >= STABLE * n.max()].max())          # last tick used for scaling
        ymax = max(ymax, np.nanmax(np.abs(y[:stab])))
        ax.plot(x[:cut], y[:cut], color=COL[code], lw=1.5, label=NAME[code])
        ax.plot(x[cut - 1:], y[cut - 1:], color=COL[code], lw=0.9, ls="--", alpha=0.30)
    ax.axhline(0, color="0.55", lw=.6)
    ax.set_xlim(1, tab["tick"].max())
    return ymax


# individual figures
for f, mlabel in zip(files, months):
    tab = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ymax = draw(ax, tab)
    ax.set_ylim(-ymax * 1.15, ymax * 1.15)
    ax.set_xlabel("分钟内 tick 序号（真实 tick,未归一化）", fontsize=10)
    ax.set_ylabel("相对分钟开盘的价格变动（指数点）", fontsize=10)
    ax.set_title(f"动量择时(信号±1)·全分钟平均·未归一化 — {mlabel}（指数点;AM+PM）",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="best"); ax.grid(True, alpha=0.25)
    fname = f"{ODIR}/fig20_{mlabel.replace('-', '_')}_未归一化.png"
    fig.tight_layout(); fig.savefig(fname, dpi=120); plt.close(fig)
    print("saved", fname)

# comparison grid — EACH panel auto-scaled to its own data (zoomed in)
tabs = [pd.read_csv(f) for f in files]
ncols = 3
nrows = int(np.ceil(len(files) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.0, nrows * 3.1), squeeze=False)
for i, ax in enumerate(axes.ravel()):
    if i >= len(files):
        ax.axis("off"); continue
    ymax = draw(ax, tabs[i])
    ax.set_ylim(-ymax * 1.15, ymax * 1.15)               # per-panel zoom
    ax.set_title(f"{months[i]}  (±{ymax:.2f}点)", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2); ax.tick_params(labelsize=7)
handles = [plt.Line2D([], [], color=COL[c], lw=2, label=NAME[c]) for c in NAME]
fig.legend(handles=handles, ncol=4, loc="lower center", fontsize=10, bbox_to_anchor=(0.5, 0.002))
fig.suptitle("图21　动量择时(信号±1)·全分钟平均·未归一化 — 跨年月对比（各图独立量程,见标题;实线=覆盖≥20%）",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=(0, 0.05, 1, 0.97))
fname = f"{OUT}/fig21_跨年月对比_未归一化.png"
fig.savefig(fname, dpi=130); plt.close(fig)
print("saved", fname)
