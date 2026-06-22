"""minute_curves_raw_summary_plot.py — two summary figures across ALL month windows.
fig22: full grid of every month's RAW un-normalized curve (per-panel zoom).
fig23: regime time-series — terminal within-minute drift (stable region) per contract,
       month by month, to show how the intra-minute effect evolves across regimes.
Run with system python3:  python3 minute_curves_raw_summary_plot.py
"""
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
NAME = {"IC0000": "IC 中证500", "IF0000": "IF 沪深300", "IH0000": "IH 上证50", "IM0000": "IM 中证1000"}
COL = {"IC0000": "#c0392b", "IF0000": "#e08a3c", "IH0000": "#27ae60", "IM0000": "#4C72B0"}
COVER, STABLE = 0.20, 0.50

files = sorted(glob.glob(f"{INDIR}/minute_curves_raw_*.csv"),
               key=lambda f: re.search(r"(\d{4})_(\d{2})", f).group(0))
months = [re.search(r"(\d{4})_(\d{2})", f).group(0).replace("_", "-") for f in files]
tabs = {m: pd.read_csv(f) for m, f in zip(months, files)}
print(f"{len(months)} months: {', '.join(months)}")


def draw(ax, tab):
    ymax = 0.0
    for code in NAME:
        s = tab[tab.code == code].sort_values("tick")
        if s.empty:
            continue
        x, y, n = s["tick"].to_numpy(), s["mean"].to_numpy(), s["n"].to_numpy()
        cut = int(x[n >= COVER * n.max()].max())
        stab = int(x[n >= STABLE * n.max()].max())
        ymax = max(ymax, np.nanmax(np.abs(y[:stab])))
        ax.plot(x[:cut], y[:cut], color=COL[code], lw=1.3)
        ax.plot(x[cut - 1:], y[cut - 1:], color=COL[code], lw=0.8, ls="--", alpha=0.3)
    ax.axhline(0, color="0.55", lw=.5)
    ax.set_xlim(1, tab["tick"].max())
    return ymax


# ---------- fig22: full grid, per-panel zoom ----------
ncols = 4
nrows = int(np.ceil(len(months) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.4, nrows * 2.5), squeeze=False)
for i, ax in enumerate(axes.ravel()):
    if i >= len(months):
        ax.axis("off"); continue
    ymax = draw(ax, tabs[months[i]])
    ax.set_ylim(-ymax * 1.15, ymax * 1.15)
    ax.set_title(f"{months[i]}  (±{ymax:.2f}点)", fontsize=9.5, fontweight="bold")
    ax.grid(True, alpha=0.2); ax.tick_params(labelsize=6.5)
handles = [plt.Line2D([], [], color=COL[c], lw=2, label=NAME[c]) for c in NAME]
fig.legend(handles=handles, ncol=4, loc="lower center", fontsize=10, bbox_to_anchor=(0.5, 0.004))
fig.suptitle(f"图22　动量择时(信号±1)·全分钟平均·未归一化 — {len(months)}个月全览（各图独立量程,见标题）",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=(0, 0.035, 1, 0.97))
fig.savefig(f"{OUT}/fig22_全月份全览_未归一化.png", dpi=125); plt.close(fig)
print("saved fig22")

# ---------- fig23: terminal within-minute drift per contract over time ----------
drift = {c: [] for c in NAME}
for m in months:
    tab = tabs[m]
    for code in NAME:
        s = tab[tab.code == code].sort_values("tick")
        if s.empty:
            drift[code].append(np.nan); continue
        x, y, n = s["tick"].to_numpy(), s["mean"].to_numpy(), s["n"].to_numpy()
        stab = int(x[n >= STABLE * n.max()].max())     # last reliable tick
        drift[code].append(float(y[stab - 1]))          # net drift by end of minute
fig, ax = plt.subplots(figsize=(12, 5.6))
xp = np.arange(len(months))
for code in NAME:
    ax.plot(xp, drift[code], marker="o", ms=4, color=COL[code], lw=1.6, label=NAME[code])
ax.axhline(0, color="0.4", lw=.8)
ax.set_xticks(xp); ax.set_xticklabels(months, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("分钟末(稳定区)相对开盘的净漂移（指数点）", fontsize=10)
ax.set_title("图23　动量择时(信号±1)·分钟内净漂移 随时间演变（各合约;指数点;2020–2025）",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="best"); ax.grid(True, alpha=0.3)
fig.text(0.5, 0.005, "每点=该(月,合约)全部分钟内 sign(上一分钟)×(price−开盘) 在稳定区(覆盖≥50%)末端的均值。"
         "正=动量延续(信号方向继续走),负=反转(被上一分钟方向打脸)。毛口径,指数点(含价位与波动影响)。",
         ha="center", fontsize=8, color="0.4")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(f"{OUT}/fig23_净漂移随时间_未归一化.png", dpi=130); plt.close(fig)
print("saved fig23")
for m, *_ in zip(months):
    pass
print("\nterminal drift (pts) by month:")
print(pd.DataFrame(drift, index=months).round(3).to_string())
