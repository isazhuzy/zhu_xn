"""minute_curves_grand_all_plot.py — ALL-MONTHS grand-average intra-minute curve.
Combines every month in raw_months/ into a single line per contract (like fig19/fig20,
but pooled across 2020-01..2026-05). Buy-and-hold, raw tick index (no normalization).
Per tick: weighted mean of each month's mean by its sample count n  ->  true history mean.
Run with system python3:  python3 minute_curves_grand_all_plot.py
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
               key=lambda f: re.search(r"\d{4}_\d{2}", f).group())
months = [re.search(r"\d{4}_\d{2}", f).group() for f in files]
big = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

fig, ax = plt.subplots(figsize=(9.5, 5.6))
ymax = 0.0
stats = {}
for code in NAME:
    s = big[big.code == code]
    if s.empty:
        continue
    s = s.assign(wm=s["mean"] * s["n"])
    g = s.groupby("tick").agg(wm=("wm", "sum"), n=("n", "sum"))
    g["mean"] = g["wm"] / g["n"]                       # history-wide weighted mean per tick
    x, y, n = g.index.to_numpy(), g["mean"].to_numpy(), g["n"].to_numpy()
    cut = int(x[n >= COVER * n.max()].max())
    stab = int(x[n >= STABLE * n.max()].max())
    ymax = max(ymax, np.nanmax(np.abs(y[:stab])))
    nmon = big[big.code == code].groupby("tick").size().max()
    ax.plot(x[:cut], y[:cut], color=COL[code], lw=1.9, label=f"{NAME[code]}  ({nmon}个月)")
    ax.plot(x[cut - 1:], y[cut - 1:], color=COL[code], lw=1.0, ls="--", alpha=0.30)
    stats[code] = (y[0], y[stab - 1], stab, nmon)
ax.axhline(0, color="0.55", lw=.7)
ax.set_xlim(1, 125)                      # focus on the real minute; dense-month tail clips off
ax.set_ylim(-ymax * 1.15, ymax * 1.15)
ax.set_xlabel("分钟内 tick 序号 1→N（真实 tick,未做时间归一化）", fontsize=10)
ax.set_ylabel("相对分钟开盘的价格变动（指数点）", fontsize=10)
ax.set_title("图26　动量择时(信号±1)·全分钟·全历史平均·未归一化（指数点;2020-01–2026-05;AM+PM）",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="best")
ax.grid(True, alpha=0.25)
fig.text(0.5, 0.005, "动量择时:上一分钟涨→本分钟做多、跌→做空(整条×sign)。每条=该合约全部月份、全部分钟内曲线按tick序号对齐后的样本量加权平均。"
         "实线=覆盖≥20%;虚线=尾部样本变薄。毛口径。",
         ha="center", fontsize=8, color="0.4")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(f"{OUT}/fig26_全历史平均_未归一化.png", dpi=130); plt.close(fig)
print("saved fig26")
for code, (a, b, stab, nmon) in stats.items():
    print(f"{code}: {nmon} months, tick1={a:+.4f} -> tick{stab}={b:+.4f} pts")
