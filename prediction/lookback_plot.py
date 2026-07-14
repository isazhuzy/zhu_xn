"""lookback_plot.py — fig113: mirror of fig111. Vary the LOOKBACK (即时→20min), predict
next 2s. Three panels VOI/OIR/MPB, each 4 contracts. Shows the cliff: signal is
instantaneous, any accumulation window (20s+) kills it.
Run: python3 lookback_plot.py   (SUF=_pilot for pilot files)
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
D = "/Users/zhuisabella/xn/prediction"
SUF = os.environ.get("SUF", "")
NM = {"IC0000": "IC 中证500", "IF0000": "IF 沪深300", "IH0000": "IH 上证50", "IM0000": "IM 中证1000"}
COL = {"IC0000": "#c0392b", "IF0000": "#e08a3c", "IH0000": "#27ae60", "IM0000": "#4C72B0"}
SEC = {"2s(即时)": 2, "20s": 20, "1min": 60, "5min": 300, "15min": 900, "20min": 1200}
g = pd.read_csv(f"{D}/lookback_grid{SUF}.csv")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, fac in zip(axes, ["VOI", "OIR", "MPB"]):
    for code in NM:
        s = g[(g.code == code) & (g.factor == fac)].copy()
        s["sec"] = s.look.map(SEC); s = s.sort_values("sec")
        ax.plot(s.sec, s.r2_oos, marker="o", ms=5, lw=1.8, color=COL[code], label=NM[code])
    ax.axhline(0, color="k", lw=.8)
    ax.set_xscale("log"); ax.set_xticks([2, 20, 60, 300, 900, 1200])
    ax.set_xticklabels(["即时", "20s", "1min", "5min", "15min", "20min"], fontsize=8.5, rotation=30)
    ax.set_xlabel("回看累积窗口 L", fontsize=9.5); ax.set_ylabel("预测下一2s收益 OOS R²", fontsize=9.5)
    ax.set_title(f"{fac}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=.25)
fig.suptitle("图113　fig111的镜像：固定往前2s，改变往回累积窗口 20s→20min —— 信号是即时的,任何窗口累积都抹掉它",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{D}/fig113_回看窗口{SUF}.png", dpi=135)
print("saved fig113")
