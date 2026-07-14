"""factor_lookback_plot.py — fig116: matched LOOK-BACK mirror of fig115.
Fixed forward = 2s; vary look-back 0.5s(即时)→20min; single-factor OOS R^2 for VOI/OIR/MPB.
Run: python3 factor_lookback_plot.py   (SUF=_pilot)
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
SEC = {"0.5s": 0.5, "2s": 2, "10s": 10, "60s": 60, "5min": 300, "20min": 1200}
r = pd.read_csv(f"{D}/flb2_results{SUF}.csv")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, fac in zip(axes, ["VOI", "OIR", "MPB"]):
    for code in NM:
        s = r[(r.code == code) & (r.factor == fac)].copy()
        s["sec"] = s.look.map(SEC); s = s.sort_values("sec")
        ax.plot(s.sec, s.r2_oos, marker="o", ms=5, lw=1.8, color=COL[code], label=NM[code])
    ax.axhline(0, color="k", lw=.8)
    ax.set_xscale("log"); ax.set_xticks([0.5, 2, 10, 60, 300, 1200])
    ax.set_xticklabels(["即时", "2s", "10s", "60s", "5min", "20min"], fontsize=8.5, rotation=25)
    ax.set_xlabel("回看累积窗口 L（前看固定2s）", fontsize=9.5); ax.set_ylabel(f"{fac} 单因子 OOS R²", fontsize=9.5)
    ax.set_title(f"{fac}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=.25)
fig.suptitle("图116　fig115的回看镜像（同定义）：固定前看2s，扫回看 即时→20min —— OIR一累积就死，VOI累积更耐用",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{D}/fig116_三因子回看{SUF}.png", dpi=135)
print("saved fig116")
