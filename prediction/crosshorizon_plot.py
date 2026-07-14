"""crosshorizon_plot.py — fig117: does cross-contract prediction survive at long horizons?
Per target contract: OOS R^2 vs horizon (20s..20min), own-OFI vs +cross-OFI. Includes the
1s reference (from crossimpact) where cross tripled own — to show the advantage is seconds-only.
Run: python3 crosshorizon_plot.py   (SUF=_pilot)
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
NM = {"IC": "IC 中证500", "IF": "IF 沪深300", "IH": "IH 上证50", "IM": "IM 中证1000"}
r = pd.read_csv(f"{D}/crosshor_results{SUF}.csv")
# 1s reference from crossimpact (own vs full OOS R², W=1s)
ref = None
try:
    xp = pd.read_csv(f"{D}/xc_predict.csv"); xp = xp[xp.W == 1]
    ref = {c[:2]: (xp[(xp.target == c) & (xp.model == "own")].r2_oos.iloc[0],
                   xp[(xp.target == c) & (xp.model == "full")].r2_oos.iloc[0]) for c in
           ["IC0000", "IF0000", "IH0000", "IM0000"]}
except Exception:
    pass

fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), sharey=True)
for ax, tc in zip(axes, ["IC", "IF", "IH", "IM"]):
    for model, c, ls, lab in [("own", "0.55", "--", "仅自身OFI"), ("full", "#c0392b", "-", "+跨合约OFI")]:
        s = r[(r.target == tc) & (r.model == model)].sort_values("hor_s")
        ax.plot(s.hor_s, s.r2_oos, marker="o", ms=5, lw=1.8, color=c, ls=ls, label=lab)
    if ref and tc in ref:
        ax.scatter([1, 1], ref[tc], marker="*", s=140, color=["0.55", "#c0392b"], zorder=5)
        ax.annotate("1s参照(crossimpact):\n跨OFI翻2-3倍", xy=(1, ref[tc][1]), xytext=(1.3, 0.045),
                    fontsize=7.5, color="0.35", arrowprops=dict(arrowstyle="->", color="0.6"))
    ax.axhline(0, color="k", lw=.8)
    ax.set_xscale("log"); ax.set_xticks([1, 20, 60, 300, 1200]); ax.set_xticklabels(["1s", "20s", "1min", "5min", "20min"], fontsize=8, rotation=25)
    ax.set_ylim(-0.01, 0.06); ax.set_title(NM[tc], fontsize=11, fontweight="bold")
    ax.set_xlabel("预测未来多久", fontsize=9); ax.legend(fontsize=8); ax.grid(True, alpha=.25)
axes[0].set_ylabel("方向预测 OOS R²", fontsize=10)
fig.suptitle("图117　跨合约预测拉长时间窗口：1s 时跨OFI翻2-3倍，但 20s→20min 全部归零（含跨合约）——共同因子的方向也只活在秒级",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(f"{D}/fig117_跨合约长窗口{SUF}.png", dpi=135)
print("saved fig117")
