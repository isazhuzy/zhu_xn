"""horizon_har_plot.py — fig110: extending the time axis. Two truths side by side.
Left:  direction OOS R² vs forecast horizon (10s→15min), for several flow-lookbacks —
       collapses to ~0 past ~10s (market efficiency), even with accumulated flow.
Right: volatility OOS R² (HAR vs AR1) at 1min/5min — strongly predictable (~0.5).
Run: python3 horizon_har_plot.py   (SUF=_pilot for pilot files)
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
HSEC = {"10s": 10, "1min": 60, "5min": 300, "15min": 900}
hz = pd.read_csv(f"{D}/horizon_grid{SUF}.csv")
har = pd.read_csv(f"{D}/har_results{SUF}.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.6))

# left: direction R² vs horizon, lookback=1min flow, all contracts; + reference short-horizon VOI
for code in NM:
    s = hz[(hz.code == code) & (hz.look == "1min")].copy()
    s["hs"] = s.hor.map(HSEC); s = s.sort_values("hs")
    ax1.plot(s.hs, s.r2_oos, marker="o", ms=5, lw=1.8, color=COL[code], label=NM[code])
ax1.axhline(0, color="k", lw=1)
ax1.set_xscale("log"); ax1.set_xticks([10, 60, 300, 900]); ax1.set_xticklabels(["10s", "1min", "5min", "15min"])
ax1.set_ylim(-0.02, 0.06)
ax1.set_xlabel("预测未来多久（累积1min流量为输入）", fontsize=10)
ax1.set_ylabel("方向预测 OOS R²", fontsize=10)
ax1.set_title("方向预测：过了10秒就≈0（含短端VOI参照）\n累积更久的流量也救不回来 → 市场有效", fontsize=11, fontweight="bold")
ax1.annotate("VOI 短端参照:\n0.5s≈8.7% → 10s≈1.7%", xy=(10, 0.017), xytext=(14, 0.042),
             fontsize=8.5, color="0.35", arrowprops=dict(arrowstyle="->", color="0.5"))
ax1.legend(fontsize=8.5); ax1.grid(True, alpha=.25)

# right: volatility R² HAR vs AR1
x = np.arange(len(NM)); wd = 0.2
for j, (H, hatch) in enumerate([("1min", None), ("5min", "//")]):
    ar = [har[(har.code == c) & (har.horizon == H) & (har.model == "AR1")].r2_oos.iloc[0] for c in NM]
    hr = [har[(har.code == c) & (har.horizon == H) & (har.model == "HAR")].r2_oos.iloc[0] for c in NM]
    ax2.bar(x + (2 * j - 1.5) * wd, ar, wd, color="0.6", hatch=hatch, edgecolor="w",
            label=f"AR1（仅近1min） {H}")
    ax2.bar(x + (2 * j - 0.5) * wd, hr, wd, color=[COL[c] for c in NM], hatch=hatch, edgecolor="w",
            label=f"HAR（1+5+30min） {H}")
ax2.set_xticks(x); ax2.set_xticklabels([NM[c].split()[0] for c in NM], fontsize=9.5)
ax2.set_ylabel("波动率(log RV) 预测 OOS R²", fontsize=10)
ax2.set_title("波动率预测：样本外 R²≈0.5，HAR 级联远胜 AR1\n方向死了，但『会不会波动』强可预测", fontsize=11, fontweight="bold")
ax2.legend(fontsize=8, ncol=2); ax2.grid(True, axis="y", alpha=.3)

fig.suptitle("图110　延伸时间轴：方向预测在分钟级归零，但波动率强可预测（HAR-RV）——换目标，不换尺度",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{D}/fig110_延伸时间轴{SUF}.png", dpi=135)
print("saved fig110")
