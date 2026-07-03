"""qi_plot.py — figures for the queue-imbalance study (Gould & Bonart 2016).
fig71: empirical P(up|I) + logistic fit per contract.  fig72: split by spread state.
fig73: monthly hit-rate stability.   Run: python3 qi_plot.py  (SUF=_pilot for pilot)
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
NAME = {"IC0000": "IC 中证500", "IF0000": "IF 沪深300", "IH0000": "IH 上证50", "IM0000": "IM 中证1000"}
COL = {"IC0000": "#c0392b", "IF0000": "#e08a3c", "IH0000": "#27ae60", "IM0000": "#4C72B0"}
bins = pd.read_csv(f"{D}/qi_bins{SUF}.csv"); res = pd.read_csv(f"{D}/qi_results{SUF}.csv")
mon = pd.read_csv(f"{D}/qi_permonth{SUF}.csv")

# fig71 — empirical P(up | I) + logistic fit, all spreads pooled
fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
for ax, code in zip(axes.ravel(), NAME):
    b = bins[(bins.code == code) & (bins.sprstate == "all")].sort_values("x")
    r = res[(res.code == code) & (res.sprstate == "all")].iloc[0]
    p = b.n_up / b.n
    ax.scatter(b.x, p, s=np.clip(b.n / b.n.max() * 120, 8, 120), color=COL[code], alpha=.75)
    xs = np.linspace(-1, 1, 200)
    ax.plot(xs, 1 / (1 + np.exp(-(r.a + r.b * xs))), "k--", lw=1.3,
            label=f"logistic a={r.a:.2f} b={r.b:.2f}")
    ax.axhline(.5, color="0.6", lw=.6); ax.axvline(0, color="0.6", lw=.6)
    ax.set_ylim(0.15, 0.85)
    ax.set_title(f"{NAME[code]}   命中率={r.hitrate:.1%}  伪R²={r.pseudoR2:.3f}",
                 fontsize=10.5, fontweight="bold")
    ax.set_xlabel("队列失衡 I=(qb−qa)/(qb+qa)", fontsize=9)
    ax.set_ylabel("P(下一次中间价变动向上)", fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=.25)
fig.suptitle("图71　盘口队列失衡 → 下一次中间价变动方向（Gould-Bonart 2016；点=经验分箱概率）",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{D}/fig71_QI概率曲线{SUF}.png", dpi=135); plt.close(fig)

# fig72 — by spread state (the large-tick effect)
fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
SPRC = {"1": "#2c7fb8", "2": "#7fcdbb", "3": "#bdbdbd"}
for ax, code in zip(axes.ravel(), NAME):
    for s in ["1", "2", "3"]:
        b = bins[(bins.code == code) & (bins.sprstate == s)].sort_values("x")
        b = b[b.n >= 200]
        r = res[(res.code == code) & (res.sprstate == s)].iloc[0]
        lab = {"1": "价差=1 tick", "2": "价差=2 ticks", "3": "价差≥3 ticks"}[s]
        ax.plot(b.x, b.n_up / b.n, marker="o", ms=3, lw=1.4, color=SPRC[s],
                label=f"{lab}  命中{r.hitrate:.0%}")
    ax.axhline(.5, color="0.6", lw=.6); ax.axvline(0, color="0.6", lw=.6)
    ax.set_ylim(0.1, 0.9); ax.set_title(NAME[code], fontsize=10.5, fontweight="bold")
    ax.set_xlabel("队列失衡 I", fontsize=9); ax.set_ylabel("P(向上)", fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=.25)
fig.suptitle("图72　价差越窄（大tick状态）信号越强 —— 分价差状态的 P(向上|I)",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{D}/fig72_QI分价差{SUF}.png", dpi=135); plt.close(fig)

# fig73 — monthly stability of the sign-rule hit rate
mon = mon.dropna(subset=["hitrate"]).copy()
mon["t"] = mon.year + (mon.month - 1) / 12.0
fig, ax = plt.subplots(figsize=(12, 5.5))
for code in NAME:
    s = mon[mon.code == code].sort_values("t")
    ax.plot(s.t, s.hitrate, color=COL[code], lw=1.5, marker="o", ms=3, label=NAME[code])
ax.axhline(.5, color="k", lw=1, ls="--", label="抛硬币=50%")
ax.set_xlabel("年份", fontsize=11); ax.set_ylabel("sign(I) 命中率", fontsize=11)
ax.set_title("图73　队列失衡方向命中率的逐月稳定性 —— 每个月都显著高于50%",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9, ncol=5, loc="lower center"); ax.grid(True, alpha=.3)
fig.tight_layout(); fig.savefig(f"{D}/fig73_QI逐月稳定性{SUF}.png", dpi=135); plt.close(fig)
print("saved fig71, fig72, fig73")
