"""voi_plot.py — figures for the VOI/OIR/MPB study (Shen 2015).
fig81: R² vs horizon (model A/B, train vs out-of-sample).  fig82: monthly stability.
fig83: OOS sign hit rate vs signal threshold.  Run: python3 voi_plot.py (SUF=_pilot)
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
res = pd.read_csv(f"{D}/voi_results{SUF}.csv"); mon = pd.read_csv(f"{D}/voi_permonth{SUF}.csv")
hit = pd.read_csv(f"{D}/voi_hitrate{SUF}.csv")

# fig81 — predictive R² vs horizon
fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
for ax, code in zip(axes.ravel(), NAME):
    A = res[(res.code == code) & (res.model == "A")].sort_values("secs")
    B = res[(res.code == code) & (res.model == "B")].sort_values("secs")
    ax.plot(A.secs, A.r2_train, color="0.55", lw=1.4, marker="s", ms=4,
            label="模型A（仅VOI）训练内")
    ax.plot(B.secs, B.r2_train, color=COL[code], lw=2, marker="o",
            label="模型B（VOI+OIR+MPB）训练内")
    ax.plot(B.secs, B.r2_test_oos, color=COL[code], lw=2, ls="--", marker="^",
            label="模型B 样本外(2025-26)")
    ax.set_xscale("log"); ax.set_xticks([0.5, 2, 10, 60])
    ax.set_xticklabels(["0.5s", "2s", "10s", "60s"])
    ax.set_title(NAME[code], fontsize=10.5, fontweight="bold")
    ax.set_xlabel("预测期 k（未来k个快照的平均中间价变动）", fontsize=9)
    ax.set_ylabel("R²", fontsize=9); ax.legend(fontsize=8); ax.grid(True, alpha=.25)
fig.suptitle("图81　盘口因子的真·预测R²（Shen 2015）—— 越短的预测期越可预测\n"
             "VOI=队伍变化（流量）· OIR=队伍水平＝(qb−qa)/(qb+qa) · MPB=平均成交价−中间价（成交方向）",
             fontsize=11, fontweight="bold")
fig.text(0.5, 0.01,
         "R²（决定系数）= 模型解释了未来价格波动的百分之几：0=什么都没解释，1=完美预测；"
         "高频收益预测里 R² 几个百分点已属强信号，真正的关键是样本外仍成立且统计显著。",
         ha="center", fontsize=8.5, color="0.35")
fig.tight_layout(rect=(0, 0.03, 1, 0.95)); fig.savefig(f"{D}/fig81_VOI预测R2{SUF}.png", dpi=135); plt.close(fig)

# fig82 — monthly stability (model B, k=20)
mon = mon.dropna(subset=["r2_refit"]).copy()
mon["t"] = mon.year + (mon.month - 1) / 12.0
fig, ax = plt.subplots(figsize=(12, 5.5))
for code in NAME:
    s = mon[mon.code == code].sort_values("t")
    ax.plot(s.t, s.r2_refit, color=COL[code], lw=1.5, marker="o", ms=3, label=NAME[code])
    so = s.dropna(subset=["r2_oos"])
    if len(so):
        ax.plot(so.t, so.r2_oos, color=COL[code], lw=1.2, ls="--", alpha=.7)
if (mon.phase == "test").any():
    tb = mon[mon.phase == "test"].t.min()
    ax.axvline(tb, color="k", lw=1, ls=":"); ax.text(tb, ax.get_ylim()[1] * .95, " 训练|测试", fontsize=9)
ax.set_xlabel("年份", fontsize=11); ax.set_ylabel("R² (模型B, k=20≈10s)", fontsize=11)
ax.set_title("图82　VOI模型逐月R²稳定性（实线=当月重拟合；虚线=用训练期系数的样本外）",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9, ncol=5, loc="upper center"); ax.grid(True, alpha=.3)
fig.tight_layout(); fig.savefig(f"{D}/fig82_VOI逐月稳定性{SUF}.png", dpi=135); plt.close(fig)

# fig83 — OOS hit rate vs threshold
fig, ax = plt.subplots(figsize=(10, 6))
w, xs = 0.2, np.arange(len(hit.thr.unique()))
thrs = sorted(hit.thr.unique())
for i, code in enumerate(NAME):
    s = hit[hit.code == code].sort_values("thr")
    bars = ax.bar(xs + (i - 1.5) * w, s.hitrate, w, color=COL[code], label=NAME[code])
    for x, (hr, cov) in zip(xs + (i - 1.5) * w, zip(s.hitrate, s.coverage)):
        ax.text(x, hr + .004, f"{cov:.0%}", ha="center", fontsize=7, color="0.35")
ax.axhline(.5, color="k", lw=1, ls="--")
ax.set_xticks(xs); ax.set_xticklabels([f"|ŷ|>{t}tick" for t in thrs])
ax.set_ylim(.45, .75); ax.set_ylabel("方向命中率（样本外, k=20）", fontsize=11)
ax.set_title("图83　信号越强命中率越高（柱上数字=触发比例）—— 模型B, 2025-26样本外",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(f"{D}/fig83_VOI命中率{SUF}.png", dpi=135); plt.close(fig)
print("saved fig81, fig82, fig83")
