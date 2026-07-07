"""qi_deep_plot.py — figures for the deeper QI study (qi_deep_ddb.py).
fig75: sign(I) hit rate & pseudo-R2 vs horizon (how far the edge reaches).
fig76: horizon decay split by spread state.
fig77: adverse selection — mean markout by I, bid-fill vs ask-fill (10s).
fig78: net maker edge (half-spread + directional markout) by I -> where quoting pays.
Run: python3 qi_deep_plot.py   (SUF=_pilot for pilot)
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
hor = pd.read_csv(f"{D}/qi_horizon{SUF}.csv")
mk = pd.read_csv(f"{D}/qi_markout{SUF}.csv")

# fig75 — hit rate & pseudo-R2 vs horizon (all spreads)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for code in NAME:
    s = hor[(hor.code == code) & (hor.sprstate == "all")].sort_values("secs")
    ax1.plot(s.secs, s.hitrate, marker="o", ms=4, lw=1.6, color=COL[code], label=NAME[code])
    ax2.plot(s.secs, s.pseudoR2, marker="o", ms=4, lw=1.6, color=COL[code], label=NAME[code])
ax1.axhline(.5, color="k", lw=1, ls="--", label="抛硬币=50%")
ax1.set_xscale("log"); ax2.set_xscale("log")
ax1.set_xlabel("预测跨度（秒，log）", fontsize=10); ax1.set_ylabel("sign(I) 命中率", fontsize=10)
ax2.set_xlabel("预测跨度（秒，log）", fontsize=10); ax2.set_ylabel("logistic 伪R²", fontsize=10)
ax1.set_title("方向命中率随跨度衰减", fontsize=11, fontweight="bold")
ax2.set_title("解释力（伪R²）随跨度衰减", fontsize=11, fontweight="bold")
for a in (ax1, ax2):
    a.set_xticks([0.5, 1, 2.5, 5, 10, 30, 60]); a.set_xticklabels(["0.5", "1", "2.5", "5", "10", "30", "60"])
    a.grid(True, alpha=.3); a.legend(fontsize=8)
fig.suptitle("图75　队列失衡 QI 的预测跨度衰减 —— 优势集中在 0.5~1 秒，30 秒后归零",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{D}/fig75_QI跨度衰减{SUF}.png", dpi=135); plt.close(fig)

# fig76 — decay by spread state, per contract
SPRC = {"1": "#2c7fb8", "2": "#7fcdbb", "3": "#bdbdbd"}
LAB = {"1": "价差=1 tick", "2": "价差=2 ticks", "3": "价差≥3 ticks"}
fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
for ax, code in zip(axes.ravel(), NAME):
    for s in ["1", "2", "3"]:
        d = hor[(hor.code == code) & (hor.sprstate == s)].sort_values("secs")
        ax.plot(d.secs, d.hitrate, marker="o", ms=3.5, lw=1.5, color=SPRC[s], label=LAB[s])
    ax.axhline(.5, color="k", lw=.9, ls="--")
    ax.set_xscale("log"); ax.set_xticks([0.5, 1, 2.5, 5, 10, 30, 60])
    ax.set_xticklabels(["0.5", "1", "2.5", "5", "10", "30", "60"])
    ax.set_title(NAME[code], fontsize=10.5, fontweight="bold")
    ax.set_xlabel("预测跨度（秒，log）", fontsize=9); ax.set_ylabel("sign(I) 命中率", fontsize=9)
    ax.grid(True, alpha=.3); ax.legend(fontsize=8)
fig.suptitle("图76　紧盘口（1 tick）的方向优势更高、也衰减更快 —— 分价差状态的跨度衰减",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{D}/fig76_QI跨度分价差{SUF}.png", dpi=135); plt.close(fig)

HMK_MAIN = 20  # 10s markout for figs 77-78

# fig77 — adverse selection: mean markout by I, bid vs ask fill
fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
for ax, code in zip(axes.ravel(), NAME):
    d = mk[(mk.code == code) & (mk.h == HMK_MAIN)]
    for side, col, lab in [("bid_fill", "#2c7fb8", "被动买入（挂买单被卖单击中）"),
                           ("ask_fill", "#c0392b", "被动卖出（挂卖单被买单击中）")]:
        s = d[(d.side == side) & (d.n >= 500)].sort_values("Ibin")
        ax.plot(s.Ibin, s.mean_markout, marker="o", ms=3, lw=1.5, color=col, label=lab)
    ax.axhline(0, color="0.5", lw=.7); ax.axvline(0, color="0.5", lw=.7)
    ax.set_title(NAME[code], fontsize=10.5, fontweight="bold")
    ax.set_xlabel("成交时的队列失衡 I", fontsize=9)
    ax.set_ylabel("成交后中间价漂移 Δmid（tick，10s）", fontsize=9)
    ax.grid(True, alpha=.3); ax.legend(fontsize=7.5)
fig.suptitle("图77　逆向选择 —— 挂单被击中后中间价往哪走（QI 提前预警毒性成交）",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{D}/fig77_QI逆向选择{SUF}.png", dpi=135); plt.close(fig)

# fig78 — net maker edge by I (half-spread + directional markout)
fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
for ax, code in zip(axes.ravel(), NAME):
    d = mk[(mk.code == code) & (mk.h == HMK_MAIN)]
    for side, col, lab in [("bid_fill", "#2c7fb8", "被动买入净收益"),
                           ("ask_fill", "#c0392b", "被动卖出净收益")]:
        s = d[(d.side == side) & (d.n >= 500)].sort_values("Ibin")
        ax.plot(s.Ibin, s.mean_netedge, marker="o", ms=3, lw=1.5, color=col, label=lab)
    ax.axhline(0, color="k", lw=.9, ls="--"); ax.axvline(0, color="0.5", lw=.7)
    ax.set_title(NAME[code], fontsize=10.5, fontweight="bold")
    ax.set_xlabel("成交时的队列失衡 I", fontsize=9)
    ax.set_ylabel("每手净做市收益（tick，含半价差）", fontsize=9)
    ax.grid(True, alpha=.3); ax.legend(fontsize=8)
fig.suptitle("图78　按 QI 决定报价方向 —— 净做市收益＝半价差＋方向漂移（>0 才该挂）",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{D}/fig78_QI做市净收益{SUF}.png", dpi=135); plt.close(fig)
print("saved fig75, fig76, fig77, fig78")
