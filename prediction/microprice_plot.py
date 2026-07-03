"""microprice_plot.py — figures for the micro-price study (Stoikov 2018).
fig91: micro-price adjustment g*(I, spread).  fig92: conditional drift of mid/wmid/micro
(the paper's signature plot).  fig93: out-of-sample RMSE vs mid, relative.
Run: python3 microprice_plot.py  (SUF=_pilot for pilot files)
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
gst = pd.read_csv(f"{D}/mp_gstar{SUF}.csv"); rmse = pd.read_csv(f"{D}/mp_rmse{SUF}.csv")
bias = pd.read_csv(f"{D}/mp_bias{SUF}.csv")

# fig91 — g*(imbalance, spread)
SPRC = {1: "#2c7fb8", 2: "#7fcdbb", 3: "#bdbdbd"}
fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
for ax, code in zip(axes.ravel(), NAME):
    for s in (1, 2, 3):
        g = gst[(gst.code == code) & (gst.spr == s) & (gst.n_state >= 1000)].sort_values("icenter")
        lab = {1: "价差=1 tick", 2: "价差=2 ticks", 3: "价差≥3 ticks"}[s]
        ax.plot(g.icenter, g.gstar, marker="o", ms=4, lw=1.6, color=SPRC[s], label=lab)
    ax.axhline(0, color="0.6", lw=.6); ax.axvline(.5, color="0.6", lw=.6)
    ax.set_title(NAME[code], fontsize=10.5, fontweight="bold")
    ax.set_xlabel("失衡 I = qb/(qb+qa)", fontsize=9)
    ax.set_ylabel("微观价格调整 g* (tick)", fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=.25)
fig.suptitle("图91　微观价格 = 中间价 + g*(失衡,价差)（Stoikov 2018；训练期2020-24估计）",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{D}/fig91_微观价格调整{SUF}.png", dpi=135); plt.close(fig)

# fig92 — signature: conditional future drift of each estimator
PC = {"mid": "0.4", "wmid": "#c0392b", "micro": "#2c7fb8"}
PL = {"mid": "中间价 mid", "wmid": "加权中间价 wmid", "micro": "微观价格 micro"}
fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
for ax, code in zip(axes.ravel(), NAME):
    for p in ("mid", "wmid", "micro"):
        b = bias[(bias.code == code) & (bias.predictor == p)].sort_values("icenter")
        ax.plot(b.icenter, b.drift_ticks, marker="o", ms=4, lw=1.6, color=PC[p], label=PL[p])
    ax.axhline(0, color="0.6", lw=.8)
    ax.set_title(NAME[code], fontsize=10.5, fontweight="bold")
    ax.set_xlabel("失衡 I = qb/(qb+qa)", fontsize=9)
    ax.set_ylabel("E[mid(t+10s) − 估计价 | I]  (tick)", fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=.25)
fig.suptitle("图92　好的“公允价”应无条件漂移（水平线=无偏）—— 样本外2025-26",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{D}/fig92_条件漂移{SUF}.png", dpi=135); plt.close(fig)

# fig93 — RMSE relative to mid
fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
for ax, code in zip(axes.ravel(), NAME):
    r = rmse[rmse.code == code]
    base = r[r.predictor == "mid"].set_index("h").rmse_ticks
    for p, c in (("micro", "#2c7fb8"), ("wmid", "#c0392b")):
        s = r[r.predictor == p].sort_values("secs")
        rel = (s.set_index("h").rmse_ticks / base - 1) * 100
        ax.plot(s.secs.to_numpy(), rel.to_numpy(), marker="o", ms=5, lw=1.8, color=c, label=PL[p])
    ax.axhline(0, color="k", lw=1, ls="--", label="基准：中间价")
    ax.set_xscale("log"); ax.set_xticks([0.5, 2, 10, 60])
    ax.set_xticklabels(["0.5s", "2s", "10s", "60s"])
    ax.set_title(NAME[code], fontsize=10.5, fontweight="bold")
    ax.set_xlabel("预测期", fontsize=9); ax.set_ylabel("RMSE 相对 mid 的变化 (%)", fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=.25)
fig.suptitle("图93　预测未来中间价的RMSE（负=比mid好）—— 微观价格短期胜出，样本外2025-26",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{D}/fig93_RMSE对比{SUF}.png", dpi=135); plt.close(fig)
print("saved fig91, fig92, fig93")
