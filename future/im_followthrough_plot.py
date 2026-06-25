"""im_followthrough_plot.py — figures for the IM follow-through study.
Run with system python3 (has matplotlib):  python3 im_followthrough_plot.py
Reads price_trend.csv, volume_peak.csv (this dir); writes figs/ here.

EXP1 price pulse — for each variable, FIX THE OTHER TWO:
  fig_price_pulse.png : (a) vary n  | (b) vary k | (c) vary x(=h)   ±1 SE
  fig_price_heatmap.png : n × x grid of mean signed fwd return (the regime view)
EXP2 volume peak — change t1 / t2 ONE AT A TIME:
  fig_volume.png : rows = |fwd move| / signed fwd / fwd vol ; cols = vary t2 | vary t1
"""
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

D = "/Users/zhuisabella/xn/future"
FIG = f"{D}/figs"
import os
os.makedirs(FIG, exist_ok=True)
TITLE = "IM 中证1000 · 2022-07..2026-05"
BLUE, RED, GREY = "#4C72B0", "#c0392b", "#888888"


def mt(sum_, ss, n):
    """mean, standard error from sum / sum-of-squares / count."""
    m = sum_ / n
    var = np.maximum(ss / n - m * m, 0.0)
    return m, np.sqrt(var / n)


# ====================== EXP1 : price pulse ======================
p = pd.read_csv(f"{D}/price_trend.csv")
p["mean"], p["se"] = mt(p["s_sum"], p["s_ss"], p["s_n"])
p["hitpct"] = 100.0 * p["hits"] / p["s_n"]

# reference point at which the other two vars are fixed in each panel
REF_N, REF_K, REF_H = 10, 0.2, 10
NS = sorted(p["n"].unique()); KS = sorted(p["k"].unique()); HS = sorted(p["h"].unique())

def panel(ax, xv, xvals, xlab, title, fanv, fanvals, legtitle, fanfmt, fixed):
    """One panel: sweep xv on the x-axis; draw one line per fanv value; pin `fixed`."""
    xpos = list(range(len(xvals)))
    cols = plt.cm.viridis(np.linspace(0.05, 0.82, len(fanvals)))
    ax.axhline(0, color="k", lw=0.9, ls="--")
    for c, fv in zip(cols, fanvals):
        m = (p[fanv] == fv)
        for col, val in fixed.items():
            m = m & (p[col] == val)
        s = p[m].set_index(xv).reindex(xvals).reset_index()
        ax.errorbar(xpos, s["mean"], yerr=s["se"], color=c, lw=1.8,
                    marker="o", ms=5, capsize=2.5, label=fanfmt(fv))
    ax.set_xticks(xpos); ax.set_xticklabels([f"{v:g}" for v in xvals])
    ax.set_xlabel(xlab, fontsize=9.5)
    ax.set_title(title, fontsize=10)
    ax.legend(title=legtitle, fontsize=7.6, title_fontsize=8.4, framealpha=0.9)
    ax.margins(x=0.08)


fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))
# A) 横轴 n，固定 k=0.2，多线 = 不同向前窗口 x  →  揭示 n×x 交互
panel(axes[0], "n", NS, "n　回看窗口（tick；1 tick≈500毫秒）",
      "A) 横轴 = n（回看窗口）；固定 k=0.2 点\n多条线 = 不同的向前窗口 x",
      "h", [5, 10, 40, 120], "向前窗口 x",
      lambda v: f"x={int(v)} tick（≈{int(v)*0.5:g}秒）", {"k": 0.2})
# B) 横轴 k，固定 x=10，多线 = 不同回看窗口 n  →  揭示 k×n 交互
panel(axes[1], "k", KS, "k　脉冲幅度阈值（指数点；IM最小变动=0.2点）",
      "B) 横轴 = k（脉冲幅度阈值）；固定 x=10 tick\n多条线 = 不同的回看窗口 n",
      "n", [5, 10, 20, 40], "回看窗口 n",
      lambda v: f"n={int(v)} tick", {"h": 10})
# C) 横轴 x，固定 n=10，多线 = 不同脉冲阈值 k  →  揭示 x×k 交互
panel(axes[2], "h", HS, "x　向前窗口（tick；1 tick≈500毫秒）",
      "C) 横轴 = x（向前窗口）；固定 n=10 tick\n多条线 = 不同的脉冲阈值 k",
      "k", [0.0, 0.4, 0.8, 1.6], "脉冲阈值 k",
      lambda v: f"k={v:g} 点", {"n": 10})

axes[0].set_ylabel("符号化未来收益　均值（指数点）\n  ＋ = 趋势 / 延续      − = 反转", fontsize=10)
fig.suptitle("实验1　价格脉冲：一个快速涨跌之后，是延续（趋势）还是反转？　"
             "IM 中证1000 · 2022-07..2026-05", fontsize=13, y=1.03)
fig.text(0.5, -0.06,
         "单位：1 tick = 一次 500 毫秒行情快照（约 120 个/分钟）；n、x 均为 tick 个数（时间）。"
         "k 与纵轴均为中证1000指数点（IM 最小变动 0.2 点；1 点 = 每手 ¥200）。\n"
         "符号化收益 = sign(脉冲方向) ×（mid[i+x] − mid[i]）；＞0 为趋势/延续，＜0 为反转。"
         "圆点 = 约 1300 万–2200 万次触发的均值；误差棒 = ±1 标准误（很窄，因样本极大）；|t|＞2 即显著。",
         ha="center", va="top", fontsize=8.4, color="#333333")
fig.tight_layout(); fig.savefig(f"{FIG}/fig_price_pulse.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# regime heatmap: mean signed fwd over n (rows) x x (cols), at k=REF_K
piv = p[p.k == REF_K].pivot(index="n", columns="h", values="mean").reindex(index=NS, columns=HS)
fig, ax = plt.subplots(figsize=(7.6, 4.8))
vmax = np.nanmax(np.abs(piv.values))
im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(len(HS))); ax.set_xticklabels(HS)
ax.set_yticks(range(len(NS))); ax.set_yticklabels(NS)
ax.set_xlabel("x  (forward horizon, ticks)"); ax.set_ylabel("n  (lookback ticks)")
for i in range(len(NS)):
    for j in range(len(HS)):
        v = piv.values[i, j]
        ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=8,
                color="white" if abs(v) > 0.6 * vmax else "black")
fig.colorbar(im, ax=ax, label="mean signed fwd return (pts)")
ax.set_title(f"EXP1 regime: trend (red) vs reversal (blue)   k={REF_K}\n{TITLE}", fontsize=10.5)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_price_heatmap.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# ====================== EXP2 : volume peak ======================
v = pd.read_csv(f"{D}/volume_peak.csv")
v["absmove"], v["abse"] = mt(v["a_sum"], v["a_ss"], v["n_peak"])
v["signed"], v["sige"] = mt(v["g_sum"], v["g_ss"], v["n_peak"])
v["fvol"], v["fve"] = mt(v["v_sum"], v["v_ss"], v["n_peak"])

RS = ["1.5", "2.0", "3.0"]          # peak strengths (lines); ALL = baseline (dashed)
RCOL = {"1.5": "#9bbcd6", "2.0": BLUE, "3.0": "#1f3a5f"}
metrics = [("absmove", "①未来 |价格变动|（指数点）\n— 波动率", "abse", False),
           ("signed", "②符号化未来收益（指数点）\n＋延续 · −反转", "sige", True),
           ("fvol", "③未来每 tick 成交量\n（手）", "fve", False)]
# three column sweeps — change t2 / t1 / x ONE AT A TIME, each pinning DIFFERENT vars
columns = [("t2", "改变 t2 长窗（固定 t1=5, x=20）", (v.t1 == 5) & (v.h == 20),
            "t2　长成交量窗口（tick）"),
           ("t1", "改变 t1 短窗（固定 t2=120, x=20）", (v.t2 == 120) & (v.h == 20),
            "t1　短成交量窗口（tick）"),
           ("h", "改变 x 向前窗口（固定 t1=10, t2=120）", (v.t1 == 10) & (v.t2 == 120),
            "x　向前窗口（tick；1 tick≈500毫秒）")]

fig, axes = plt.subplots(3, 3, figsize=(15.5, 12), sharey="row")
for ci, (xcol, ctitle, cmask, xlab) in enumerate(columns):
    sub = v[cmask]
    xvals = sorted(sub[xcol].unique())
    xpos = list(range(len(xvals)))
    for ri, (mcol, ylab, ecol, zero) in enumerate(metrics):
        ax = axes[ri, ci]
        if zero:
            ax.axhline(0, color="k", lw=0.8)
        base = sub[sub.r == "ALL"].set_index(xcol).reindex(xvals)
        ax.plot(xpos, base[mcol], color=GREY, lw=1.6, ls="--",
                marker="s", ms=5, label="ALL 全样本基线")
        for r in RS:
            s = sub[sub.r == r].set_index(xcol).reindex(xvals)
            ax.errorbar(xpos, s[mcol], yerr=s[ecol], color=RCOL[r], lw=1.9,
                        marker="o", ms=5, capsize=2.5, label=f"峰值 spike＞{r}")
        ax.set_xticks(xpos); ax.set_xticklabels([f"{int(x)}" for x in xvals])
        ax.margins(x=0.1)
        if ci == 0:
            ax.set_ylabel(ylab, fontsize=9.5)
        if ri == 0:
            ax.set_title(ctitle, fontsize=10.5)
        if ri == 2:
            ax.set_xlabel(xlab, fontsize=9.5)
axes[0, 0].legend(title="成交量峰值强度", fontsize=8, title_fontsize=8.6, loc="best")
fig.suptitle("实验2　成交量峰值：一次放量之后，对随后的 ticks 有什么影响？　"
             "IM 中证1000 · 2022-07..2026-05", fontsize=13, y=1.0)
fig.text(0.5, -0.03,
         "峰值定义：spike =（过去 t1 笔的量 / t1）÷（过去 t2 笔的量 / t2），t2＞t1；spike＞r 即一次"
         "“成交量峰值”。ALL = 全样本基线（每个 tick）。\n"
         "单位：t1、t2、x 均为 tick 个数（1 tick = 一次 500 毫秒快照）；价格/波动率为中证1000指数点"
         "（1 点 = 每手 ¥200）；成交量为手。②方向 = sign(短窗价格方向) ×（mid[i+x] − mid[i]），"
         "＞0 延续、＜0 反转。圆点 = 该格全历史均值，误差棒 = ±1 标准误。",
         ha="center", va="top", fontsize=8.3, color="#333333")
fig.tight_layout(); fig.savefig(f"{FIG}/fig_volume.png", dpi=130, bbox_inches="tight")
plt.close(fig)

print("saved:")
for f in ("fig_price_pulse.png", "fig_price_heatmap.png", "fig_volume.png"):
    print(f"  {FIG}/{f}")
