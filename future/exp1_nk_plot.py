"""exp1_nk_plot.py — dense n×k figures at fixed x, plus markdown table.
Run: python3 exp1_nk_plot.py   (reads exp1_nk.csv; writes 3 figs)
 fig_pulse_nk_mean.png : n×k 热力图，色=符号化收益均值（红趋势/蓝反转，裁剪±0.5点）
 fig_pulse_nk_t.png    : n×k 热力图，色=t 值（清晰度）
 fig_pulse_nk_lines.png: 每条线一个 n，纵轴 signed vs 阈值 k
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

import sys
D = "/Users/zhuisabella/xn/future"
FIG = f"{D}/figs"
SRC = sys.argv[1] if len(sys.argv) > 1 else f"{D}/exp1_nk_clean.csv"   # default = 净化版
TAG = sys.argv[2] if len(sys.argv) > 2 else ""                   # fig filename suffix
NOTE = "（已过滤坏tick）" if "clean" in SRC else ""
d = pd.read_csv(SRC)
X = int(d["x"].iloc[0])
d["mean"] = d["s_sum"] / d["s_n"]
d["se"] = np.sqrt(np.maximum(d["s_ss"] / d["s_n"] - d["mean"] ** 2, 0) / d["s_n"])
d["t"] = d["mean"] / d["se"]
d["hitpct"] = 100 * d["hits"] / d["s_n"]
NS = sorted(d["n"].unique()); KT = sorted(d["k_ticks"].unique())
M = d.pivot(index="n", columns="k_ticks", values="mean").reindex(index=NS, columns=KT)
T = d.pivot(index="n", columns="k_ticks", values="t").reindex(index=NS, columns=KT)
N = d.pivot(index="n", columns="k_ticks", values="s_n").reindex(index=NS, columns=KT)


def heatmap(Z, vlim, cmap, fname, title, sub, fmt):
    fig, ax = plt.subplots(figsize=(13.5, 6.6))
    im = ax.imshow(Z.values, cmap=cmap, vmin=-vlim, vmax=vlim, aspect="auto")
    ax.set_xticks(range(len(KT))); ax.set_xticklabels(KT)
    ax.set_yticks(range(len(NS))); ax.set_yticklabels(NS)
    ax.set_xlabel("k　脉冲幅度阈值（价格 tick；1 tick = 0.2 点 → 最高 90 tick = 18 点）", fontsize=9.5)
    ax.set_ylabel("n　回看窗口（tick；越小越新鲜）", fontsize=9.5)
    for i in range(len(NS)):
        for j in range(len(KT)):
            v = Z.values[i, j]
            if not np.isfinite(v):
                continue
            faint = N.values[i, j] < 50000
            disp = min(max(v, -vlim), vlim)
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=6,
                    color=("#777777" if faint else
                           ("white" if abs(disp) > 0.58 * vlim else "black")))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, extend="both")
    ax.set_title(title, fontsize=11)
    fig.text(0.5, -0.06, sub, ha="center", va="top", fontsize=8.3, color="#333")
    fig.tight_layout(); fig.savefig(f"{FIG}/{fname[:-4]}{TAG}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


heatmap(M, 0.5, "seismic", "fig_pulse_nk_mean.png",
        f"实验1 价格脉冲：n×k → 趋势(红)/反转(蓝)　均值（指数点，固定向前 x={X} tick≈5秒）\n"
        f"IM 中证1000 · 2022-07..2026-05{NOTE}",
        "计算方法：脉冲 = mid[i]−mid[i−n]，|脉冲|>k 触发；格内 = sign(脉冲)×(mid[i+x]−mid[i]) 的均值。"
        "色阶裁剪到 ±0.5 点放大对比（超出即饱和，真实值见格内/CSV）。\n"
        "单位：n、x 为 tick（500ms/个）；k、数值为指数点（1 点=每手¥200）。灰字 = 样本<5万、不可靠。",
        fmt=lambda v: f"{v:+.2f}" if abs(v) < 10 else f"{v:+.0f}")

heatmap(T, 12.0, "PuOr_r", "fig_pulse_nk_t.png",
        f"实验1 价格脉冲：n×k → t 值（清晰度/置信度，固定向前 x={X} tick）\n"
        f"IM 中证1000 · 2022-07..2026-05{NOTE}",
        "t = 均值 ÷ 标准误（无量纲）。|t|>2 即显著；橙=趋势、紫=反转，越深越确定。"
        "灰字 = 样本<5万、不可靠（高 k / 小 n 角落多为坏 tick）。",
        fmt=lambda v: f"{v:+.0f}")

# threshold-response lines (clip y; mark thin)
fig, ax = plt.subplots(figsize=(11, 6))
ax.axhline(0, color="k", lw=0.9, ls="--")
nlines = [1, 2, 3, 5, 8, 13, 20, 30]
cols = plt.cm.viridis(np.linspace(0.05, 0.9, len(nlines)))
YLIM = (-2.5, 0.15)
for c, n in zip(cols, nlines):
    s = d[d.n == n].sort_values("k_ticks")
    ax.errorbar(s["k_ticks"], s["mean"].clip(*YLIM), yerr=s["se"], color=c, lw=1.8,
                marker="o", ms=4.5, capsize=2, label=f"n={n}")
    for _, r in s.iterrows():
        if r["s_n"] < 50000:
            ax.plot(r["k_ticks"], min(max(r["mean"], YLIM[0]), YLIM[1]),
                    marker="x", color="#d62728", ms=6, mew=1.4)
ax.set_xticks(KT); ax.set_ylim(*YLIM)
ax.set_xlabel("k　脉冲幅度阈值（价格 tick；1 tick = 0.2 点 → 最高 90 tick = 18 点）")
ax.set_ylabel(f"符号化未来收益均值（指数点，向前 x={X} tick）\n＋ = 趋势 · − = 反转")
ax.legend(title="回看窗口 n", fontsize=8, title_fontsize=9, ncol=2)
ax.set_title("阈值响应：放大 k 后趋势如何转为反转（y 轴裁剪到 −2.5；红✗=样本<5万）\n"
             f"IM 中证1000 · 2022-07..2026-05{NOTE}", fontsize=11)
fig.text(0.5, -0.02, "计算方法：脉冲=mid[i]−mid[i−n]，|脉冲|>k 触发；纵轴=sign(脉冲)×(mid[i+x]−mid[i]) 均值。",
         ha="center", va="top", fontsize=8.2, color="#333")
fig.tight_layout(); fig.savefig(f"{FIG}/fig_pulse_nk_lines{TAG}.png", dpi=130, bbox_inches="tight")
plt.close(fig)

print("saved 3 figs: fig_pulse_nk_mean.png / fig_pulse_nk_t.png / fig_pulse_nk_lines.png")

# markdown table (mean) + clearest cells
def md(metric, fmt):
    piv = d.pivot(index="n", columns="k_ticks", values=metric).reindex(index=NS, columns=KT)
    head = "| n \\ k(tick) | " + " | ".join(str(k) for k in KT) + " |"
    sep = "|" + "---|" * (len(KT) + 1)
    out = [head, sep]
    for n in NS:
        out.append(f"| **{n}** | " + " | ".join(fmt(piv.loc[n, k]) for k in KT) + " |")
    return "\n".join(out)

print(f"\n### 均值 signed (指数点), 固定 x={X}\n")
print(md("mean", lambda v: f"{v:+.2f}" if pd.notna(v) else "·"))
d["abst"] = d["t"].abs()
print("\n### 最清晰的反转格 (|t|最大, 样本≥10万)\n")
top = d[(d.s_n >= 100000) & (d["mean"] < 0)].sort_values("abst", ascending=False).head(8)
print("| n | k(tick) | k(点) | mean(点) | t | hit% | n_events |")
print("|---|---|---|---|---|---|---|")
for _, r in top.iterrows():
    print(f"| {int(r.n)} | {int(r.k_ticks)} | {r.k_pts:g} | {r['mean']:+.3f} | {r.t:+.1f} | "
          f"{r.hitpct:.1f} | {int(r.s_n):,} |")
