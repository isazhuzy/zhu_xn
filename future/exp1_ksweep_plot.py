"""exp1_ksweep_plot.py — figures + table for the extended EXP1 k-sweep.
Run with system python3 (matplotlib):  python3 exp1_ksweep_plot.py
Reads exp1_ksweep.csv (this dir); writes figs/ + prints a markdown table.

Two heatmaps over n (rows) x k_ticks (cols) at a chosen horizon h:
  LEFT  = mean SIGNED return (regime: red=trend, blue=reversal)
  RIGHT = t-stat (CLARITY: |t|>2 significant; faded where sample is thin)
Plus a per-n line plot of signed return vs k (the threshold response).
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
H = 10                                    # horizon for the heatmaps (peak-edge ~5s)

d = pd.read_csv(f"{D}/exp1_ksweep.csv")
d["mean"] = d["s_sum"] / d["s_n"]
d["se"] = np.sqrt(np.maximum(d["s_ss"] / d["s_n"] - d["mean"] ** 2, 0) / d["s_n"])
d["t"] = d["mean"] / d["se"]
d["hitpct"] = 100 * d["hits"] / d["s_n"]
NS = sorted(d["n"].unique()); KT = sorted(d["k_ticks"].unique())

dh = d[d.h == H]
M = dh.pivot(index="n", columns="k_ticks", values="mean").reindex(index=NS, columns=KT)
T = dh.pivot(index="n", columns="k_ticks", values="t").reindex(index=NS, columns=KT)
N = dh.pivot(index="n", columns="k_ticks", values="s_n").reindex(index=NS, columns=KT)

# ---------------- two SEPARATE heatmaps: regime (mean) and clarity (t) ----------------
def heatmap(Z, vlim, cmap, fname, title, sub, fmt):
    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    im = ax.imshow(Z.values, cmap=cmap, vmin=-vlim, vmax=vlim, aspect="auto")
    ax.set_xticks(range(len(KT))); ax.set_xticklabels(KT)
    ax.set_yticks(range(len(NS))); ax.set_yticklabels(NS)
    ax.set_xlabel("k　脉冲幅度阈值（价格 tick；1 tick = 0.2 点）", fontsize=9.5)
    ax.set_ylabel("n　回看窗口（tick；越小越新鲜）", fontsize=9.5)
    for i in range(len(NS)):
        for j in range(len(KT)):
            v = Z.values[i, j]
            faint = N.values[i, j] < 50000              # thin sample → grey text
            disp = min(max(v, -vlim), vlim)             # clip for text-colour test
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=8,
                    color=("#777777" if faint else
                           ("white" if abs(disp) > 0.58 * vlim else "black")))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, extend="both")
    ax.set_title(title, fontsize=11)
    fig.text(0.5, -0.04, sub, ha="center", va="top", fontsize=8.4, color="#333")
    fig.tight_layout(); fig.savefig(f"{FIG}/{fname}", dpi=130, bbox_inches="tight")
    plt.close(fig)

# (1) MEAN — its own figure; colour scale CLIPPED to ±0.5 pt for vivid, distinct colour
heatmap(M, 0.5, "seismic", "fig_exp1_ksweep_mean.png",
        f"实验1 扩展：价格脉冲 → 趋势(红) / 反转(蓝)　均值（指数点，向前 x={H} tick）\n"
        "IM 中证1000 · 2022-07..2026-05",
        "颜色范围裁剪到 ±0.5 点以放大对比（超出即饱和，箭头表示；真实数值见格内）。"
        "格内 = 均值（指数点，1 点 = 每手 ¥200）；灰字 = 样本<5万、不可靠。"
        "脉冲 = mid[i]−mid[i−n]，|脉冲|>k 才触发。",
        fmt=lambda v: f"{v:+.3f}")

# (2) t-STAT — its own figure; clarity / confidence
heatmap(T, 12.0, "PuOr_r", "fig_exp1_ksweep_t.png",
        f"实验1 扩展：t 值 = 清晰度 / 置信度（向前 x={H} tick）\n"
        "IM 中证1000 · 2022-07..2026-05",
        "t = 均值 ÷ 标准误（无量纲）。|t|>2 即显著；颜色越深越确定。"
        "正(橙)=趋势、负(紫)=反转。灰字 = 样本<5万、不可靠。",
        fmt=lambda v: f"{v:+.1f}")

# ---------------- threshold-response lines: signed vs k, per small n ----------------
fig, ax = plt.subplots(figsize=(9.5, 5.6))
ax.axhline(0, color="k", lw=0.9, ls="--")
nlines = [1, 2, 3, 5, 10, 20]
cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(nlines)))
for c, n in zip(cols, nlines):
    s = dh[dh.n == n].sort_values("k_ticks")
    ax.errorbar(s["k_ticks"], s["mean"], yerr=s["se"], color=c, lw=1.8,
                marker="o", ms=5, capsize=2.5, label=f"n={n}")
ax.set_xticks(KT)
ax.set_xlabel("k　脉冲幅度阈值（价格 tick；1 tick = 0.2 点 → 最高 20 tick = 4.0 点）")
ax.set_ylabel("符号化未来收益　均值（指数点）\n＋ = 趋势 · − = 反转")
ax.legend(title="回看窗口 n", fontsize=8.5, title_fontsize=9)
ax.set_title(f"阈值响应：放大 k 后，趋势如何转为反转（向前 x={H} tick）\n"
             "IM 中证1000 · 2022-07..2026-05", fontsize=11)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_exp1_ksweep_lines.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# ---------------------------- markdown results table ----------------------------
def md_table(metric, fmt):
    piv = dh.pivot(index="n", columns="k_ticks", values=metric).reindex(index=NS, columns=KT)
    head = "| n \\\\ k(tick) | " + " | ".join(str(k) for k in KT) + " |"
    sep = "|" + "---|" * (len(KT) + 1)
    lines = [head, sep]
    for n in NS:
        cells = " | ".join(fmt(piv.loc[n, k]) for k in KT)
        lines.append(f"| **{n}** | {cells} |")
    return "\n".join(lines)

print(f"### EXP1 extended — horizon x={H} ticks  (mean signed return, index pts)\n")
print(md_table("mean", lambda v: f"{v:+.3f}" if pd.notna(v) else "·"))
print(f"\n### t-stat (clarity)  x={H}\n")
print(md_table("t", lambda v: f"{v:+.1f}" if pd.notna(v) else "·"))

# clearest cells across ALL horizons
d["abst"] = d["t"].abs()
top = d[d.s_n >= 50000].sort_values("abst", ascending=False).head(12)
print("\n### Clearest cells (|t| highest, sample>=50k), any horizon\n")
print("| n | k(tick) | k(pt) | x | mean(pt) | t | hit% | n_events | regime |")
print("|---|---|---|---|---|---|---|---|---|")
for _, r in top.iterrows():
    reg = "趋势 trend" if r["mean"] > 0 else "反转 reversal"
    print(f"| {int(r.n)} | {int(r.k_ticks)} | {r.k_pts:g} | {int(r.h)} | "
          f"{r['mean']:+.3f} | {r.t:+.1f} | {r.hitpct:.1f} | {int(r.s_n):,} | {reg} |")
print(f"\nsaved {FIG}/fig_exp1_ksweep_mean.png , {FIG}/fig_exp1_ksweep_t.png , "
      f"{FIG}/fig_exp1_ksweep_lines.png")
