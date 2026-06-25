"""exp2_burst_plot.py — figure for EXP2 burst, baseline t2/t1≈5 (ratio aligned).
Run with system python3:  python3 exp2_burst_plot.py
Reads exp2_burst.csv; writes figs/fig_volume_burst.png.
峰值 = V_t2/V_t1 < r（基线=t2/t1≈5，越接近 1 越爆发）。
 col1 尺度家族(比例=5)  col2 比例家族(t1=10, 比例4/5/6)  col3 向前窗口 x.
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
GREY = "#888888"


def mt(sum_, ss, n):
    m = sum_ / n
    return m, np.sqrt(np.maximum(ss / n - m * m, 0.0) / n)


v = pd.read_csv(f"{D}/exp2_burst.csv")
v["r"] = v["r"].astype(str)
v["key"] = v["t1"] * 1000 + v["t2"]
v["absmove"], v["abse"] = mt(v["a_sum"], v["a_ss"], v["n_peak"])
v["signed"], v["sige"] = mt(v["g_sum"], v["g_ss"], v["n_peak"])
v["fvol"], v["fve"] = mt(v["v_sum"], v["v_ss"], v["n_peak"])

RS = ["4.0", "3.0", "2.0", "1.5"]                 # mild -> extreme burst
RCOL = {"4.0": "#9bbcd6", "3.0": "#5b8fc9", "2.0": "#2f5e9e", "1.5": "#16305c"}
metrics = [("absmove", "①未来 |价格变动|（指数点）\n— 波动率", "abse", False, (0.6, 6.0)),
           ("signed", "②符号化未来收益（指数点）\n＋延续 · −反转", "sige", True, (-2.0, 0.4)),
           ("fvol", "③未来每 tick 成交量（手）", "fve", False, (0.0, 7.0))]

SCALE = [(4, 20), (8, 40), (12, 60), (20, 100), (30, 150)]   # ratio=5, vary scale
RATIO = [(10, 40), (10, 50), (10, 60)]                       # t1=10, ratio 4/5/6
HPAIR = (20, 100)                                            # for the horizon column

# (xkey-builder, x-tick labels, subset-rows, x-axis label, title)
def col_scale():
    keys = [t1 * 1000 + t2 for t1, t2 in SCALE]
    sub = v[(v.key.isin(keys)) & (v.h == 20)]
    return sub, "key", [f"{t1}/{t2}" for t1, t2 in SCALE], keys, \
        "t1/t2（比例固定=5）", "尺度家族：固定 t2/t1=5，扫窗口大小（x=20）"

def col_ratio():
    keys = [t1 * 1000 + t2 for t1, t2 in RATIO]
    sub = v[(v.key.isin(keys)) & (v.h == 20)]
    return sub, "key", [f"{t2//10}\n({10}/{t2})" for _, t2 in RATIO], keys, \
        "t2/t1 比例（t1=10）", "比例家族：固定 t1=10，扫比例 4/5/6（x=20）"

def col_horizon():
    sub = v[(v.key == HPAIR[0] * 1000 + HPAIR[1])]
    xs = sorted(sub["h"].unique())
    return sub, "h", [str(x) for x in xs], xs, \
        "x 向前窗口（tick；1 tick≈500毫秒）", f"向前窗口：固定 t1={HPAIR[0]}, t2={HPAIR[1]}（比例5）"

fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharey="row")
for ci, colfn in enumerate([col_scale, col_ratio, col_horizon]):
    sub, xcol, xticklab, xorder, xlab, ctitle = colfn()
    xpos = list(range(len(xorder)))
    for ri, (mcol, ylab, ecol, zero, ylim) in enumerate(metrics):
        ax = axes[ri, ci]
        if zero:
            ax.axhline(0, color="k", lw=0.8)
        base = sub[sub.r == "ALL"].set_index(xcol).reindex(xorder)
        ax.plot(xpos, base[mcol], color=GREY, lw=1.6, ls="--",
                marker="s", ms=5, label="ALL 全样本基线")
        for r in RS:
            s = sub[sub.r == r].set_index(xcol).reindex(xorder)
            ax.errorbar(xpos, s[mcol], yerr=s[ecol], color=RCOL[r], lw=1.9,
                        marker="o", ms=5, capsize=2.5, label=f"爆发 V_t2/V_t1＜{r}")
            for xp, val, nn in zip(xpos, s[mcol], s["n_peak"]):
                if pd.notna(val) and nn < 50000:
                    ax.plot(xp, min(max(val, ylim[0]), ylim[1]), marker="x",
                            color="#d62728", ms=7, mew=1.6)
        ax.set_xticks(xpos); ax.set_xticklabels(xticklab, fontsize=8)
        ax.set_ylim(*ylim); ax.margins(x=0.1)
        if ci == 0:
            ax.set_ylabel(ylab, fontsize=9.5)
        if ri == 0:
            ax.set_title(ctitle, fontsize=10)
        if ri == 2:
            ax.set_xlabel(xlab, fontsize=9.5)
axes[0, 0].legend(title="峰值强度（越小越爆发）", fontsize=7.6, title_fontsize=8.4, loc="best")
fig.suptitle("实验2（V_t2/V_t1 爆发版，基线 t2/t1≈5）放量爆发之后对随后 ticks 的影响　"
             "IM 中证1000 · 2022-07..2026-05", fontsize=12.5, y=1.0)
fig.text(0.5, -0.03,
         "峰值定义：spike = V_t2/V_t1（t2 含 t1），基线=t2/t1≈5；越接近 1 越爆发，事件 = spike < r。"
         "红 ✗ = 样本<5万、不可靠（多为安静期假爆发）。\n"
         "单位：t1/t2/x 为 tick（500毫秒/个）；①②为指数点（1点=¥200/手）；③为手。y 轴已按行裁剪。",
         ha="center", va="top", fontsize=8.3, color="#333333")
fig.tight_layout(); fig.savefig(f"{FIG}/fig_volume_burst.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"saved {FIG}/fig_volume_burst.png")

print("\n=== spike=V_t2/V_t1 (burst, ratio<r), baseline t2/t1=5, h=20 ===")
for (t1, t2), g in v[v.h == 20].groupby(["t1", "t2"]):
    print(f"-- t1={t1} t2={t2} (ratio {t2/t1:g}) --")
    for _, r in g.sort_values("r").iterrows():
        flag = " <-- thin" if r["n_peak"] < 50000 and r["r"] != "ALL" else ""
        print(f"   r={r['r']:>3}: absmove={r['absmove']:.3f} signed={r['signed']:+.4f} "
              f"fvol={r['fvol']:.2f} n={int(r['n_peak'])}{flag}")
