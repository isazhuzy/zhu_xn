"""exp2_ratio_plot.py — figure for the REDEFINED EXP2 (total-volume ratio).
Run with system python3:  python3 exp2_ratio_plot.py
Reads exp2_ratio.csv; writes figs/fig_volume_ratio.png.
峰值 spike = V_t1/V_t2 (原始总量之比，t2 包含 t1) = 近 t1 占 t2 总量的比例。
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
GREY, BLUE = "#888888", "#4C72B0"


def mt(sum_, ss, n):
    m = sum_ / n
    var = np.maximum(ss / n - m * m, 0.0)
    return m, np.sqrt(var / n)


v = pd.read_csv(f"{D}/exp2_ratio.csv")
v["r"] = v["r"].astype(str)
v["absmove"], v["abse"] = mt(v["a_sum"], v["a_ss"], v["n_peak"])
v["signed"], v["sige"] = mt(v["g_sum"], v["g_ss"], v["n_peak"])
v["fvol"], v["fve"] = mt(v["v_sum"], v["v_ss"], v["n_peak"])

RS = ["0.3", "0.5", "0.7"]
RCOL = {"0.3": "#9bbcd6", "0.5": BLUE, "0.7": "#1f3a5f"}
metrics = [("absmove", "①未来 |价格变动|（指数点）\n— 波动率", "abse", False),
           ("signed", "②符号化未来收益（指数点）\n＋延续 · −反转", "sige", True),
           ("fvol", "③未来每 tick 成交量\n（手）", "fve", False)]
columns = [("t2", "改变 t2 长窗（固定 t1=5, x=20）", (v.t1 == 5) & (v.h == 20),
            "t2　长窗（tick）"),
           ("t1", "改变 t1 短窗（固定 t2=120, x=20）", (v.t2 == 120) & (v.h == 20),
            "t1　短窗（tick）"),
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
                        marker="o", ms=5, capsize=2.5,
                        label=f"峰值 近t1占比＞{int(float(r)*100)}%")
        ax.set_xticks(xpos); ax.set_xticklabels([f"{int(x)}" for x in xvals])
        ax.margins(x=0.1)
        if ci == 0:
            ax.set_ylabel(ylab, fontsize=9.5)
        if ri == 0:
            ax.set_title(ctitle, fontsize=10.5)
        if ri == 2:
            ax.set_xlabel(xlab, fontsize=9.5)
axes[0, 0].legend(title="峰值强度（集中度）", fontsize=8, title_fontsize=8.6, loc="best")
fig.suptitle("实验2（新定义）成交量峰值 = 总量集中度：放量之后对随后 ticks 的影响　"
             "IM 中证1000 · 2022-07..2026-05", fontsize=12.5, y=1.0)
fig.text(0.5, -0.03,
         "新峰值定义：spike = V_t1 / V_t2 = 过去 t1 笔的【成交量总和】÷ 过去 t2 笔的总和，"
         "t2 窗口【包含】t1；= 近 t1 占 t2 总量的比例 ∈(0,1]。峰值 = 该比例 ＞ r。ALL=全样本。\n"
         "单位：t1、t2、x 均为 tick 个数（1 tick=500毫秒）；①②为指数点（1点=每手¥200）；③为手。"
         "①波动率 = 未来 x 笔的 |mid[i+x]−mid[i]| 平均。②=sign(短窗方向)×(mid[i+x]−mid[i])。",
         ha="center", va="top", fontsize=8.3, color="#333333")
fig.tight_layout(); fig.savefig(f"{FIG}/fig_volume_ratio.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"saved {FIG}/fig_volume_ratio.png")

# quick stdout readout (h=20)
print("\n=== spike=V_t1/V_t2 (总量比), h=20 ===")
for (t1, t2), g in v[v.h == 20].groupby(["t1", "t2"]):
    for _, r in g.sort_values("r").iterrows():
        print(f"  t1={t1:>2} t2={t2:>3} r={r['r']:>3}: absmove={r['absmove']:.3f} "
              f"signed={r['signed']:+.4f} fvol={r['fvol']:.2f} n={int(r['n_peak'])}")
