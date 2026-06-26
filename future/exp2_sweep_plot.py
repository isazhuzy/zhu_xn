"""exp2_sweep_plot.py — two clean sweeps × two ratio calcs.
Run: python3 exp2_sweep_plot.py    (reads exp2_sweep.csv; writes 2 figs)
 fig_sweep_t2.png : fix t1=10, sweep t2   (cols = norm calc | total calc)
 fig_sweep_t1.png : fix t2=120, sweep t1
Rows = ①波动 ②方向 ③未来量.  Lines = peak thresholds.  红✗ = n<5万.
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


def mt(s, ss, n):
    m = s / n
    return m, np.sqrt(np.maximum(ss / n - m * m, 0.0) / n)


v = pd.read_csv(f"{D}/exp2_sweep.csv")
v["r"] = v["r"].astype(str)
v["absmove"], v["abse"] = mt(v["a_sum"], v["a_ss"], v["n_peak"])
v["signed"], v["sige"] = mt(v["g_sum"], v["g_ss"], v["n_peak"])
v["fvol"], v["fve"] = mt(v["v_sum"], v["v_ss"], v["n_peak"])

metrics = [("absmove", "①未来 |价格变动|（点）", "abse", False, (0.8, 6.0)),
           ("signed", "②符号化未来收益（点）\n＋延续 · −反转", "sige", True, (-2.0, 0.4)),
           ("fvol", "③未来每 tick 量（手）", "fve", False, (0.0, 7.0))]
METH = [("norm", "算法A 归一化：spike=(V_t1/t1)/(V_t2/t2)　基线=1，可跨窗口比",
         ["1.5", "2.0", "3.0"], {"1.5": "#9bbcd6", "2.0": "#2f5e9e", "3.0": "#16305c"},
         lambda r: f"spike＞{r}（{r}×平均量）"),
        ("total", "算法B 原始总量：spike=V_t1/V_t2　占比∈(0,1]，不可跨窗口比",
         ["0.3", "0.5", "0.7"], {"0.3": "#f4b183", "0.5": "#d6701c", "0.7": "#7a3b06"},
         lambda r: f"spike＞{r}（近t1占比>{int(float(r)*100)}%）")]


def make_fig(xcol, fixcol, fixval, xlab, suptitle, fname):
    sub0 = v[(v[fixcol] == fixval) & (v.h == 20)]     # main sweep at x=20 (10s)
    xvals = sorted(sub0[xcol].unique())
    xpos = list(range(len(xvals)))
    xticklab = [f"{int(x)}\n({x*0.5:g}s)" for x in xvals]
    fig, axes = plt.subplots(3, 2, figsize=(13, 11.5), sharey="row")
    for ci, (me, mtitle, RS, RCOL, lab) in enumerate(METH):
        sm = sub0[sub0.method == me]
        for ri, (mcol, ylab, ecol, zero, ylim) in enumerate(metrics):
            ax = axes[ri, ci]
            if zero:
                ax.axhline(0, color="k", lw=0.8)
            base = sm[sm.r == "ALL"].set_index(xcol).reindex(xvals)
            ax.plot(xpos, base[mcol], color=GREY, lw=1.6, ls="--",
                    marker="s", ms=5, label="ALL 基线")
            for r in RS:
                s = sm[sm.r == r].set_index(xcol).reindex(xvals)
                ax.errorbar(xpos, s[mcol], yerr=s[ecol], color=RCOL[r], lw=1.9,
                            marker="o", ms=5, capsize=2.5, label=lab(r))
                for xp, val, nn in zip(xpos, s[mcol], s["n_peak"]):
                    if pd.notna(val) and nn < 50000:
                        ax.plot(xp, min(max(val, ylim[0]), ylim[1]), marker="x",
                                color="#d62728", ms=7, mew=1.6)
            ax.set_xticks(xpos); ax.set_xticklabels(xticklab, fontsize=7.5)
            ax.set_ylim(*ylim); ax.margins(x=0.08)
            if ci == 0:
                ax.set_ylabel(ylab, fontsize=9.5)
            if ri == 0:
                ax.set_title(mtitle, fontsize=9.5)
            if ri == 2:
                ax.set_xlabel(xlab, fontsize=9.5)
            if ri == 1 and ci == 0:
                ax.legend(fontsize=7.4, loc="best")
    fig.suptitle(suptitle, fontsize=12.5, y=1.0)
    fig.text(0.5, -0.035,
             "1 tick = 500 毫秒（x 轴括号为真实时长）。两列 = 两种成交量比算法（同样窗口、不同阈值口径）；峰值 = spike＞r。\n"
             "行指标：①波动 = |mid[i+x]−mid[i]| 平均；②方向 = sign(mid[i]−mid[i−t1])×(mid[i+x]−mid[i])，＋延续/−反转；③未来 x 笔每 tick 量。\n"
             "单位：t1/t2/x 为 tick；①②为指数点（1 点 = 每手 ¥200）；③为手。红✗ = 样本<5万、不可靠。y 轴按行裁剪。",
             ha="center", va="top", fontsize=8.0, color="#333")
    fig.tight_layout(); fig.savefig(f"{FIG}/{fname}", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {FIG}/{fname}")


make_fig("t2", "t1", 10, "t2　长窗（tick）",
         "实验2 · 固定 t1=10（5秒），扫长窗 t2　IM 中证1000 · 2022-07..2026-05",
         "fig_sweep_t2.png")
make_fig("t1", "t2", 120, "t1　短窗（tick）",
         "实验2 · 固定 t2=120（60秒），扫短窗 t1　IM 中证1000 · 2022-07..2026-05",
         "fig_sweep_t1.png")

# readout: signed + n at h=20, both methods
for xcol, fixcol, fixval in [("t2", "t1", 10), ("t1", "t2", 120)]:
    print(f"\n=== fix {fixcol}={fixval}, sweep {xcol}, h=20 (signed / n) ===")
    sub = v[(v[fixcol] == fixval) & (v.h == 20)]
    for me, *_ in METH:
        print(f" -- {me} --")
        for xv in sorted(sub[xcol].unique()):
            cells = sub[(sub.method == me) & (sub[xcol] == xv)].sort_values("r")
            txt = "  ".join(f"{r['r']}:{r['signed']:+.3f}(n={int(r['n_peak']/1000)}k)"
                            for _, r in cells.iterrows())
            print(f"   {xcol}={xv:>3}: {txt}")
