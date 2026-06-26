"""exp2_x_plot.py — EXP2 forward-window sweep, small multiples (clean layout).
Run: python3 exp2_x_plot.py   (reads exp2_x.csv; writes 2 figs, one per method)
 每个子图 = 一个 (t1,t2) 组合（行=t1 尺度，列=比率）；子图内 y=阈值 r、x=向前窗口 x。
 色 = ②符号化收益均值（指数点，＋延续/−反转，裁剪±0.6）。红✗ = 样本<2万。
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
CLIP = 0.6
v = pd.read_csv(f"{D}/exp2_x.csv")
v["r"] = v["r"].astype(str)
v["signed"] = v["g_sum"] / v["n_peak"]
T1S = sorted(v["t1"].unique()); RATIOS = sorted({t2 // t1 for t1, t2 in zip(v.t1, v.t2)})
XS = sorted(v["x"].unique())
XLAB = [f"{x}\n{x*0.5:g}s" for x in XS]
MNAME = {"norm": "算法A 每tick平均量比 (V_t1/t1)/(V_t2/t2)",
         "total": "算法B 原始总量比 V_t1/V_t2"}


def make(method, RS):
    sub = v[v.method == method]
    nrow, ncol = len(T1S), len(RATIOS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.0 * nrow),
                             sharex=True, sharey=True, constrained_layout=True)
    im = None
    for ri, t1 in enumerate(T1S):
        for ci, rat in enumerate(RATIOS):
            ax = axes[ri, ci]; t2 = t1 * rat
            cell = sub[(sub.t1 == t1) & (sub.t2 == t2)]
            Z = cell.pivot(index="r", columns="x", values="signed").reindex(index=RS, columns=XS)
            Nn = cell.pivot(index="r", columns="x", values="n_peak").reindex(index=RS, columns=XS)
            im = ax.imshow(Z.values, cmap="seismic", vmin=-CLIP, vmax=CLIP,
                           aspect="auto", origin="upper")
            ax.set_xticks(range(len(XS))); ax.set_xticklabels(XLAB)
            ax.set_yticks(range(len(RS))); ax.set_yticklabels(RS)
            ax.tick_params(labelsize=7.5, length=0)
            for ii in range(len(RS)):
                for jj in range(len(XS)):
                    val = Z.values[ii, jj]
                    if not np.isfinite(val):
                        continue
                    thin = Nn.values[ii, jj] < 20000
                    ax.text(jj, ii, (f"{val:+.2f}" if abs(val) < 10 else f"{val:+.0f}"),
                            ha="center", va="center", fontsize=7,
                            color=("#777" if thin else
                                   ("white" if abs(min(max(val, -CLIP), CLIP)) > 0.6 * CLIP else "black")))
            ax.set_title(f"t1={t1}, t2={t2}（×{rat}）", fontsize=9.5)
    cb = fig.colorbar(im, ax=axes, fraction=0.04, pad=0.015, extend="both")
    cb.set_label("②符号化收益均值（指数点）　红=延续 / 蓝=反转", fontsize=9.5)
    fig.suptitle(f"实验2：放量爆发后【方向】随向前窗口 x 的变化 — {MNAME[method]}\n"
                 "行=t1 尺度，列=比率 t2/t1；子图内 y=阈值 r、x=向前 x　IM 中证1000（已过滤坏tick）",
                 fontsize=12.5)
    fig.supxlabel("x　向前窗口（tick / 秒）", fontsize=11)
    fig.supylabel("r　峰值阈值（spike＞r）", fontsize=11)
    fig.text(0.01, -0.01,
             f"峰值定义见标题（{MNAME[method]}），事件=spike＞r。②方向=sign(mid[i]−mid[i−t1])×(mid[i+x]−mid[i])。"
             "单位：t1/t2/x 为 tick(500ms)，数值为指数点(1点=¥200)。红✗(灰字)=样本<2万。色阶裁剪±0.6。",
             ha="left", va="top", fontsize=8.2, color="#444")
    fig.savefig(f"{FIG}/fig_exp2_x_{method}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {FIG}/fig_exp2_x_{method}.png")


make("norm", ["1.5", "2.0", "3.0"])
make("total", ["0.3", "0.5", "0.7"])
