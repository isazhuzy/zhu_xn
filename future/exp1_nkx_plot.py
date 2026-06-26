"""exp1_nkx_plot.py — two small-multiple figures over the n×k×x grid (clean layout).
Run: python3 exp1_nkx_plot.py   (reads exp1_nkx.csv; writes 2 figs)
 fig_pulse_nkx_byN.png : 每子图固定一个 n；子图内 y=k(价格tick)、x=x(向前tick)。
 fig_pulse_nkx_byK.png : 每子图固定一个 k；子图内 y=n(回看tick)、x=x(向前tick)。
色 = 符号化收益均值（指数点，红趋势/蓝反转，裁剪±0.5）。共享坐标、公共轴标题，避免文字重叠。
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
CLIP = 0.5
d = pd.read_csv(f"{D}/exp1_nkx.csv")
d["mean"] = d["s_sum"] / d["s_n"]
NS = sorted(d["n"].unique()); KT = sorted(d["k_ticks"].unique()); XS = sorted(d["x"].unique())
XLAB = [f"{x}\n{x*0.5:g}s" for x in XS]


def small_multiples(fix_col, panel_vals, ycol, yvals, ylab, supx, fname, main, note):
    ncol = 3
    nrow = int(np.ceil(len(panel_vals) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 4.7 * nrow),
                             sharex=True, sharey=True, constrained_layout=True)
    axflat = axes.flatten()
    im = None
    for ax, pv in zip(axflat, panel_vals):
        Z = (d[d[fix_col] == pv]
             .pivot(index=ycol, columns="x", values="mean")
             .reindex(index=yvals, columns=XS))
        im = ax.imshow(Z.values, cmap="seismic", vmin=-CLIP, vmax=CLIP,
                       aspect="auto", origin="upper")
        ax.set_xticks(range(len(XS))); ax.set_xticklabels(XLAB)
        ax.set_yticks(range(len(yvals))); ax.set_yticklabels(yvals)
        ax.tick_params(labelsize=8, length=0)
        for ii in range(len(yvals)):
            for jj in range(len(XS)):
                v = Z.values[ii, jj]
                if np.isfinite(v):
                    ax.text(jj, ii, (f"{v:+.2f}" if abs(v) < 10 else f"{v:+.0f}"),
                            ha="center", va="center", fontsize=7,
                            color="white" if abs(min(max(v, -CLIP), CLIP)) > 0.6 * CLIP else "black")
        if fix_col == "k_ticks":
            ax.set_title(f"k = {pv} tick（{pv*0.2:g} 点）", fontsize=11)
        else:
            ax.set_title(f"n = {pv} tick（{pv*0.5:g} 秒）", fontsize=11)
    for ax in axflat[len(panel_vals):]:
        ax.axis("off")
    cb = fig.colorbar(im, ax=axes, fraction=0.045, pad=0.015, extend="both")
    cb.set_label("符号化收益均值（指数点）　红 = 趋势 / 蓝 = 反转", fontsize=10)
    fig.suptitle(main, fontsize=14)
    fig.supxlabel(supx, fontsize=11)
    fig.supylabel(ylab, fontsize=11)
    fig.text(0.01, -0.012, note, ha="left", va="top", fontsize=8.5, color="#444")
    fig.savefig(f"{FIG}/{fname}", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {FIG}/{fname}")


NOTE = ("计算方法：脉冲 = mid[i]−mid[i−n]，|脉冲|>k 触发；色 = sign(脉冲)×(mid[i+x]−mid[i]) 的均值（裁剪 ±0.5 点）。"
        "单位：n、x 为 tick（500ms/个），k 为价格 tick（0.2 点），数值为指数点（1 点 = 每手 ¥200）。已过滤坏 tick。")

small_multiples(
    "n", NS, "k_ticks", KT, "k　脉冲幅度阈值（价格 tick）", "x　向前窗口（tick / 秒）",
    "fig_pulse_nkx_byN.png",
    "实验1：固定 n（每子图一个回看），看 k × x 的趋势↔反转\nIM 中证1000 · 2022-07..2026-05（已过滤坏tick）",
    NOTE)

small_multiples(
    "k_ticks", KT, "n", NS, "n　回看窗口（tick）", "x　向前窗口（tick / 秒）",
    "fig_pulse_nkx_byK.png",
    "实验1：固定 k（每子图一个阈值），看 n × x 的趋势↔反转\nIM 中证1000 · 2022-07..2026-05（已过滤坏tick）",
    NOTE)
