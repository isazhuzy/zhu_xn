"""Pick with WHAT:
  WHAT=mom  (default)  minute-momentum curve   <- fwd{H}_momentum_{CODE}{SUF}.csv
  WHAT=voi             VOI-sorted cumsum curve <- voi_cumsum_curve{SUF}.csv
  WHAT=all             both
Run: SUF=_pilot WHAT=voi /Users/zhuisabella/xn/.venv/bin/python plot.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

_av = {f.name for f in fm.fontManager.ttflist}
for _f in ["Arial Unicode MS", "PingFang HK", "Heiti TC", "STHeiti", "Songti SC"]:
    if _f in _av:
        matplotlib.rcParams["font.sans-serif"] = [_f]; break
matplotlib.rcParams["axes.unicode_minus"] = False

D = "/Users/zhuisabella/xn/manual"
CODE = os.environ.get("CODE", "IF0000")
H = int(os.environ.get("H", "30"))
SUF = os.environ.get("SUF", "")
WHAT = os.environ.get("WHAT", "mom")
NAME = {"IC0000": "IC 中证500", "IF0000": "IF 沪深300",
        "IH0000": "IH 上证50", "IM0000": "IM 中证1000"}


def plot_momentum():
    s = pd.read_csv(f"{D}/fwd{H}_momentum_{CODE}{SUF}.csv")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(s))
    ax.axhline(0, color="0.5", lw=.7)
    ax.fill_between(x, s["mean"] - 2 * s.se, s["mean"] + 2 * s.se,
                    color="#4C72B0", alpha=.25, label="±2·se")
    ax.plot(x, s["mean"], lw=1.4, color="#4C72B0", label="均值")
    tick = [i for i, h in enumerate(s.hm) if h.endswith(("00", "30"))]
    ax.set_xticks(tick); ax.set_xticklabels(s.hm.iloc[tick], fontsize=8)
    ax.set_xlabel("日内分钟 t（信号 = 第 t 分钟方向）")
    ax.set_ylabel(f"动量策略收益（bps，持有 {H} 分钟）")
    ax.set_title(f"{CODE} 分钟动量：sign(第t分钟涨跌) × 未来{H}分钟收益，按日内分钟平均",
                 fontsize=11, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=.3)
    fig.tight_layout(); fig.savefig(f"{D}/fig_fwd{H}_momentum_{CODE}{SUF}.png", dpi=135)
    plt.close(fig)
    print(f"saved fig_fwd{H}_momentum_{CODE}{SUF}.png")


def plot_voi_cumsum():
    cv = pd.read_csv(f"{D}/voi_cumsum_curve{SUF}.csv")
    # for k fwd
    ks = sorted(int(c[3:]) for c in cv.columns if c.startswith("cum"))
    pal = ["0.6", "#27ae60", "#e08a3c", "#4C72B0", "#c0392b", "#8e44ad"]
    kcol = {k: pal[i % len(pal)] for i, k in enumerate(ks)}
    kmain = ks[-1]                          # longest horizon gets the min marker

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, code in zip(axes.ravel(), NAME):
        s = cv[cv.code == code].sort_values("rank")
        if s.empty:
            ax.set_axis_off(); continue
        x = s.q * 100
        for k in ks:
            ax.plot(x, s[f"cum{k}"], color=kcol[k], lw=1.8,
                    label=f"k={k} ({k*0.5:g}秒)")
        # z = s[s.voi == 0]                   # shade the big VOI==0 tie block
        # if len(z):
        #     ax.axvspan(z.q.min() * 100, z.q.max() * 100, color="0.85", alpha=.5,
        #                label="VOI=0 区间")
        ax.axhline(0, color="k", lw=.6)
        imin = s[f"cum{kmain}"].idxmin()    # bottom of the check mark
        ax.plot(s.q[imin] * 100, s[f"cum{kmain}"][imin], "v",
                color=kcol[kmain], ms=7)
        # ax.annotate(f"最低点 VOI≈{s.voi[imin]:.0f}",
        #             (s.q[imin] * 100, s[f"cum{kmain}"][imin]),
        #             textcoords="offset points", xytext=(6, -12),
        #             fontsize=8, color="0.3")
        top = ax.secondary_xaxis("top")     # VOI value living at each percentile
        qs = [1, 10, 30, 50, 70, 90, 99]
        vals = np.interp(qs, s.q * 100, s.voi)
        top.set_xticks(qs)
        top.set_xticklabels([f"{v:.0f}" for v in vals], fontsize=7.5)
        top.set_xlabel("该分位处的 VOI 值（手）", fontsize=8, color="0.35")
        ax.set_title(f"{NAME[code]}   n={s.n_total.iloc[0]:,}", fontsize=10.5,
                     fontweight="bold")
        ax.set_xlabel("VOI 排序（%）", fontsize=9)
        ax.set_ylabel("未来价格变动cumsum（tick）", fontsize=9)
        ax.legend(fontsize=8, loc="upper center"); ax.grid(True, alpha=.25)
    fig.suptitle("VOI 排序 → 未来k个快照价格变动的累计和\n",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(f"{D}/fig_voi_cumsum{SUF}.png", dpi=135); plt.close(fig)
    print(f"saved fig_voi_cumsum{SUF}.png  (k = {ks})")


if __name__ == "__main__":
    if WHAT in ("mom", "all"):
        plot_momentum()
    if WHAT in ("voi", "all"):
        plot_voi_cumsum()
