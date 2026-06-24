"""tickcount_hist_plot.py — fig56: ticks-per-minute distribution per contract.
Reads ticks_per_minute_hist.csv (code, cnt, n_minutes). Run: python3 tickcount_hist_plot.py
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
_av = {f.name for f in fm.fontManager.ttflist}
for _f in ["Arial Unicode MS", "PingFang HK", "Heiti TC", "STHeiti", "Songti SC"]:
    if _f in _av:
        matplotlib.rcParams["font.sans-serif"] = [_f]; break
matplotlib.rcParams["axes.unicode_minus"] = False

D = "/Users/zhuisabella/xn/intraminute"
NAME = {"IC0000": "IC 中证500", "IF0000": "IF 沪深300", "IH0000": "IH 上证50", "IM0000": "IM 中证1000"}
COL = {"IC0000": "#c0392b", "IF0000": "#e08a3c", "IH0000": "#27ae60", "IM0000": "#4C72B0"}
h = pd.read_csv(f"{D}/ticks_per_minute_hist.csv")


def stats(cnt, n):
    order = np.argsort(cnt); cnt, n = cnt[order], n[order]
    tot = n.sum(); mean = (cnt * n).sum() / tot
    cum = np.cumsum(n)
    med = cnt[np.searchsorted(cum, .5 * tot)]
    return mean, med, tot


fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
for ax, code in zip(axes.ravel(), NAME):
    s = h[h.code == code].sort_values("cnt")
    cnt = s["cnt"].to_numpy().astype(float); n = s["n_minutes"].to_numpy().astype(float)
    mean, med, tot = stats(cnt, n)
    frac = n / tot * 100
    full = 100 * n[cnt >= 115].sum() / tot
    tail = 100 * n[cnt > 120].sum() / tot
    ax.bar(cnt, frac, width=1.0, color=COL[code], alpha=0.75)
    ax.axvline(120, color="0.35", lw=1.2, ls="-", label="500ms 上限 = 120")
    ax.axvline(med, color="k", lw=1.4, ls="--", label=f"中位数 = {med:.0f}")
    ax.axvline(mean, color="0.4", lw=1.4, ls=":", label=f"均值 = {mean:.1f}")
    ax.set_xlim(0, 135)
    ax.set_xlabel("每分钟 tick 数（去重时间戳）", fontsize=9)
    ax.set_ylabel("占该合约全部分钟的 %", fontsize=9)
    ax.set_title("%s   中位数 %.0f · 均值 %.1f · ≥115满tick %.0f%% · >120(超密)%.1f%%"
                 % (NAME[code], med, mean, full, tail), fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, axis="y", alpha=0.25)
fig.suptitle("图56　每分钟实际 tick 数分布（2020-01..2026-05；去重时间戳；会话内分钟）\n"
             "左偏：多数分钟接近 500ms 满档(120)，一条安静分钟的尾巴把均值拉低；x>135 的超密尾部已省略",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(f"{D}/figs/fig56_每分钟tick数分布.png", dpi=135); plt.close(fig)
print("saved fig56")
