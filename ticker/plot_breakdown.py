"""
plot_breakdown.py — plots the per-minute window breakdowns produced by
open_breakdown.py.

UNITS (made explicit on every figure):
  * top panel    = mean_ret x 1e4  -> "bp, per-trade AVERAGE within the minute"
  * bottom panel = cumsum(total) x 1e4 -> "bp, raw SUM across signal days,
                   NOT averaged per day"; endpoint = the 30-min bar height
  * 1 bp = 0.01%. Broken line = no active sample in that minute.
Figures:
  figs/minute_{name}_{HHMM}_{HHMM}_t{thr}_{month}.png      (per month slice)
  figs/minute_{name}_{HHMM}_{HHMM}_t{thr}_bymonth.png      (Jan vs Feb overlay)
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# --- CJK font: pick the first family that ACTUALLY exists on this machine ---
_avail = {f.name for f in fm.fontManager.ttflist}
for _f in ["Arial Unicode MS", "Songti SC", "Heiti TC", "SimSun", "STHeiti",
           "PingFang HK", "Microsoft YaHei", "Noto Sans CJK SC",
           "WenQuanYi Zen Hei", "Noto Sans CJK JP"]:
    if _f in _avail:
        matplotlib.rcParams["font.sans-serif"] = [_f]
        break
matplotlib.rcParams["axes.unicode_minus"] = False

OUT = "/Users/zhuisabella/xn/ticker/open_breakdown"
FIG_DIR = f"{OUT}/figs"

THRESHOLDS = [5, 10, 15, 20]
EXCLUDE_MONTHS = ["2023-03"]                       # drop thin March sample
COL = {"IC0000": "#c0392b", "IF0000": "#DD8452",
       "IH0000": "#27ae60", "IM0000": "#4C72B0"}
LS = {"2023-01": "-", "2023-02": "--", "2023-03": ":"}
WIN_LABEL = {"pos_open": "最强红柱时段（正贡献）",
             "neg_pm":   "最深绿柱时段（负贡献）"}

FOOTNOTE = ("注：所有收益均为无量纲小数收益×10⁴换算成 bp（1 bp = 0.01%），非百分数。"
            "上图＝该分钟内单笔平均收益（mean_ret×10⁴）；"
            "下图＝窗口内逐分钟累加的收益总和（Σtotal×10⁴），为跨交易日求和、未做日均。"
            "线段中断处＝该分钟无有效样本。")


def _plot_skipna(ax, x, y, **kw):
    """Plot skipping NaN gaps is NOT desired: keep gaps visible (information)."""
    ax.plot(x, y, **kw)


def plot_threshold(d, thr, title_prefix, fname):
    """One month-slice: top = per-trade average (bp), bottom = cumulative sum (bp)."""
    sub = d[d["thr_ticks"] == thr]
    mins = sorted(sub["m"].unique())
    prods = [p for p in COL if p in sub["product"].unique()]
    if not mins or not prods:
        return
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7.6), sharex=True)
    for c in prods:
        s = sub[sub["product"] == c].set_index("m").reindex(mins)
        _plot_skipna(a1, range(len(mins)), s["mean_ret"].values * 1e4,
                     marker="o", ms=4, color=COL.get(c), label=c)
        _plot_skipna(a2, range(len(mins)),
                     s["total"].fillna(0).cumsum().values * 1e4,
                     marker="o", ms=3, color=COL.get(c), label=c)
    for ax in (a1, a2):
        ax.axhline(0, color="k", lw=.8)
        ax.grid(alpha=.25)
    a1.set_ylabel("单笔平均收益率 (bp/笔)")
    a1.set_title(f"{title_prefix} 逐分钟拆解 @ {thr} 跳\n"
                 f"上：该分钟单笔平均收益率　下：窗口累计收益总和（跨日求和，非日均）",
                 fontsize=11)
    a1.legend(ncol=len(prods), fontsize=9)
    a2.set_ylabel("累计收益总和 (bp，跨日求和)")
    a2.set_xticks(range(len(mins)), mins, rotation=45, ha="right")
    a2.set_xlabel("分钟")
    a2.legend(ncol=len(prods), fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(0.01, 0.01, FOOTNOTE, fontsize=7.5, color="0.35", wrap=True)
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def plot_month_compare(d, thr, title_prefix, fname):
    """Jan vs Feb overlay. Top = cumulative SUM per month (bp) — months differ
    in trading-day counts, so also show bottom = active sample size per minute."""
    sub = d[(d["thr_ticks"] == thr) & (d["month"] != "all")]
    mins = sorted(sub["m"].unique())
    prods = [p for p in COL if p in sub["product"].unique()]
    months = sorted(sub["month"].unique())
    if not mins or not prods or not months:
        return
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7.6), sharex=True)
    for c in prods:
        for mon in months:
            s = (sub[(sub["product"] == c) & (sub["month"] == mon)]
                 .set_index("m").reindex(mins))
            a1.plot(range(len(mins)),
                    s["total"].fillna(0).cumsum().values * 1e4,
                    ls=LS.get(mon, "-"), marker="o", ms=3,
                    color=COL.get(c), label=f"{c} {mon}")
            a2.plot(range(len(mins)), s["active"].values,
                    ls=LS.get(mon, "-"), marker="o", ms=3,
                    color=COL.get(c), label=f"{c} {mon}")
    for ax in (a1, a2):
        ax.axhline(0, color="k", lw=.8)
        ax.grid(alpha=.25)
    a1.set_ylabel("当月累计收益总和 (bp，跨日求和)")
    a1.set_title(f"{title_prefix} 分月对比 @ {thr} 跳（实线=1月，虚线=2月）\n"
                 f"上：各月累计收益总和（未按交易日数归一，对比时参考下图样本量）",
                 fontsize=11)
    a1.legend(ncol=len(prods), fontsize=8)
    a2.set_ylabel("有效分钟数（有信号的交易日数）")
    a2.set_xticks(range(len(mins)), mins, rotation=45, ha="right")
    a2.set_xlabel("分钟")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(0.01, 0.01, FOOTNOTE + " 各月交易日数不同，累计总和不可直接跨月比大小。",
             fontsize=7.5, color="0.35", wrap=True)
    fig.savefig(fname, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(OUT, "minute_*.csv")))
    if not files:
        raise SystemExit(f"no minute_*.csv found in {OUT} — run open_breakdown.py first")

    for fp in files:
        stem = os.path.splitext(os.path.basename(fp))[0]
        d = pd.read_csv(fp)
        d["m"] = d["minute"].astype(str).str[:5]
        if "month" not in d:
            d["month"] = "all"
        d = d[~d["month"].isin(EXCLUDE_MONTHS)]
        win_name = d["window"].iloc[0] if "window" in d else stem
        start, end = d["m"].min(), d["m"].max()
        title = f"{WIN_LABEL.get(win_name, win_name)} {start}-{end}"
        thrs = [t for t in sorted(d["thr_ticks"].unique()) if t in THRESHOLDS]
        for thr in thrs:
            for mon in sorted(d["month"].unique()):
                tag = "all" if mon == "all" else mon.replace("-", "")
                plot_threshold(d[d["month"] == mon], thr,
                               f"{title}（{mon}）" if mon != "all" else title,
                               f"{FIG_DIR}/{stem}_t{thr}_{tag}.png")
            plot_month_compare(d, thr, title, f"{FIG_DIR}/{stem}_t{thr}_bymonth.png")
        print(f"[{win_name}] figures saved for {stem}")
    print("all figures ->", FIG_DIR)
