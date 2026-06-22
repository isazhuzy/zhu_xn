"""
minute_deepdive.py — day-by-day decomposition of selected "special minutes".

Answers: is a minute's total PnL a broad daily pattern, or driven by a few
extreme days? For each (product, minute) below and each threshold:
  * per-day return series at that minute (one row per signal day)
  * stats: n, mean, median, hit rate, t-stat vs 0, top-3-day share of total,
    Jan/Feb split
  * one figure per (product, minute): daily bars (red=+, green=-, 中国惯例),
    one panel per threshold, month boundary marked

Run on the RAW tick CSV (reuses matrix.py only).
Outputs:
  {OUT}/deepdive_daily.csv     - long table: product, thr_ticks, minute, date, ret
  {OUT}/deepdive_summary.csv   - the stats table
  {OUT}/figs_deepdive/deepdive_{product}_{HHMM}.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matrix import _minute_frame, apply_threshold
from datetime import time as dtime

# --- CJK font ---------------------------------------------------------------
_avail = {f.name for f in fm.fontManager.ttflist}
for _f in ["Arial Unicode MS", "Songti SC", "Heiti TC", "SimSun", "STHeiti",
           "PingFang HK", "Microsoft YaHei", "Noto Sans CJK SC",
           "WenQuanYi Zen Hei", "Noto Sans CJK JP"]:
    if _f in _avail:
        matplotlib.rcParams["font.sans-serif"] = [_f]
        break
matplotlib.rcParams["axes.unicode_minus"] = False

RAW = "/Users/zhuisabella/xn/ticker/IC_IF_IH_IM_20230104_20230304.csv"
OUT = "/Users/zhuisabella/xn/ticker/open_breakdown"
FIG_DIR = f"{OUT}/figs_deepdive"

# ===== CHANGE ME ============================================================
THRESHOLDS = [5, 10, 15, 20]
EXCLUDE_MONTHS = [3]                      # drop the ~2 March trading days

# (product, "HH:MM") pairs to decompose. Default = the key special minutes.
TARGETS = [
    ("IC0000", "09:31"), ("IM0000", "09:31"),   # 开盘首信号分钟（反转）
    ("IC0000", "09:34"), ("IM0000", "09:34"),   # 最强动量分钟
    ("IC0000", "13:37"), ("IM0000", "13:37"),   # 午后双合约显著负
    ("IC0000", "13:51"), ("IM0000", "13:51"),   # 午后崩塌分钟
    ("IC0000", "13:52"), ("IM0000", "13:52"),   # 13:51 的反向修复
]
# ============================================================================

RED, GREEN = "#c0392b", "#27ae60"          # 红涨绿跌




def day_series(R, minute):
    """Per-day returns at one minute: drop days with no signal (0/NaN)."""
    h, m = map(int, minute.split(":"))
    key = dtime(h, m)
    if key not in R.columns:
        # fallback: string match, in case columns are strings on other runs
        cols = {str(c)[:5]: c for c in R.columns}
        if minute not in cols:
            return pd.Series(dtype=float)
        key = cols[minute]
    s = R[key].replace(0, np.nan).dropna()
    s.index = pd.to_datetime(s.index)
    s = s[~s.index.month.isin(EXCLUDE_MONTHS)]
    return s.sort_index()


def stats_row(s):
    n = len(s)
    if n == 0:
        return None
    total = s.sum()
    hit = (s > 0).mean()
    t = s.mean() / (s.std(ddof=1) / np.sqrt(n)) if n > 1 and s.std(ddof=1) > 0 else np.nan
    # top-3 days in the direction of the total, as a share of total
    top3 = s.sort_values(ascending=(total < 0)).iloc[:3].sum()
    share = top3 / total if total != 0 else np.nan
    jan, feb = s[s.index.month == 1], s[s.index.month == 2]
    return dict(n=n, total_bp=total * 1e4, mean_bp=s.mean() * 1e4,
                median_bp=s.median() * 1e4, hit=hit, t=t,
                top3_share=share,
                jan_total_bp=jan.sum() * 1e4, jan_n=len(jan),
                feb_total_bp=feb.sum() * 1e4, feb_n=len(feb))


def plot_target(series_by_thr, product, minute, fname):
    thrs = [t for t in THRESHOLDS if t in series_by_thr and len(series_by_thr[t])]
    if not thrs:
        return
    # common date axis = union of all dates (so panels align vertically)
    dates = sorted(set().union(*[set(series_by_thr[t].index) for t in thrs]))
    x = np.arange(len(dates))
    pos = {d: i for i, d in enumerate(dates)}
    fig, axes = plt.subplots(len(thrs), 1, figsize=(12, 2.3 * len(thrs)),
                             sharex=True, squeeze=False)
    for ax, thr in zip(axes[:, 0], thrs):
        s = series_by_thr[thr]
        st = stats_row(s)
        xi = [pos[d] for d in s.index]
        ax.bar(xi, s.values * 1e4,
               color=[RED if v > 0 else GREEN for v in s.values], width=.8)
        ax.axhline(0, color="k", lw=.8)
        # month boundary
        for i in range(1, len(dates)):
            if dates[i].month != dates[i - 1].month:
                ax.axvline(i - 0.5, color="0.5", ls="--", lw=.8)
        ax.grid(alpha=.2, axis="y")
        ax.set_ylabel(f"@{thr}跳\n(bp)")
        ax.text(.995, .95,
                f"n={st['n']}  均值={st['mean_bp']:.2f}  中位={st['median_bp']:.2f}bp  "
                f"胜率={st['hit']:.2f}  t={st['t']:.1f}  前3日占比={st['top3_share']:.0%}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(fc="white", ec="0.8", alpha=.8))
    axes[0, 0].set_title(f"{product} {minute} 逐日收益率分解（红=正 绿=负；虚线=月界；"
                         f"前3日占比越高说明越依赖个别交易日）")
    step = max(1, len(dates) // 20)
    axes[-1, 0].set_xticks(x[::step],
                           [d.strftime("%m-%d") for d in dates][::step],
                           rotation=45, ha="right")
    axes[-1, 0].set_xlabel("交易日")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    df = pd.read_csv(RAW, dtype={"code": "string"}, parse_dates=["m_nDatetime"])

    contracts = sorted({p for p, _ in TARGETS})
    frames = {c: _minute_frame(df, c, use_mid=True, mode="momentum", lookback=1)
              for c in contracts}
    R_cache = {(c, t): apply_threshold(frames[c], t, "tick")
               for c in contracts for t in THRESHOLDS}

    daily_rows, summary_rows = [], []
    for product, minute in TARGETS:
        series_by_thr = {}
        for thr in THRESHOLDS:
            s = day_series(R_cache[(product, thr)], minute)
            series_by_thr[thr] = s
            if len(s) == 0:
                continue
            daily_rows.append(pd.DataFrame(
                {"product": product, "thr_ticks": thr, "minute": minute,
                 "date": s.index.strftime("%Y-%m-%d"), "ret": s.values}))
            st = stats_row(s)
            summary_rows.append({"product": product, "thr_ticks": thr,
                                 "minute": minute, **st})
        tag = minute.replace(":", "")
        plot_target(series_by_thr, product, minute,
                    f"{FIG_DIR}/deepdive_{product}_{tag}.png")
        print(f"done {product} {minute}")
    if not daily_rows:
        raise SystemExit("no data extracted — check that TARGETS minutes "
                         "match R column labels (HH:MM vs HH:MM:SS)")


    pd.concat(daily_rows, ignore_index=True).to_csv(
        f"{OUT}/deepdive_daily.csv", index=False)
    summ = pd.DataFrame(summary_rows).round(
        {"total_bp": 1, "mean_bp": 2, "median_bp": 2, "hit": 3, "t": 2,
         "top3_share": 2, "jan_total_bp": 1, "feb_total_bp": 1})
    summ.to_csv(f"{OUT}/deepdive_summary.csv", index=False)
    print(f"\nsaved {OUT}/deepdive_daily.csv, deepdive_summary.csv, figs -> {FIG_DIR}")
    print(summ.to_string(index=False))
