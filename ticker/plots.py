"""
Figures for the momentum sweep.
  Fig 1  profitability heatmap   (reads summary.csv -> no recompute, all contracts)
  Fig 2  intraday PnL profile    (one contract, recomputed)        -> Q2/Q3
  Fig 3  cross-month stability   (one contract, recomputed)        -> noise check
  Fig 4  2x2 intraday grid       (the four 0000 series on one slide)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matrix import _minute_frame, apply_threshold, split_by_month
from analyze import bucket_profile

# ---- language + colours ----------------------------------------------------
LANG = "en"                                   # "en" or "zh"
def setup_font():
    if LANG == "zh":
        for f in ["PingFang SC", "Arial Unicode MS", "Heiti SC", "STHeiti",
                  "Songti SC", "Microsoft YaHei"]:
            matplotlib.rcParams["font.sans-serif"] = [f]
            matplotlib.rcParams["axes.unicode_minus"] = False
            break
L = {"en": {"thr": "signal threshold (ticks)", "prod": "contract",
            "prof": "share of profitable minutes", "pnl": "total PnL contribution",
            "hit": "hit rate", "tod": "time of day (30-min)",
            "meanret": "mean return / active min"},
     "zh": {"thr": "信号阈值（跳）", "prod": "合约", "prof": "盈利分钟占比",
            "pnl": "累计收益贡献", "hit": "胜率", "tod": "日内时段（30分钟）",
            "meanret": "每有效分钟平均收益"}}[LANG]
C_GAIN, C_LOSS = "#c0392b", "#27ae60"          # red=gain, green=loss (CN convention)

STATS_DIR = "/Users/zhuisabella/xn/ticker/momentum_stats"
FIG_DIR   = "/Users/zhuisabella/xn/ticker/figs"


def fig_heatmap(summary_csv, fname, products=None):
    s = pd.read_csv(summary_csv)
    s = s[s["month"] == "ALL"]
    if products:
        s = s[s["product"].isin(products)]
    M = s.pivot(index="product", columns="thr_ticks", values="pct_profitable")
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(M) + 1.5))
    im = ax.imshow(M.values, cmap="RdYlGn_r", vmin=0.40, vmax=0.60, aspect="auto")
    ax.set_xticks(range(M.shape[1]), M.columns)
    ax.set_yticks(range(M.shape[0]), M.index)
    ax.set_xlabel(L["thr"]); ax.set_ylabel(L["prod"])
    if len(M) <= 12:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if abs(v-0.5) > 0.06 else "black")
    fig.colorbar(im, ax=ax).set_label(L["prof"])
    ax.set_title(L["prof"] + "  (0.50 = coin flip)")
    fig.tight_layout(); fig.savefig(fname, dpi=150); plt.close(fig)


def _profile(df, c, thr):
    frame = _minute_frame(df, c, use_mid=True, mode="momentum", lookback=1)
    return apply_threshold(frame, thr, "tick")


def fig_intraday(df, c, thr, fname):
    p = bucket_profile(_profile(df, c, thr))
    colors = [C_GAIN if v >= 0 else C_LOSS for v in p["total"]]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    a1.bar(range(len(p)), p["total"], color=colors); a1.axhline(0, color="k", lw=.8)
    a1.set_ylabel(L["pnl"]); a1.set_title(f"{c}  @ {thr} ticks")
    a2.bar(range(len(p)), p["hit_rate"],
           color=[C_GAIN if h >= .5 else C_LOSS for h in p["hit_rate"]])
    a2.axhline(.5, color="k", lw=.8, ls="--"); a2.set_ylim(0, 1)
    a2.set_ylabel(L["hit"]); a2.set_xticks(range(len(p)), p.index, rotation=45, ha="right")
    a2.set_xlabel(L["tod"]); fig.tight_layout(); fig.savefig(fname, dpi=150); plt.close(fig)


def fig_stability(df, c, thr, fname):
    R = _profile(df, c, thr)
    profs = {m: bucket_profile(Rm)["mean_ret"] for m, Rm in split_by_month(R).items()}
    buckets = sorted(set().union(*[p.index for p in profs.values()]))
    fig, ax = plt.subplots(figsize=(8, 4.5)); w = 0.8 / max(len(profs), 1)
    for k, (m, s) in enumerate(profs.items()):
        ax.bar(np.arange(len(buckets)) + k*w, [s.get(b, np.nan) for b in buckets],
               width=w, label=m)
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(np.arange(len(buckets)) + w*(len(profs)-1)/2, buckets, rotation=45, ha="right")
    ax.set_ylabel(L["meanret"]); ax.set_xlabel(L["tod"])
    ax.set_title(f"{c} @ {thr} ticks — month by month (do the bars agree?)")
    ax.legend(); fig.tight_layout(); fig.savefig(fname, dpi=150); plt.close(fig)


def fig_intraday_grid(df, contracts, thr, fname):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, c in zip(axes.ravel(), contracts):
        p = bucket_profile(_profile(df, c, thr))
        ax.bar(range(len(p)), p["total"],
               color=[C_GAIN if v >= 0 else C_LOSS for v in p["total"]])
        ax.axhline(0, color="k", lw=.8); ax.set_title(f"{c} @ {thr} ticks")
        ax.set_xticks(range(len(p)), p.index, rotation=45, ha="right", fontsize=7)
    fig.supylabel(L["pnl"]); fig.tight_layout(); fig.savefig(fname, dpi=150); plt.close(fig)


def fig_month_stability(buckets_csv, contract, thr, fname):
    """Per-contract: 8 intraday buckets, grouped bars by month — does the
    intraday shape (incl. the open) repeat month to month?"""
    b = pd.read_csv(buckets_csv)
    b = b[(b["product"] == contract) & (b["thr_ticks"] == thr) & (b["month"] != "ALL")]
    if b.empty:
        print(f"no per-month rows for {contract} @ {thr}"); return
    piv = b.pivot(index="bucket", columns="month", values="mean_ret").sort_index()
    months = list(piv.columns)
    fig, ax = plt.subplots(figsize=(8.5, 4.6)); w = 0.8 / max(len(months), 1)
    for k, m in enumerate(months):
        ax.bar(np.arange(len(piv)) + k * w, piv[m].values * 1e4, width=w, label=m)
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(np.arange(len(piv)) + w * (len(months) - 1) / 2, piv.index,
                  rotation=45, ha="right")
    ax.set_ylabel(L["meanret"] + " (bp)"); ax.set_xlabel(L["tod"])
    t = (f"{contract} @ {thr} 跳 — 分月对照（各时段是否逐月一致？）" if LANG == "zh"
         else f"{contract} @ {thr} ticks — by month (does each bucket agree?)")
    ax.set_title(t)
    ax.legend(title="month"); fig.tight_layout(); fig.savefig(fname, dpi=150); plt.close(fig)


def fig_open_stability(buckets_csv, products, thr, fname, bucket="09:30"):
    """Focused: the opening bucket's mean return, per product, grouped by month.
    All-positive across months = the open effect is real; sign flips = noise."""
    b = pd.read_csv(buckets_csv)
    b = b[(b["thr_ticks"] == thr) & (b["bucket"] == bucket) &
          (b["month"] != "ALL") & (b["product"].isin(products))]
    if b.empty:
        print(f"no rows for bucket {bucket} @ {thr}"); return
    piv = b.pivot(index="product", columns="month", values="mean_ret").reindex(products)
    months = list(piv.columns)
    fig, ax = plt.subplots(figsize=(8, 4.4)); w = 0.8 / max(len(months), 1)
    for k, m in enumerate(months):
        ax.bar(np.arange(len(piv)) + k * w, piv[m].values * 1e4, width=w, label=m)
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(np.arange(len(piv)) + w * (len(months) - 1) / 2, piv.index)
    ax.set_ylabel(L["meanret"] + " (bp)")
    t = (f"开盘 {bucket} 段分月对照 @ {thr} 跳（柱同号=稳定，变号=噪音）" if LANG == "zh"
         else f"open {bucket} by month @ {thr} ticks (same sign=stable, flip=noise)")
    ax.set_title(t)
    ax.legend(title="month"); fig.tight_layout(); fig.savefig(fname, dpi=150); plt.close(fig)


if __name__ == "__main__":
    setup_font()
    os.makedirs(FIG_DIR, exist_ok=True)
    df = pd.read_csv(
        "/Users/zhuisabella/xn/ticker/IC_IF_IH_IM_20230104_20230304.csv",
        dtype={"code": "string"}, parse_dates=["m_nDatetime"],
    )
    main4 = ["IC0000", "IF0000", "IH0000", "IM0000"]

    fig_heatmap(f"{STATS_DIR}/summary.csv", f"{FIG_DIR}/fig1_profitability.png",
                products=main4)                       # drop products= to show all 53
    fig_intraday(df, "IF0000", 5, f"{FIG_DIR}/fig2_intraday_IF0000_t5.png")
    fig_stability(df, "IF0000", 5, f"{FIG_DIR}/fig3_stability_IF0000_t5.png")
    fig_intraday_grid(df, main4, 5, f"{FIG_DIR}/fig4_grid_t5.png")

    # NEW: per-month stability (reads buckets_all.csv — rerun analyze.py first
    # so it contains the 'month' column)
    BK = f"{STATS_DIR}/buckets_all.csv"
    fig_month_stability(BK, "IF0000", 5, f"{FIG_DIR}/fig5_monthstab_IF0000_t5.png")
    fig_open_stability(BK, main4, 5, f"{FIG_DIR}/fig6_openstab_t5.png")
    print("saved figures to", FIG_DIR)
