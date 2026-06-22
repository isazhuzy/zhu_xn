"""
make_figures.py — conclusion figures for the open first-two-minute study.
All titles Chinese, all axes labelled with Mandarin names + units.

Both samples come from ONE file (full-history bars), so they are consistent:
  全历史  = 2015-2026 (cleaned)        ~2660 days (IC/IF/IH), ~890 (IM)
  38天    = 2023-01-04 .. 2023-02-28   (the original window)

Run:  /usr/local/bin/python3.14 make_figures.py   (python3.14 has matplotlib)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle
from analyze_open_bars import load_wide, clean, momentum_at, contrarian_row, PRODUCTS

# ---- CJK font (Arial Unicode MS handles hanzi + the minus sign) -------------
_av = {f.name for f in fm.fontManager.ttflist}
for _f in ["Arial Unicode MS", "PingFang HK", "Heiti TC", "STHeiti", "Songti SC"]:
    if _f in _av:
        matplotlib.rcParams["font.sans-serif"] = [_f]
        break
matplotlib.rcParams["axes.unicode_minus"] = False

BARS = "/Users/zhuisabella/xn/ticker/open_breakdown/open_bars_all_2015_2026.csv"
OUT = "/Users/zhuisabella/xn/ticker/figs_conclusion"
os.makedirs(OUT, exist_ok=True)

NAME = {"IC0000": "IC 中证500", "IF0000": "IF 沪深300",
        "IH0000": "IH 上证50", "IM0000": "IM 中证1000"}
TICK_LBL = {"IC0000": "IC\n中证500", "IF0000": "IF\n沪深300",
            "IH0000": "IH\n上证50", "IM0000": "IM\n中证1000"}
COL = {"IC0000": "#c0392b", "IF0000": "#e08a3c",
       "IH0000": "#27ae60", "IM0000": "#4C72B0"}
RED, BLUE = "#c0392b", "#2c6fbb"
THRS = [5, 10, 15, 20, 25, 30]

w_full = clean(load_wide(BARS), 150)
w_full["yr"] = w_full["d"].dt.year
w_38 = w_full[(w_full["d"] >= pd.Timestamp("2023-01-04")) &
              (w_full["d"] <= pd.Timestamp("2023-02-28"))]
print("samples — 全历史 days:",
      {p: int(w_full[w_full.code == p].c30.notna().sum()) for p in PRODUCTS},
      "| 38天 days:", int(w_38[w_38.code == "IC0000"].c30.notna().sum()))


def stat(w, p, minute, thr):
    s = momentum_at(w[w.code == p], minute, thr)
    return contrarian_row(s) if len(s) else None


def fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y)); x, y = x[m], y[m]
    b1, b0 = np.polyfit(x, y, 1)
    return b1, b0, np.corrcoef(x, y)[0, 1] ** 2, len(x)


# =============================================================== Fig 1
def fig1():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    x = np.arange(4); w = 0.38
    e38 = [stat(w_38, p, "09:31", 5)["avg_bp"] for p in PRODUCTS]
    eAll = [stat(w_full, p, "09:31", 5)["avg_bp"] for p in PRODUCTS]
    w38 = [stat(w_38, p, "09:31", 5)["win"] * 100 for p in PRODUCTS]
    wAll = [stat(w_full, p, "09:31", 5)["win"] * 100 for p in PRODUCTS]
    for ax, a, b, ylab, ttl in [
            (ax1, e38, eAll, "每笔平均收益（bp，1bp=0.01%）", "(a) 每笔平均收益（毛）"),
            (ax2, w38, wAll, "胜率（%）", "(b) 胜率")]:
        b1 = ax.bar(x - w / 2, a, w, color=RED, label="38天（2023年1–2月）")
        b2 = ax.bar(x + w / 2, b, w, color=BLUE, label="全历史（2015–2026）")
        ax.bar_label(b1, fmt="%.1f", fontsize=8, padding=2)
        ax.bar_label(b2, fmt="%.1f", fontsize=8, padding=2)
        ax.set_ylabel(ylab); ax.set_xlabel("合约")
        ax.set_xticks(x, [TICK_LBL[p] for p in PRODUCTS])
        ax.set_title(ttl); ax.grid(axis="y", alpha=.25); ax.legend(loc="upper right")
    ax1.axhline(0, color="k", lw=.8)
    ax2.axhline(50, color="k", lw=1, ls="--"); ax2.text(3.3, 50.4, "50%（随机）", fontsize=8)
    fig.suptitle("开盘首分钟（09:31）逆势交易(反转）：小样本看似很强，全历史几乎消失",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, 0.01,
             "注：逆势=反向操作9:30→9:31涨跌、持有1分钟至9:32；阈值5跳（=1.0指数点）；毛收益未扣交易成本。",
             ha="center", fontsize=8.5, color="0.35")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(f"{OUT}/fig1_样本量对比.png", dpi=150); plt.close(fig)


# =============================================================== Fig 2
def fig2():
    prods = ["IF0000", "IH0000"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10))
    for i, p in enumerate(prods):
        wf = w_full[w_full.code == p]
        m1f = (wf.c31 / wf.c30 - 1) * 1e4
        m2f = (wf.c32 / wf.c31 - 1) * 1e4
        lim = np.nanpercentile(np.abs(pd.concat([m1f, m2f])), 98)
        for j, (w, lbl) in enumerate([(w_38, "38天（2023年1–2月）"),
                                      (w_full, "全历史（2015–2026）")]):
            ax = axes[i, j]
            wp = w[w.code == p]
            x = ((wp.c31 / wp.c30 - 1) * 1e4).values
            y = ((wp.c32 / wp.c31 - 1) * 1e4).values
            b1, b0, r2, n = fit(x, y)
            ax.scatter(x, y, s=14, alpha=.35, color=COL[p], edgecolor="none")
            xs = np.array([-lim, lim])
            ax.plot(xs, b0 + b1 * xs, color="k", lw=2)
            ax.axhline(0, color="0.6", lw=.6); ax.axvline(0, color="0.6", lw=.6)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_xlabel("第1分钟涨跌 m1（9:30→9:31，bp）")
            ax.set_ylabel("第2分钟收益 m2（9:31→9:32，bp）")
            ax.set_title(f"{NAME[p]}　{lbl}\n回归斜率 β={b1:.2f}　R²={r2:.2f}　样本 n={n}",
                         fontsize=11)
            ax.grid(alpha=.2)
    fig.suptitle("回归斜率从陡峭(小样本)变平坦(全历史)，R²从~0.56塌到~0.01",
                 fontsize=13.5, fontweight="bold")
    fig.text(0.5, 0.005, "斜率 β = 开盘第1分钟涨跌在下一分钟被“吐回”的比例；β=−0.5 表示回吐一半，β≈0 表示几乎是随机游走（无规律）。",
             ha="center", fontsize=8.5, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(f"{OUT}/fig2_反转回归散点.png", dpi=150); plt.close(fig)


# =============================================================== Fig 3
def fig3():
    years = sorted(w_full["yr"].unique())
    M = np.full((4, len(years)), np.nan)
    for i, p in enumerate(PRODUCTS):
        for j, yr in enumerate(years):
            s = momentum_at(w_full[(w_full.code == p) & (w_full.yr == yr)], "09:31", 10)
            if len(s) >= 5:
                M[i, j] = contrarian_row(s)["avg_bp"]
    vmax = 4.0
    fig, ax = plt.subplots(figsize=(12, 4.2))
    im = ax.imshow(M, cmap="RdYlGn_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(years)), years)
    ax.set_yticks(range(4), [NAME[p] for p in PRODUCTS])
    ax.set_xlabel("年份"); ax.set_ylabel("合约")
    for i in range(4):
        for j in range(len(years)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center",
                        fontsize=8, color="black")
    if 2023 in years:
        jx = years.index(2023)
        ax.add_patch(Rectangle((jx - .5, -.5), 1, 4, fill=False, edgecolor="k", lw=2.2))
        ax.text(jx, -.62, "原始样本", ha="center", fontsize=8.5, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, pad=.01)
    cb.set_label("每笔平均收益（bp）　红=正(盈利) / 绿=负(亏损)")
    ax.set_title("开盘逆势收益逐年不稳定：09:31逆势@10跳，每笔平均收益（bp）",
                 fontsize=13, fontweight="bold", pad=22)
    fig.text(0.5, 0.02,
             "正负年份交替——2016–18、2022–23为正，2015、2024–25转负；空白=该年样本不足"
             "（IM 中证1000期货2022年中上市）。",
             ha="center", fontsize=8.5, color="0.35")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(f"{OUT}/fig3_逐年稳定性.png", dpi=150); plt.close(fig)


# =============================================================== Fig 4
def fig4():
    fig, ax = plt.subplots(figsize=(10, 6))
    for p in PRODUCTS:
        y38 = [stat(w_38, p, "09:31", t)["win"] * 100 if stat(w_38, p, "09:31", t) else np.nan for t in THRS]
        yAll = [stat(w_full, p, "09:31", t)["win"] * 100 for t in THRS]
        ax.plot(THRS, y38, "-o", color=COL[p], lw=2, ms=6, label=f"{NAME[p]}（38天）")
        ax.plot(THRS, yAll, "--s", color=COL[p], lw=1.6, ms=5, mfc="white",
                label=f"{NAME[p]}（全历史）")
    ax.axhline(50, color="k", lw=1, ls=":"); ax.text(29, 50.6, "50%（随机）", fontsize=9, ha="right")
    ax.set_xlabel("信号阈值（跳，1跳=0.2指数点）")
    ax.set_ylabel("逆势胜率（%）")
    ax.set_xticks(THRS)
    ax.set_title("“提高阈值→胜率上升”是小样本假象\n"
                 "实线(38天)随阈值飙到90–100%；虚线(全历史)始终贴近55%",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=.3)
    ax.legend(ncol=2, fontsize=8.5, loc="center left", bbox_to_anchor=(1.01, .5))
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_阈值胜率假象.png", dpi=150); plt.close(fig)


# =============================================================== Fig 5
def fig5():
    fig, ax = plt.subplots(figsize=(11, 6))
    for p in PRODUCTS:
        s = momentum_at(w_full[w_full.code == p], "09:31", 5)   # momentum
        c = (-s).sort_index()                                   # contrarian, bp later
        ax.plot(c.index, c.cumsum().values * 1e4, color=COL[p], lw=1.6, label=NAME[p])
    ax.axhline(0, color="k", lw=.8)
    ax.axvspan(pd.Timestamp("2023-01-04"), pd.Timestamp("2023-02-28"),
               color="0.8", alpha=.5)
    ax.text(pd.Timestamp("2023-01-15"), ax.get_ylim()[1] * .95, "原始38天样本",
            fontsize=9, rotation=90, va="top")
    ax.set_xlabel("日期")
    ax.set_ylabel("累计每笔收益（bp，逐笔求和，毛）")
    ax.set_title("开盘逆势策略累计毛收益（09:31@5跳，2015–2026）\n"
                 "整体缓慢上行但单笔仅~1bp（≈买卖价差），且2024–25走平/回撤",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=.3); ax.legend(loc="upper left")
    fig.text(0.5, 0.005, "注：逐笔收益直接相加（非复利、非按交易日归一）；毛收益未扣买卖价差与手续费。",
             ha="center", fontsize=8.5, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(f"{OUT}/fig5_累计收益曲线.png", dpi=150); plt.close(fig)


# =============================================================== Fig 6
def fig6():
    """逆势胜率 vs 阈值，第1分钟(09:31) 与 第2分钟(09:32) 并排，38天 vs 全历史。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    panels = [(axes[0], "09:31", "(a) 第1分钟 09:31（信号=9:30→9:31涨跌）"),
              (axes[1], "09:32", "(b) 第2分钟 09:32（信号=9:31→9:32涨跌）")]
    for ax, minute, ttl in panels:
        for p in PRODUCTS:
            y38 = [stat(w_38, p, minute, t)["win"] * 100 if stat(w_38, p, minute, t) else np.nan for t in THRS]
            yAll = [stat(w_full, p, minute, t)["win"] * 100 if stat(w_full, p, minute, t) else np.nan for t in THRS]
            ax.plot(THRS, y38, "-o", color=COL[p], lw=2, ms=6, label=f"{NAME[p]}（38天）")
            ax.plot(THRS, yAll, "--s", color=COL[p], lw=1.6, ms=5, mfc="white",
                    label=f"{NAME[p]}（全历史）")
        ax.axhline(50, color="k", lw=1, ls=":")
        ax.text(29, 50.6, "50%（随机）", fontsize=8.5, ha="right")
        ax.set_xlabel("信号阈值（跳，1跳=0.2指数点）")
        ax.set_xticks(THRS); ax.set_title(ttl, fontsize=11); ax.grid(alpha=.3)
    axes[0].set_ylabel("逆势胜率（%）")
    axes[1].legend(ncol=2, fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, .5))
    fig.suptitle("逆势胜率 vs 阈值：第1分钟(09:31)尚有信号、第2分钟(09:32)已是噪音",
                 fontsize=13.5, fontweight="bold")
    fig.text(0.5, 0.01,
             "注：胜率=逆势交易盈利的占比（毛口径）。第1分钟实线(38天)随阈值飙升、虚线(全历史)稳定贴近55%；"
             "第2分钟两样本均在50%附近徘徊，无边际。",
             ha="center", fontsize=8.5, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(f"{OUT}/fig6_两分钟阈值胜率.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    print("saved 6 figures ->", OUT)
    for f in sorted(os.listdir(OUT)):
        print("  ", f)
