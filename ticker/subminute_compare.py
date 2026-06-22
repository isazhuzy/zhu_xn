"""
subminute_compare.py — focused comparison of entry ticks N = 6, 8, 10, 12.

Fade the first-N-tick move of minute 1 (up->short, down->long), enter at tick N,
hold to 09:31 close and to 09:32 close. Reports per contract AND an equal-weight
4-contract basket aggregated per day (the honest aggregate: the 4 index futures
are highly correlated at the open, so pooling all trades would overstate t).

  per trade:  pos=-sign(mid[N]-mid[1]);  pnl=pos*(exit/entry-1)*1e4 (bp, gross)
  basket(day) = mean over the contracts that traded that day
  t = mean / (std/sqrt(n))

Run:  /usr/local/bin/python3.14 subminute_compare.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from subminute_demo import build_records, SRC, PRODUCTS, NAME, COL

N_LIST = [6, 8, 10, 12]
OUT = "/Users/zhuisabella/xn/ticker/figs_conclusion"


def st(a):
    a = np.asarray(a, float); a = a[~np.isnan(a)]; n = len(a)
    if n == 0:
        return None
    se = a.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return dict(n=n, win=round((a > 0).mean(), 3), mean_bp=round(a.mean(), 2),
                se_bp=round(se, 2), t=round(a.mean() / se, 2) if se else np.nan)


def per_trade(recs):
    """long frame: code, N, day, pnl1 (->09:31), pnl2 (->09:32)."""
    rows = []
    for (code, day), r in recs.items():
        m1 = r["m1"]
        for N in N_LIST:
            if len(m1) < N:
                continue
            entry = m1[N - 1]; sig = m1[N - 1] - m1[0]
            if sig == 0:
                continue
            pos = -np.sign(sig)
            rows.append({"code": code, "N": N, "day": day,
                         "pnl1": pos * (r["ex1"] / entry - 1) * 1e4 if N < len(m1) else np.nan,
                         "pnl2": pos * (r["ex2"] / entry - 1) * 1e4})
    return pd.DataFrame(rows)


def table(L):
    rows = []
    for ex, col in [("09:31", "pnl1"), ("09:32", "pnl2")]:
        for code in PRODUCTS:
            for N in N_LIST:
                s = st(L[(L.code == code) & (L.N == N)][col])
                if s:
                    rows.append({"exit": ex, "group": code, "N": N, **s})
        for N in N_LIST:                                  # equal-weight basket, per day
            basket = L[L.N == N].groupby("day")[col].mean()
            s = st(basket)
            if s:
                rows.append({"exit": ex, "group": "组合(4合约等权)", "N": N, **s})
    return pd.DataFrame(rows)


def plot(L, tab):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for j, (ex, col) in enumerate([("09:31", "pnl1"), ("09:32", "pnl2")]):
        for p in PRODUCTS:
            s = tab[(tab.exit == ex) & (tab.group == p)].sort_values("N")
            axes[0, j].plot(s["N"], s["mean_bp"], "-o", color=COL[p], ms=5, lw=1.2,
                            alpha=.7, label=NAME[p])
            axes[1, j].plot(s["N"], s["win"] * 100, "-o", color=COL[p], ms=5, lw=1.2,
                            alpha=.7, label=NAME[p])
        b = tab[(tab.exit == ex) & (tab.group == "组合(4合约等权)")].sort_values("N")
        axes[0, j].errorbar(b["N"], b["mean_bp"], yerr=b["se_bp"], fmt="-s", color="k",
                            lw=2.4, ms=7, capsize=4, label="组合(等权,±1SE)", zorder=5)
        for _, r in b.iterrows():                          # t-stat on the basket
            axes[0, j].annotate(f"t={r['t']:.1f}", (r["N"], r["mean_bp"]),
                                textcoords="offset points", xytext=(0, 9),
                                ha="center", fontsize=8, fontweight="bold")
        axes[1, j].plot(b["N"], b["win"] * 100, "-s", color="k", lw=2.4, ms=7, label="组合(等权)")
        axes[0, j].axhline(0, color="0.5", lw=.8); axes[1, j].axhline(50, color="0.5", lw=1, ls=":")
        axes[0, j].set_title(f"持有到 {ex} 收盘", fontsize=12)
        for ax in (axes[0, j], axes[1, j]):
            ax.grid(alpha=.3); ax.set_xticks(N_LIST)
        axes[1, j].set_xlabel("进场 tick 序号 N（约 N/2 秒）")
    axes[0, 0].set_ylabel("每笔平均收益（bp，毛）")
    axes[1, 0].set_ylabel("逆势胜率（%）")
    axes[0, 1].legend(fontsize=8, loc="best")
    fig.suptitle("图10　进场tick 6/8/10/12 对比（38天）：反转择时，前N个tick定方向",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, 0.01, "黑色=4合约等权组合(按日聚合,±1SE,标注t值)；t≈0 即与噪音无异。"
             "信号仅几个tick、常在买卖价差内，方向噪音大。毛收益。",
             ha="center", fontsize=8.5, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(f"{OUT}/fig10_进场tick对比.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    df = pd.read_csv(SRC, dtype={"code": "string"}, parse_dates=["m_nDatetime"])
    L = per_trade(build_records(df))
    tab = table(L)
    tab.to_csv("/Users/zhuisabella/xn/ticker/open_breakdown/subminute_compare.csv", index=False)
    pd.set_option("display.width", 200, "display.max_rows", 200)
    for ex in ["09:31", "09:32"]:
        print(f"\n===== 进场tick 6/8/10/12 | 持有到{ex}收盘 | 反转 | 38天 =====")
        print(tab[tab.exit == ex].drop(columns="exit").to_string(index=False))
    plot(L, tab)
    print("\nsaved fig10_进场tick对比.png ->", OUT)
