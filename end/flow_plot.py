"""
flow_plot.py — mirror plot_clean_path_ratio.py's 2x2 last-15-min mean-path layout,
but split up>=1% days at 14:45 by trailing SELL-PRESSURE imbalance (flow_analysis.py)
instead of the path/displacement ratio.
solid = low sell-pressure (net buying), dashed = high sell-pressure (net selling /
profit-taking). If profit-taking drives reversal, the dashed line should end lower.
Run on system python3.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import day_stats, CODES
from flow_analysis import load_flow, panels, trailing_sum, W

WINDOW_START = 14 * 60 + 45
tcols = list(range(WINDOW_START, 901))
labels = [f"{t // 60:02d}:{t % 60:02d}" for t in tcols]
FIGDIR = "/Users/zhuisabella/xn/end/figs"
COLORS = {"IC0000": "tab:blue", "IF0000": "tab:orange", "IH0000": "tab:green", "IM0000": "tab:red"}
THRESH_BP = 150

df = load_flow()
# resolve semantics once (same rule as flow_analysis)
cn = 0.0
for code in CODES:
    close, ab, aa = panels(df, code)
    ret, sig = close.diff(axis=1), ab - aa
    m = (ret.notna() & sig.notna()).to_numpy()
    if m.sum() > 1000:
        cn += np.corrcoef(ret.to_numpy()[m], sig.to_numpy()[m])[0, 1]
actbid_is_buy = cn > 0

results, n_report = {}, []
for code in CODES:
    close, ab, aa = panels(df, code)
    buy = ab if actbid_is_buy else aa
    sell = aa if actbid_is_buy else ab
    # bad-tick-day removal, same 150bp rule as the clean_* figs
    steps = close.diff(axis=1).div(close.shift(1, axis=1)).abs() * 1e4
    bad = steps.gt(THRESH_BP).any(axis=1)
    close, buy, sell = close.loc[~bad], buy.loc[~bad], sell.loc[~bad]
    _, _, cumret, _ = day_stats(close)
    buyW, sellW = trailing_sum(buy, W), trailing_sum(sell, W)
    imb = (sellW - buyW) / (sellW + buyW)

    up = cumret[WINDOW_START] >= 0.01
    med = imb.loc[up, WINDOW_START].median()
    lo_mask = up & (imb[WINDOW_START] <= med)   # low sell-pressure
    hi_mask = up & (imb[WINDOW_START] > med)    # high sell-pressure (profit-taking)
    for label, mask in [("low sell-pressure", lo_mask), ("high sell-pressure", hi_mask)]:
        sub = close.loc[mask, tcols].dropna()
        base = sub[WINDOW_START]
        rel = sub.sub(base, axis=0).div(base, axis=0) * 1e4
        results[(code, label)] = dict(path=[rel[c].mean() for c in tcols], n=len(sub))
        n_report.append((code, label, len(sub)))

print(f"semantics: actbid is {'BUYER' if actbid_is_buy else 'SELLER'}-initiated")
for code, label, n in n_report:
    print(f"{code:8s} {label:20s} n={n}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
for ax, code in zip(axes.flat, CODES):
    for label, ls, mk in [("low sell-pressure", "-", "o"), ("high sell-pressure", "--", "x")]:
        r = results[(code, label)]
        ax.plot(labels, r["path"], color=COLORS[code], ls=ls, marker=mk,
                label=f"{label} (n={r['n']})")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(labels); ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.set_xlabel("signal minute (HH:MM)", fontsize=9)
    ax.set_ylabel("mean fwd price change vs 14:45 (bp)", fontsize=9)
    ax.set_title(code, fontsize=12)
    ax.grid(True, which="major", color="gray", alpha=0.3, lw=0.6)
    ax.legend(fontsize=8)
fig.suptitle(f"Last 15 min, up>=1% from open @14:45, split by trailing-{W}min aggressor "
             f"SELL-PRESSURE at 14:45\n(solid=net buying, dashed=net selling/profit-taking; "
             f"bad-tick days removed)")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig_flow_sellpressure_path.png", dpi=140)
print("\nsaved fig_flow_sellpressure_path.png")
