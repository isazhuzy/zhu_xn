"""
plot_continuation_by_year.py — precise year-by-year view of the closing
continuation effect (D in summary_verdict), to visually confirm it is NOT a
2015 artifact (unlike every reversal-hunt filter, A/B/C in the same table).
Run on system python3.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import load, build_panel, day_stats, tod_to_hm, CODES

FIGDIR = "/Users/zhuisabella/xn/end/figs"
COLORS = {"IC0000": "tab:blue", "IF0000": "tab:orange", "IH0000": "tab:green", "IM0000": "tab:red"}
df = load()
t_ref = 14 * 60 + 55

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, code in zip(axes.flat, CODES):
    piv = build_panel(df, code)
    open_, close_eod, cumret, fwd = day_stats(piv)
    years = pd.Series(cumret.index.year, index=cumret.index)
    mask = cumret[t_ref] >= 0.01
    rows = []
    for yr in sorted(years.unique()):
        m = mask & (years == yr)
        y = (fwd[t_ref] * 1e4)[m].dropna()
        if len(y) >= 3:
            rows.append(dict(year=yr, mean_bp=y.mean(), n=len(y)))
    tab = pd.DataFrame(rows)
    if len(tab) == 0:
        continue
    colors = ["tab:red" if yr == 2015 else COLORS[code] for yr in tab["year"]]
    bars = ax.bar(tab["year"].astype(str), tab["mean_bp"], color=colors)
    for b, n in zip(bars, tab["n"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"n={n}",
                fontsize=6, ha="center", va="bottom" if b.get_height() >= 0 else "top")
    ax.axhline(0, color="k", lw=0.8)
    pos_frac = (tab["mean_bp"] > 0).mean()
    ax.set_title(f"{code}: {int(pos_frac*100)}% of years positive (2015 = red)")
    ax.set_ylabel("mean fwd return, 14:55->close (bp)")
    ax.tick_params(axis="x", rotation=45)
fig.suptitle("Closing-minutes continuation @14:55 is NOT a 2015 artifact\n(contrast with fig_pr03, where the reversal-hunt effect WAS)")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig_verdict_continuation_by_year.png", dpi=130)
print("saved fig_verdict_continuation_by_year.png")
