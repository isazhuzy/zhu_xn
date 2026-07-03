"""
plot_demonstrate_no_reversal.py — one figure that (a) shows the actual raw-tick
glitch responsible for the "reversal-looking" mean dip/spike at 14:58-15:00,
and (b) contrasts the distorted MEAN path against the robust MEDIAN path to
prove there is no real post-14:55 reversal. Run on system python3.
"""
import pandas as pd
import matplotlib.pyplot as plt

from common import load, build_panel, day_stats, CODES

FIGDIR = "/Users/zhuisabella/xn/end/figs"
df = load()
tcols = [895, 896, 897, 898, 899, 900]
labels = ["14:55", "14:56", "14:57", "14:58", "14:59", "15:00"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# ---- left panel: the actual glitch days, raw price, 14:20-15:00 ----
ax = axes[0]
piv_ic = build_panel(df, "IC0000")
row_ic = piv_ic.loc["2015-08-28", 890:900]
ax2 = ax.twinx()
ax.plot(row_ic.index, row_ic.values, color="tab:blue", marker="o", ms=4, label="IC0000 2015-08-28 (left axis)")
piv_ih = build_panel(df, "IH0000")
row_ih = piv_ih.loc["2015-07-09", 890:900]
ax2.plot(row_ih.index, row_ih.values, color="tab:green", marker="s", ms=4, label="IH0000 2015-07-09 (right axis)")
ax.set_xticks(row_ic.index[::2])
ax.set_xticklabels([f"{t//60:02d}:{t%60:02d}" for t in row_ic.index[::2]], rotation=45)
ax.set_ylabel("IC0000 price", color="tab:blue")
ax2.set_ylabel("IH0000 price", color="tab:green")
ax.set_title("The actual raw ticks: price briefly HALVES then snaps back\n(a data glitch, not a real trade)")
lines1, l1 = ax.get_legend_handles_labels()
lines2, l2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, l1 + l2, fontsize=8, loc="lower left")

# ---- right panel: mean (distorted) vs median (robust) path, these 2 contracts ----
ax = axes[1]
for code, color in [("IC0000", "tab:blue"), ("IH0000", "tab:green")]:
    piv = build_panel(df, code)
    open_, close_eod, cumret, fwd = day_stats(piv)
    mask = cumret[895] >= 0.01
    sub = piv.loc[mask, tcols].dropna()
    base = sub[895]
    rel = sub.sub(base, axis=0).div(base, axis=0) * 1e4
    mean_path = [rel[c].mean() for c in tcols]
    med_path = [rel[c].median() for c in tcols]
    ax.plot(labels, mean_path, color=color, ls="--", marker="x", label=f"{code} MEAN (glitch-distorted)")
    ax.plot(labels, med_path, color=color, ls="-", marker="o", label=f"{code} MEDIAN (robust)")
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("path relative to 14:55 (bp)")
ax.set_title("Mean says 'reversal' (fooled by 1 bad day out of ~300);\nmedian says 'holds the gain'")
ax.legend(fontsize=7.5)

fig.suptitle("Why the 14:55 t-stat 'drop' is NOT a real reversal", fontsize=13)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig_verdict_glitch_proof.png", dpi=140)
print("saved fig_verdict_glitch_proof.png")
