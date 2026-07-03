"""
plot_last5min_path.py — does price reverse AFTER 14:55, or just keep drifting?
Uses the MEDIAN path (not mean) since a couple of single bad-tick days in 2015
(one -4997bp glitch in IC0000, one +9986bp glitch in IH0000) distort the mean
badly at 14:58-15:00. Run on system python3.
"""
import pandas as pd
import matplotlib.pyplot as plt

from common import load, build_panel, day_stats, CODES

FIGDIR = "/Users/zhuisabella/xn/end/figs"
COLORS = {"IC0000": "tab:blue", "IF0000": "tab:orange", "IH0000": "tab:green", "IM0000": "tab:red"}
df = load()
tcols = [895, 896, 897, 898, 899, 900]
labels = ["14:55", "14:56", "14:57", "14:58", "14:59", "15:00"]

fig, ax = plt.subplots(figsize=(9, 5.5))
for code in CODES:
    piv = build_panel(df, code)
    open_, close_eod, cumret, fwd = day_stats(piv)
    mask = cumret[895] >= 0.01
    sub = piv.loc[mask, tcols].dropna()
    base = sub[895]
    rel = sub.sub(base, axis=0).div(base, axis=0) * 1e4
    med = [rel[c].median() for c in tcols]
    ax.plot(labels, med, marker="o", color=COLORS[code], label=f"{code} (n={len(sub)})")

ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("median price path relative to 14:55 (bp)")
ax.set_title("After the 14:55 jump: does price give it back, or keep drifting?\n(median path — robust to a couple of 2015 bad-tick glitches)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig_verdict_last5min_median_path.png", dpi=130)
print("saved fig_verdict_last5min_median_path.png")
