"""
plot_summary_verdict.py — the precise, consolidated verdict figure: for every
effect x contract tested in xn/end, plot full-history t-stat vs ex-2015
t-stat side by side (forest-plot style), so it's visually unambiguous which
effects are 2015-driven artifacts and which are real. Run on system python3.
"""
import pandas as pd
import matplotlib.pyplot as plt

FIGDIR = "/Users/zhuisabella/xn/end/figs"
tab = pd.read_csv("/Users/zhuisabella/xn/end/summary_verdict.csv")

EFFECT_LABELS = {
    "A. plain reversal-hunt (best min)": "A. plain '>=1% from open' reversal hunt",
    "B. path-ratio 'noisy' subsample": "B. path/displacement-ratio 'noisy' subsample",
    "C. calm-day (real-time range)": "C. calm-day filter (real-time-computable)",
    "D. closing continuation @14:55": "D. closing-minutes CONTINUATION @14:55",
}
tab["effect_label"] = tab["effect"].map(EFFECT_LABELS)
tab["row"] = tab["effect_label"] + " | " + tab["code"]
tab = tab.sort_values(["effect", "code"], ascending=[False, True]).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10, 9))
ypos = range(len(tab))
for y, (_, r) in zip(ypos, tab.iterrows()):
    ax.plot([r["full_t"], r["ex15_t"]], [y, y], color="lightgray", lw=1.5, zorder=1)
    ax.scatter(r["full_t"], y, color="tab:blue", s=55, zorder=2,
               label="full history (2015-2026)" if y == len(tab) - 1 else None)
    ax.scatter(r["ex15_t"], y, color="tab:red", s=55, marker="D", zorder=2,
               label="ex-2015" if y == len(tab) - 1 else None)

ax.axvline(0, color="k", lw=0.8)
ax.axvline(2, color="gray", ls="--", lw=0.8)
ax.axvline(-2, color="gray", ls="--", lw=0.8)
ax.text(2.05, len(tab) - 0.5, "t=+2\n(sig. continuation)", fontsize=7, va="top", color="gray")
ax.text(-2.05, len(tab) - 0.5, "t=-2\n(sig. reversal)", fontsize=7, va="top", ha="right", color="gray")

ax.set_yticks(list(ypos))
ax.set_yticklabels(tab["row"])
ax.set_xlabel("t-stat of forward return, signal-minute -> 15:00 close")
ax.set_title("Every reversal/continuation effect tested in xn/end:\nfull-history t-stat vs. ex-2015 t-stat (gray line = how much 2015 moved it)")
ax.legend(loc="lower right", fontsize=9)
ax.grid(axis="y", alpha=0.15)

# shade effect groups
boundaries = tab.reset_index().groupby("effect_label")["index"].agg(["min", "max"])
for i, (_, b) in enumerate(boundaries.iterrows()):
    if i % 2 == 0:
        ax.axhspan(b["min"] - 0.5, b["max"] + 0.5, color="gray", alpha=0.05)

plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig_verdict_forest.png", dpi=140)
print("saved fig_verdict_forest.png")
