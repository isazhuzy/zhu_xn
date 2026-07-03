"""
plot_clean_path_ratio.py — same format as plot_clean_no_reversal.py (bad-tick
days removed, last-15-min mean path relative to window start), but the
up>=1%-from-open trigger at 14:45 is split by 路程/位移比 (path/displacement
ratio, common.py::path_ratio) into "smooth" (ratio<=median, efficient trend)
vs "noisy" (ratio>median, choppy path) subsamples — same split used in
filter_path_ratio.py. Tests whether a choppy vs clean run-up to 14:45 changes
what happens in the final 15 minutes. Run on system python3.
"""
import pandas as pd
import matplotlib.pyplot as plt

from common import load, build_panel, day_stats, path_ratio, CODES

THRESH_BP = 150
WINDOW_START = 14 * 60 + 45  # 14:45, last 15 minutes
tcols = list(range(WINDOW_START, 901))
labels = [f"{t // 60:02d}:{t % 60:02d}" for t in tcols]
FIGDIR = "/Users/zhuisabella/xn/end/figs"
COLORS = {"IC0000": "tab:blue", "IF0000": "tab:orange", "IH0000": "tab:green", "IM0000": "tab:red"}

df = load()
results = {}
n_report = []
for code in CODES:
    piv = build_panel(df, code)
    open_, close_eod, cumret, fwd = day_stats(piv)
    ratio = path_ratio(piv)

    # same bad-tick-day removal as plot_clean_no_reversal.py
    steps = piv.diff(axis=1).div(piv.shift(1, axis=1)).abs() * 1e4
    bad_day = steps.gt(THRESH_BP).any(axis=1)
    clean_piv = piv.loc[~bad_day]
    clean_ratio = ratio.loc[~bad_day]
    open_c, close_c, cumret_c, fwd_c = day_stats(clean_piv)

    base_mask = cumret_c[WINDOW_START] >= 0.01
    med = clean_ratio.loc[base_mask, WINDOW_START].median()
    smooth_mask = base_mask & (clean_ratio[WINDOW_START] <= med)
    noisy_mask = base_mask & (clean_ratio[WINDOW_START] > med)

    for label, mask in [("smooth", smooth_mask), ("noisy", noisy_mask)]:
        sub = clean_piv.loc[mask, tcols].dropna()
        base = sub[WINDOW_START]
        rel = sub.sub(base, axis=0).div(base, axis=0) * 1e4
        results[(code, label)] = dict(path=[rel[c].mean() for c in tcols], n=len(sub))
        n_report.append((code, label, len(sub)))

print(f"{'code':8s} {'group':7s} {'n':>5s}  (median ratio@14:45 split, bad-tick days removed)")
for code, label, n in n_report:
    print(f"{code:8s} {label:7s} {n:5d}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
for ax, code in zip(axes.flat, CODES):
    for label, ls, mk in [("smooth", "-", "o"), ("noisy", "--", "x")]:
        r = results[(code, label)]
        ax.plot(labels, r["path"], color=COLORS[code], ls=ls, marker=mk,
                 label=f"{label} (n={r['n']})")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(labels)  # every minute, not every 3rd -- so the exact minute of any dip is readable
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.set_xlabel("signal minute (time of day, HH:MM)", fontsize=9)
    ax.set_ylabel("forward price change relative to price @14:45 (bp)", fontsize=9)
    ax.set_title(code, fontsize=12)
    ax.grid(True, which="major", axis="both", color="gray", alpha=0.3, linestyle="-", linewidth=0.6)
    ax.minorticks_on()
    ax.grid(True, which="minor", axis="y", color="gray", alpha=0.15, linestyle=":", linewidth=0.4)
    ax.legend(fontsize=8)
fig.suptitle("Last 15 min, up>=1% from open @14:45, split by path/displacement ratio at 14:45\n(solid=smooth/efficient trend, dashed=noisy/choppy path; bad-tick days removed)")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig_verdict_clean_pathratio_path.png", dpi=140)
print("\nsaved fig_verdict_clean_pathratio_path.png")
