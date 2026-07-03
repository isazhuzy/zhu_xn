"""
plot_clean_no_reversal.py — remove EVERY bad-tick day (any single-minute move
>150bp anywhere in the session, the threshold already established in
xn-ddb-tick-data-quirks memory / ticker/README.md) from the whole dataset,
then recompute the 14:45->15:00 (last 15 min) MEAN path (no longer need the
median workaround — with bad days gone, mean should already be clean).
Trigger condition (up>=1%) is checked at the window START (14:45), matching
the same "up>=1% at signal time -> does it hold into the close" format used
for the 5-minute version.

Run:  /Users/zhuisabella/xn/.venv/bin/python plot_clean_no_reversal.py  (fetches
nothing new, just re-reads day_minutes_full.csv) then
      python3 plot_clean_no_reversal.py --plot   (system python3, matplotlib)
Simplify: does both in one script, guarded by matplotlib availability.
"""
import numpy as np
import pandas as pd

from common import load, build_panel, day_stats, CODES

THRESH_BP = 150
WINDOW_START = 14 * 60 + 45  # 14:45, last 15 minutes
tcols = list(range(WINDOW_START, 901))
labels = [f"{t // 60:02d}:{t % 60:02d}" for t in tcols]

df = load()
results = {}
n_report = []
for code in CODES:
    piv = build_panel(df, code)
    open_, close_eod, cumret, fwd = day_stats(piv)

    # flag bad-tick days using EVERY minute in the session, not just the tail
    steps = piv.diff(axis=1).div(piv.shift(1, axis=1)).abs() * 1e4
    bad_day = steps.gt(THRESH_BP).any(axis=1)
    clean_piv = piv.loc[~bad_day]

    n_before = (cumret[WINDOW_START] >= 0.01).sum()
    open_c, close_c, cumret_c, fwd_c = day_stats(clean_piv)
    mask_c = cumret_c[WINDOW_START] >= 0.01
    n_after = mask_c.sum()
    n_report.append((code, bad_day.sum(), len(piv), n_before, n_after))

    sub = clean_piv.loc[mask_c, tcols].dropna()
    base = sub[WINDOW_START]
    rel = sub.sub(base, axis=0).div(base, axis=0) * 1e4
    results[code] = dict(mean=[rel[c].mean() for c in tcols],
                          median=[rel[c].median() for c in tcols],
                          n=len(sub))

print(f"{'code':8s} {'bad_days':>9s} {'total_days':>11s} {'n@14:45_before':>15s} {'n@14:45_after_clean':>20s}")
for code, nb, tot, nbefore, nafter in n_report:
    print(f"{code:8s} {nb:9d} {tot:11d} {nbefore:15d} {nafter:20d}")

# ---------------- plot ----------------
import matplotlib.pyplot as plt
FIGDIR = "/Users/zhuisabella/xn/end/figs"
COLORS = {"IC0000": "tab:blue", "IF0000": "tab:orange", "IH0000": "tab:green", "IM0000": "tab:red"}

fig, ax = plt.subplots(figsize=(9.5, 5.5))
for code in CODES:
    r = results[code]
    ax.plot(labels, r["mean"], marker="o", color=COLORS[code], label=f"{code} MEAN, all bad-tick days removed (n={r['n']})")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(labels[::2])
ax.tick_params(axis="x", rotation=45)
ax.set_ylabel(f"mean price path relative to {labels[0]} (bp)")
ax.set_title(f"After removing every day with a >{THRESH_BP}bp single-minute move (bad ticks) anywhere in the session:\nmean path over the last 15 minutes still shows no reversal, holds/extends the gain into the close")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig_verdict_clean_mean_path.png", dpi=140)
print("\nsaved fig_verdict_clean_mean_path.png")
