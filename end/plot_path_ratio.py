"""
plot_path_ratio.py — figures for the 路程/位移比 (path-length/displacement)
end-of-day reversal exploration. Run on system python3 (matplotlib not in
the DolphinDB .venv):  python3 plot_path_ratio.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import load, build_panel, day_stats, path_ratio, scan_minute, tod_to_hm, CODES

FIGDIR = "/Users/zhuisabella/xn/end/figs"
os.makedirs(FIGDIR, exist_ok=True)
AFTERNOON = list(range(780, 900))
THRESH = 0.01
COLORS = {"IC0000": "tab:blue", "IF0000": "tab:orange", "IH0000": "tab:green", "IM0000": "tab:red"}

df = load()
panels, stats_, ratios, years = {}, {}, {}, {}
for code in CODES:
    piv = build_panel(df, code)
    open_, close_eod, cumret, fwd = day_stats(piv)
    panels[code] = piv
    stats_[code] = (open_, close_eod, cumret, fwd)
    ratios[code] = path_ratio(piv)
    years[code] = pd.Series(cumret.index.year, index=cumret.index)

# ---------- fig1: ratio distribution among up>=1% observations ----------
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, code in zip(axes.flat, CODES):
    _, _, cumret, _ = stats_[code]
    ratio = ratios[code]
    base = cumret >= THRESH
    vals = pd.concat([ratio[t][base[t]] for t in AFTERNOON if t in ratio.columns]).dropna()
    vals = vals[vals < vals.quantile(0.99)]
    med = vals.median()
    ax.hist(vals, bins=60, color=COLORS[code], alpha=0.8)
    ax.axvline(med, color="k", ls="--", lw=1.2, label=f"median={med:.2f}")
    ax.axvline(1.0, color="gray", ls=":", lw=1, label="ratio=1 (straight line)")
    ax.set_title(code)
    ax.set_xlabel("path / displacement ratio")
    ax.legend(fontsize=8)
fig.suptitle("Path/displacement ratio distribution, conditional on cumret(t)>=1% (afternoon obs, all years)")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig_pr01_ratio_distribution.png", dpi=130)
print("saved fig_pr01_ratio_distribution.png")

# ---------- fig2: t-stat scan, smooth vs noisy, per contract ----------
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
for ax, code in zip(axes.flat, CODES):
    _, _, cumret, fwd = stats_[code]
    ratio = ratios[code]
    base = cumret >= THRESH
    allvals = pd.concat([ratio[t][base[t]] for t in AFTERNOON if t in ratio.columns])
    med = allvals.median()
    smooth = base & (ratio <= med)
    noisy = base & (ratio > med)
    s_smooth = scan_minute(smooth, fwd, AFTERNOON)
    s_noisy = scan_minute(noisy, fwd, AFTERNOON)
    if len(s_smooth):
        ax.plot(s_smooth["tod"], s_smooth["t"], color="tab:blue", marker=".", ms=3, label="smooth (ratio<=median)")
    if len(s_noisy):
        ax.plot(s_noisy["tod"], s_noisy["t"], color="tab:red", marker=".", ms=3, label="noisy (ratio>median)")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(-2, color="gray", ls="--", lw=0.8)
    ax.axhline(2, color="gray", ls="--", lw=0.8)
    ax.set_title(code)
    ax.legend(fontsize=8)
    ax.set_ylabel("t-stat")
xt = [t for t in AFTERNOON if t % 15 == 0]
for ax in axes[-1]:
    ax.set_xticks(xt)
    ax.set_xticklabels([tod_to_hm(t) for t in xt], rotation=45)
fig.suptitle("Forward-return t-stat by signal minute: smooth vs noisy path to the >=1% gain")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig_pr02_tstat_smooth_vs_noisy.png", dpi=130)
print("saved fig_pr02_tstat_smooth_vs_noisy.png")

# ---------- fig3: year-by-year mean_bp at the noisy group's best minute ----------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, code in zip(axes.flat, CODES):
    _, _, cumret, fwd = stats_[code]
    ratio = ratios[code]
    base = cumret >= THRESH
    allvals = pd.concat([ratio[t][base[t]] for t in AFTERNOON if t in ratio.columns])
    med = allvals.median()
    noisy = base & (ratio > med)
    s_noisy = scan_minute(noisy, fwd, AFTERNOON)
    if len(s_noisy) == 0:
        continue
    t_best = int(s_noisy.sort_values("t").iloc[0]["tod"])
    yy = years[code]
    rows = []
    for yr in sorted(yy.unique()):
        m = noisy[t_best] & (yy == yr)
        y = (fwd[t_best] * 1e4)[m].dropna()
        if len(y) >= 3:
            rows.append(dict(year=yr, mean_bp=y.mean(), n=len(y)))
    tab = pd.DataFrame(rows)
    if len(tab) == 0:
        continue
    colors = ["tab:red" if y == 2015 else "tab:blue" for y in tab["year"]]
    ax.bar(tab["year"].astype(str), tab["mean_bp"], color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"{code}: noisy-group best minute = {tod_to_hm(t_best)}")
    ax.set_ylabel("mean fwd return (bp)")
    ax.tick_params(axis="x", rotation=45)
fig.suptitle("Year-by-year: is the 'noisy path reverts more' effect just 2015 again? (2015 = red)")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig_pr03_noisy_by_year.png", dpi=130)
print("saved fig_pr03_noisy_by_year.png")

# ---------- fig4: continuous corr(ratio(t), fwd(t)) across the whole afternoon ----------
fig, ax = plt.subplots(figsize=(10, 5))
for code in CODES:
    _, _, cumret, fwd = stats_[code]
    ratio = ratios[code]
    base = cumret >= THRESH
    corrs = []
    for t in AFTERNOON:
        if t not in ratio.columns:
            continue
        m = base[t]
        x, y = ratio[t][m], (fwd[t] * 1e4)[m]
        ok = x.notna() & y.notna()
        corrs.append((t, x[ok].corr(y[ok]) if ok.sum() >= 20 else np.nan))
    ct = pd.DataFrame(corrs, columns=["tod", "corr"])
    ax.plot(ct["tod"], ct["corr"], label=code, color=COLORS[code], marker=".", ms=3)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(xt)
ax.set_xticklabels([tod_to_hm(t) for t in xt], rotation=45)
ax.set_ylabel("corr(ratio(t), fwd(t))")
ax.set_title("Does a higher path/displacement ratio predict a bigger reversal?\n(negative = yes; note decay away from the 13:00 reopen)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig_pr04_continuous_corr.png", dpi=130)
print("saved fig_pr04_continuous_corr.png")

# ---------- fig5: illustrative smooth-day vs noisy-day price paths ----------
# (picked from a sane day-range band to avoid bad-tick / contract-roll outliers
# — see ddb-tick-data-quirks memory: a few known bad ticks fake huge jumps —
# and from the 25th/75th percentile of ratio rather than the raw extremes, so
# the examples are representative rather than cherry-picked freak days)
code = "IF0000"
piv = panels[code]
open_, close_eod, cumret, fwd = stats_[code]
ratio = ratios[code]
t_ref = 13 * 60  # 13:00
sane_day = (piv.max(axis=1) / piv.min(axis=1) - 1) < 0.05  # exclude bad-tick/roll-jump days
base_day = (cumret[t_ref] >= THRESH) & ratio[t_ref].notna() & sane_day
cand = ratio[t_ref][base_day].sort_values()
smooth_day = cand.index[int(len(cand) * 0.10)]   # near-low ratio = straightish rally
noisy_day = cand.index[int(len(cand) * 0.90)]    # near-high ratio = choppy rally

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
for ax, day, label in [(axes[0], smooth_day, "smooth"), (axes[1], noisy_day, "noisy")]:
    row = piv.loc[day].dropna()
    tods = row.index
    rel = (row / row.loc[570] - 1) * 1e4
    ax.plot(tods, rel, color=COLORS[code])
    ax.axvline(t_ref, color="gray", ls="--", lw=1, label="13:00 signal time")
    ax.axhline(0, color="k", lw=0.6)
    r = ratio.loc[day, t_ref]
    ax.set_title(f"{label} example: {day.date()} (ratio@13:00={r:.2f})")
    ax.set_xlabel("minute of day")
    ax.set_ylabel("return from open (bp)")
    xt2 = [t for t in tods if t % 30 == 0]
    ax.set_xticks(xt2)
    ax.set_xticklabels([tod_to_hm(t) for t in xt2], rotation=45)
    ax.legend(fontsize=8)
fig.suptitle(f"{code}: what 'smooth' vs 'noisy' path to a >=1% gain by 13:00 actually looks like")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig_pr05_example_days.png", dpi=130)
print("saved fig_pr05_example_days.png")
