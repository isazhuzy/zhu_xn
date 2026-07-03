"""
plot_eod_reversal.py — figures for the end-of-day (>=1% up-day) reversal scan.
Run on system python3 (matplotlib not in .venv):  python3 plot_eod_reversal.py
"""
import pandas as pd
import matplotlib.pyplot as plt

SCAN = "/Users/zhuisabella/xn/end/scan_by_minute.csv"
YEAR = "/Users/zhuisabella/xn/end/scan_by_year.csv"
FIGDIR = "/Users/zhuisabella/xn/end/figs"
CODES = ["IC0000", "IF0000", "IH0000", "IM0000"]

import os
os.makedirs(FIGDIR, exist_ok=True)

scan = pd.read_csv(SCAN)
year = pd.read_csv(YEAR)

# fig1: t-stat by minute-of-day, one line per contract
fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
for code in CODES:
    s = scan[scan.code == code].sort_values("tod")
    axes[0].plot(s["tod"], s["t"], label=code, marker=".", ms=3)
    axes[1].plot(s["tod"], s["mean_bp"], label=code, marker=".", ms=3)
axes[0].axhline(0, color="k", lw=0.8)
axes[0].axhline(-2, color="r", lw=0.8, ls="--", label="t=-2 (sig.)")
axes[0].axhline(2, color="r", lw=0.8, ls="--")
axes[0].set_ylabel("t-stat (day-clustered)")
axes[0].set_title("Forward return (t -> 15:00 close) conditional on cumret(t) >= 1%, full history 2015-2026")
axes[0].legend(ncol=5, fontsize=8)
axes[1].axhline(0, color="k", lw=0.8)
axes[1].set_ylabel("mean fwd return (bp)")
xt = [t for t in sorted(scan.tod.unique()) if t % 15 == 0]
axes[1].set_xticks(xt)
axes[1].set_xticklabels([f"{t//60:02d}:{t%60:02d}" for t in xt], rotation=45)
axes[1].set_xlabel("minute of day (signal time t)")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig01_scan_by_minute.png", dpi=130)
print("saved fig01_scan_by_minute.png")

# fig2: year-by-year mean_bp at each contract's "best" (most negative t) minute
fig, ax = plt.subplots(figsize=(10, 5))
codes_present = year["code"].unique()
width = 0.2
years = sorted(year["year"].unique())
for i, code in enumerate(codes_present):
    s = year[year.code == code].set_index("year").reindex(years)
    ax.bar([y + i * width for y in years], s["mean_bp"], width=width, label=code)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("mean fwd return at contract's 'best' minute (bp)")
ax.set_title("Year-by-year: the 'best' minute's edge is dominated by 2015, flips sign otherwise")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig02_year_breakdown.png", dpi=130)
print("saved fig02_year_breakdown.png")
