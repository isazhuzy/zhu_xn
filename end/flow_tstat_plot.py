"""
flow_tstat_plot.py — verdict figure for the sell-pressure (profit-taking) test.
Reads flow_summary.csv and plots, across the afternoon 13:00-14:59:
  top row  : t-stat of (high sell-pressure - low sell-pressure) mean fwd->close
             (negative = profit-taking hypothesis: more selling -> weaker close)
  bottom   : continuous corr(sell-pressure imbalance, fwd->close) within up>=1% days
solid = full history, dashed = ex-2015. Shaded band = |t|<2 (not significant).
Run on system python3.
"""
import pandas as pd
import matplotlib.pyplot as plt

o = pd.read_csv("/Users/zhuisabella/xn/end/flow_summary.csv")
CODES = ["IC0000", "IF0000", "IH0000", "IM0000"]
COLORS = {"IC0000": "tab:blue", "IF0000": "tab:orange", "IH0000": "tab:green", "IM0000": "tab:red"}
FIGDIR = "/Users/zhuisabella/xn/end/figs"

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
for code in CODES:
    s = o[o.code == code].sort_values("tod")
    x = s["hm"]
    ax1.plot(x, s["t_hi_minus_lo"], color=COLORS[code], lw=1.6, label=f"{code} full")
    ax1.plot(x, s["t_hi_minus_lo_ex15"], color=COLORS[code], lw=1.3, ls="--", alpha=0.8)
    ax2.plot(x, s["corr_imb_fwd"], color=COLORS[code], lw=1.6, label=code)

ax1.axhspan(-2, 2, color="gray", alpha=0.12)
ax1.axhline(0, color="k", lw=0.8)
for y in (-2, 2):
    ax1.axhline(y, color="gray", lw=0.7, ls=":")
ax1.set_ylabel("t-stat: high − low sell-pressure\n(negative = profit-taking → weaker close)", fontsize=10)
ax1.set_title("Sell-pressure effect on the forward return to 15:00 close, up≥1%-from-open days\n"
              "(solid = full history, dashed = ex-2015; shaded |t|<2 = not significant)", fontsize=12)
ax1.legend(fontsize=8, ncol=4, loc="upper left")
ax1.grid(True, alpha=0.3)

ax2.axhline(0, color="k", lw=0.8)
ax2.set_ylabel("corr(sell-pressure, fwd→close)\nwithin up≥1% days", fontsize=10)
ax2.set_xlabel("signal minute (time of day, HH:MM)", fontsize=11)
ax2.legend(fontsize=8, ncol=4, loc="upper left")
ax2.grid(True, alpha=0.3)
# thin x ticks: every 5th minute
ticks = [i for i in range(len(o[o.code == "IC0000"])) if i % 5 == 0]
lab = o[o.code == "IC0000"].sort_values("tod")["hm"].tolist()
ax2.set_xticks(ticks); ax2.set_xticklabels([lab[i] for i in ticks], rotation=45)

plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig_flow_tstat.png", dpi=140)
print("saved fig_flow_tstat.png")
