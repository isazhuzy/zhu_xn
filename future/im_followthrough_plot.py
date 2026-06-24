"""im_followthrough_plot.py — figures for the IM follow-through study.
Run with system python3 (has matplotlib):  python3 im_followthrough_plot.py
Reads price_trend.csv, volume_peak.csv (this dir); writes figs/ here.

EXP1 price pulse — for each variable, FIX THE OTHER TWO:
  fig_price_pulse.png : (a) vary n  | (b) vary k | (c) vary x(=h)   ±1 SE
  fig_price_heatmap.png : n × x grid of mean signed fwd return (the regime view)
EXP2 volume peak — change t1 / t2 ONE AT A TIME:
  fig_volume.png : rows = |fwd move| / signed fwd / fwd vol ; cols = vary t2 | vary t1
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

_av = {f.name for f in fm.fontManager.ttflist}
for _f in ["Arial Unicode MS", "PingFang HK", "Heiti TC", "STHeiti", "Songti SC"]:
    if _f in _av:
        matplotlib.rcParams["font.sans-serif"] = [_f]; break
matplotlib.rcParams["axes.unicode_minus"] = False

D = "/Users/zhuisabella/xn/future"
FIG = f"{D}/figs"
import os
os.makedirs(FIG, exist_ok=True)
TITLE = "IM 中证1000 · 2022-07..2026-05"
BLUE, RED, GREY = "#4C72B0", "#c0392b", "#888888"


def mt(sum_, ss, n):
    """mean, standard error from sum / sum-of-squares / count."""
    m = sum_ / n
    var = np.maximum(ss / n - m * m, 0.0)
    return m, np.sqrt(var / n)


# ====================== EXP1 : price pulse ======================
p = pd.read_csv(f"{D}/price_trend.csv")
p["mean"], p["se"] = mt(p["s_sum"], p["s_ss"], p["s_n"])
p["hitpct"] = 100.0 * p["hits"] / p["s_n"]

# reference point at which the other two vars are fixed in each panel
REF_N, REF_K, REF_H = 10, 0.2, 10
NS = sorted(p["n"].unique()); KS = sorted(p["k"].unique()); HS = sorted(p["h"].unique())

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
panels = [
    ("n", "n  (lookback ticks)", NS,
     (p.k == REF_K) & (p.h == REF_H), f"k={REF_K}, x={REF_H}"),
    ("k", "k  (threshold, index pts)", KS,
     (p.n == REF_N) & (p.h == REF_H), f"n={REF_N}, x={REF_H}"),
    ("h", "x  (forward horizon, ticks)", HS,
     (p.n == REF_N) & (p.k == REF_K), f"n={REF_N}, k={REF_K}"),
]
for ax, (col, xlab, xs, mask, fixed) in zip(axes, panels):
    s = p[mask].sort_values(col)
    ax.axhline(0, color=GREY, lw=0.8, ls="--")
    ax.errorbar(s[col], s["mean"], yerr=s["se"], color=BLUE, lw=1.9,
                marker="o", ms=6, capsize=3)
    for _, r in s.iterrows():
        ax.annotate(f"t={r['mean']/r['se']:+.0f}", (r[col], r["mean"]),
                    textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=7.5, color=GREY)
    ax.set_xlabel(xlab); ax.set_title(f"vary {col}   (fixed {fixed})", fontsize=10.5)
    ax.set_ylabel("mean signed fwd return (pts)\n>0 trend  ·  <0 reversal")
    if col in ("n", "h"):
        ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels(xs)
fig.suptitle(f"EXP1 price pulse — does a move lead to a trend?   {TITLE}",
             fontsize=12.5, y=1.02)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_price_pulse.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# regime heatmap: mean signed fwd over n (rows) x x (cols), at k=REF_K
piv = p[p.k == REF_K].pivot(index="n", columns="h", values="mean").reindex(index=NS, columns=HS)
fig, ax = plt.subplots(figsize=(7.6, 4.8))
vmax = np.nanmax(np.abs(piv.values))
im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(len(HS))); ax.set_xticklabels(HS)
ax.set_yticks(range(len(NS))); ax.set_yticklabels(NS)
ax.set_xlabel("x  (forward horizon, ticks)"); ax.set_ylabel("n  (lookback ticks)")
for i in range(len(NS)):
    for j in range(len(HS)):
        v = piv.values[i, j]
        ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=8,
                color="white" if abs(v) > 0.6 * vmax else "black")
fig.colorbar(im, ax=ax, label="mean signed fwd return (pts)")
ax.set_title(f"EXP1 regime: trend (red) vs reversal (blue)   k={REF_K}\n{TITLE}", fontsize=10.5)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_price_heatmap.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# ====================== EXP2 : volume peak ======================
v = pd.read_csv(f"{D}/volume_peak.csv")
v["absmove"], v["abse"] = mt(v["a_sum"], v["a_ss"], v["n_peak"])
v["signed"], v["sige"] = mt(v["g_sum"], v["g_ss"], v["n_peak"])
v["fvol"], v["fve"] = mt(v["v_sum"], v["v_ss"], v["n_peak"])

REF_H2 = 20
RS = ["1.5", "2.0", "3.0"]          # peak strengths (lines); ALL = baseline (dashed)
RCOL = {"1.5": "#9bbcd6", "2.0": BLUE, "3.0": "#1f3a5f"}
metrics = [("absmove", "|fwd move|  (pts)  — volatility", "abse"),
           ("signed", "signed fwd  (pts)\n>0 continue · <0 reverse", "sige"),
           ("fvol", "fwd volume / tick", "fve")]

fig, axes = plt.subplots(3, 2, figsize=(12, 11), sharex="col")
cols = [("t2", "vary t2  (t1=5 fixed)", v.t1 == 5, "t2"),
        ("t1", "vary t1  (t2=120 fixed)", v.t2 == 120, "t1")]
for ci, (xcol, ctitle, cmask, xlab) in enumerate(cols):
    sub = v[cmask & (v.h == REF_H2)]
    for ri, (mcol, ylab, ecol) in enumerate(metrics):
        ax = axes[ri, ci]
        base = sub[sub.r == "ALL"].sort_values(xcol)
        ax.plot(base[xcol], base[mcol], color=GREY, lw=1.6, ls="--",
                marker="s", ms=5, label="ALL (baseline)")
        for r in RS:
            s = sub[sub.r == r].sort_values(xcol)
            ax.errorbar(s[xcol], s[mcol], yerr=s[ecol], color=RCOL[r], lw=1.8,
                        marker="o", ms=5, capsize=2.5, label=f"peak r>{r}")
        if mcol == "signed":
            ax.axhline(0, color="k", lw=0.7)
        ax.set_ylabel(ylab)
        ax.set_xticks(sorted(sub[xcol].unique()))
        if ri == 0:
            ax.set_title(ctitle, fontsize=11)
        if ri == 2:
            ax.set_xlabel(f"{xlab}  (ticks)")
axes[0, 0].legend(fontsize=8, loc="best")
fig.suptitle(f"EXP2 volume peak — effect of a spike on the next {REF_H2} ticks   {TITLE}",
             fontsize=12.5, y=0.995)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_volume.png", dpi=130, bbox_inches="tight")
plt.close(fig)

print("saved:")
for f in ("fig_price_pulse.png", "fig_price_heatmap.png", "fig_volume.png"):
    print(f"  {FIG}/{f}")
