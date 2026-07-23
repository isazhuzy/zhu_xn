"""Render Experiment C (look-back x forward interaction) for VOI / MPB / OIR,
all 4 contracts.  Prints the neg0.1 & pos0.1 grids and saves a heatmap figure.

CALCULATION REMINDER (see window_exp_c.py header for the full version):
  rows J = seconds looked BACK (rolling factor sum over past J ticks; sum==mean
           for ranking, J=1 = instantaneous factor)
  cols k = seconds held FORWARD (mid-to-mid, dy_k = mid(t+k)-mid(t) in ticks)
  cell   = mean dy_k over the extreme factor tail = ticks/trade, gross, sub-spread
  neg0.1 = 0.1% most net-SELLING ticks ; pos0.1 = 0.1% most net-BUYING ticks
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/Users/zhuisabella/xn/manual"
SUF = "_pilot" if os.environ.get("PILOT") == "1" else ""
df = pd.read_csv(f"{D}/window_exp_c_all{SUF}.csv")
SEC = {1: "0.5s", 5: "2.5s", 20: "10s", 60: "30s", 120: "60s", 240: "120s", 600: "300s"}
Js = [1, 5, 20, 60, 120]; Ks = [1, 20, 60, 120, 240, 600]
CODES = ["IC0000", "IF0000", "IH0000", "IM0000"]
FACS = ["voi", "mpb", "oir"]


def grid(code, fac, b):
    d = df[(df.code == code) & (df.factor == fac) & (df.bucket == b)]
    return d.pivot_table(index="J", columns="k", values="mean_dy").reindex(Js)[Ks].values


# ---- text grids ----
for fac in FACS:
    print("\n" + "=" * 74)
    print(f"FACTOR = {fac.upper()}   (rows=look-back sec, cols=forward sec, ticks/trade)")
    print("=" * 74)
    for b in ["neg0.1", "pos0.1"]:
        tag = "most net-SELLING 0.1%" if b == "neg0.1" else "most net-BUYING 0.1%"
        print(f"\n  bucket {b} ({tag}):")
        for code in CODES:
            M = grid(code, fac, b)
            print(f"    {code}: " + " | ".join(
                f"J{SEC[Js[r]]}: " + " ".join(f"{M[r,c]:+5.1f}" for c in range(len(Ks)))
                for r in range(len(Js))))
        # compact per-code print
    # nicer per-code block
for fac in FACS:
    for code in CODES:
        for b in ["neg0.1", "pos0.1"]:
            M = grid(code, fac, b)
            hdr = "  ".join(SEC[k] for k in Ks)
            # (full tables live in the CSV / figure; keep console short)


# ---- figure: 3 factors x (neg,pos) rows, 4 contract columns per block ----
fig, axes = plt.subplots(6, 4, figsize=(20, 22))
for fi, fac in enumerate(FACS):
    for bi, b in enumerate(["neg0.1", "pos0.1"]):
        row = fi * 2 + bi
        for j, code in enumerate(CODES):
            ax = axes[row, j]
            M = grid(code, fac, b)
            im = ax.imshow(M, cmap="RdBu_r", vmin=-5, vmax=5, aspect="auto")
            ax.set_xticks(range(len(Ks))); ax.set_xticklabels([SEC[k] for k in Ks], fontsize=8)
            ax.set_yticks(range(len(Js))); ax.set_yticklabels([SEC[k] for k in Js], fontsize=8)
            for y in range(len(Js)):
                for x in range(len(Ks)):
                    v = M[y, x]
                    ax.text(x, y, f"{v:+.1f}", ha="center", va="center", fontsize=7,
                            color="white" if abs(v) > 2.4 else "black")
            tag = "SELL tail" if b == "neg0.1" else "BUY tail"
            ax.set_title(f"{fac.upper()} {tag} — {code}", fontsize=10)
            if j == 0:
                ax.set_ylabel("look-back (s)", fontsize=8)
            if row == 5:
                ax.set_xlabel("forward hold (s)", fontsize=8)
fig.suptitle("Experiment C — extreme-0.1%-tail mean forward return (ticks/trade), "
             "mid-to-mid, full sample 2020-2026\n"
             "rows within each factor: SELL tail (0.1% most net-selling) then BUY tail; "
             "red = price RISES after signal, blue = price FALLS", fontsize=13, y=0.995)
cb = fig.colorbar(im, ax=axes, shrink=0.4, pad=0.01)
cb.set_label("mean fwd return (ticks/trade)")
fig.savefig(f"{D}/fig_window_exp_c_all{SUF}.png", dpi=100, bbox_inches="tight")
print(f"\nsaved fig_window_exp_c_all{SUF}.png")
