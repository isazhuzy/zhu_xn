"""
plot_close_continuation.py — figures for the closing-minutes continuation
finding (deepdive_close_continuation.py). Run on system python3:
    python3 plot_close_continuation.py
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

FIGDIR = "/Users/zhuisabella/xn/end/figs"
os.makedirs(FIGDIR, exist_ok=True)
CODES = ["IC0000", "IF0000", "IH0000", "IM0000"]
COLORS = {"IC0000": "tab:blue", "IF0000": "tab:orange", "IH0000": "tab:green", "IM0000": "tab:red"}

zoom = pd.read_csv("/Users/zhuisabella/xn/end/deepdive_zoom_1430_1500.csv")
win = pd.read_csv("/Users/zhuisabella/xn/end/deepdive_window_screen.csv")

# ---------- fig1: zoomed t-stat curve 14:30-15:00, all contracts on one axis ----------
fig, ax = plt.subplots(figsize=(10, 5))
for code in CODES:
    s = zoom[zoom.code == code].sort_values("tod")
    ax.plot(s["hm"], s["t"], color=COLORS[code], marker=".", ms=4, label=code)
ax.axhline(0, color="k", lw=0.8)
ax.axhline(2, color="gray", ls="--", lw=0.8, label="t=+2 (sig. continuation)")
ax.axhline(-2, color="gray", ls=":", lw=0.8, label="t=-2 (sig. reversal)")
ax.axvline("14:55", color="red", lw=1, alpha=0.4)
ax.set_xticks(s["hm"][::2])
ax.tick_params(axis="x", rotation=45)
ax.set_ylabel("t-stat, fwd return t->15:00 close")
ax.set_title("The sign flip: reversal (14:30-14:50) -> continuation (14:52-14:59)\nup>=1%-from-open days, plain trigger")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig_cc01_zoom_tstat.png", dpi=130)
print("saved fig_cc01_zoom_tstat.png")

# ---------- fig2: window-length x signal-time heatmap, per contract ----------
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, code in zip(axes.flat, CODES):
    sub = win[win.code == code]
    piv = sub.pivot(index="hm", columns="w", values="t")
    piv = piv.reindex(index=sorted(piv.index, key=lambda h: (int(h[:2]), int(h[3:]))))
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-4, vmax=6, aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_xlabel("lookback window w (min)")
    ax.set_title(code)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                         color="white" if abs(v) > 3 else "black")
    plt.colorbar(im, ax=ax, label="t-stat")
fig.suptitle("t-stat of fwd return (signal->close): up>=1% from open AND still rising over last w min")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig_cc02_window_heatmap.png", dpi=130)
print("saved fig_cc02_window_heatmap.png")
