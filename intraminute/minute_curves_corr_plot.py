"""minute_curves_corr_plot.py — cross-month correlation of the intra-minute curve.
For each contract, take every month's raw per-tick mean displacement over ticks 2..100
(tick 1 is structurally 0), and compute the month x month Pearson correlation of those
curves. Shows which months share the same within-minute shape. RAW tick index (no
time-normalization), cut at tick 100. Run:  python3 minute_curves_corr_plot.py
"""
import glob
import re
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

INDIR = "/Users/zhuisabella/xn/intraminute/raw_months"
OUT = "/Users/zhuisabella/xn/intraminute/figs"
NAME = {"IC0000": "IC 中证500", "IF0000": "IF 沪深300", "IH0000": "IH 上证50", "IM0000": "IM 中证1000"}
TMIN, TMAX = 2, 100

files = sorted(glob.glob(f"{INDIR}/minute_curves_raw_*.csv"),
               key=lambda f: re.search(r"(\d{4})_(\d{2})", f).group(0))
months = [re.search(r"(\d{4})_(\d{2})", f).group(0).replace("_", "-") for f in files]
tabs = {m: pd.read_csv(f) for m, f in zip(months, files)}


def curve_matrix(code):
    """rows = months that have this contract, cols = ticks TMIN..TMAX (mean displacement)."""
    rows, labels = [], []
    for m in months:
        s = tabs[m][tabs[m].code == code].set_index("tick")["mean"]
        if s.empty:
            continue
        vec = s.reindex(range(TMIN, TMAX + 1)).to_numpy()
        rows.append(vec); labels.append(m)
    return np.array(rows), labels


fig, axes = plt.subplots(2, 2, figsize=(15, 14))
for ax, code in zip(axes.ravel(), NAME):
    M, labels = curve_matrix(code)
    C = np.corrcoef(M)                                   # month x month correlation
    off = C[~np.eye(len(C), dtype=bool)].mean()         # average off-diagonal corr
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6.5)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_title(f"{NAME[code]}　{len(labels)}个月  (平均相关={off:+.2f})",
                 fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle("图24　分钟内曲线 跨月相关性（每格=两月份 tick2–100 价格曲线的Pearson相关;原始tick,未归一化）",
             fontsize=15, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.975))
fig.savefig(f"{OUT}/fig24_跨月相关性.png", dpi=130); plt.close(fig)
print("saved fig24")
for code in NAME:
    M, labels = curve_matrix(code)
    C = np.corrcoef(M)
    off = C[~np.eye(len(C), dtype=bool)].mean()
    print(f"{code}: {len(labels)} months, mean off-diagonal corr = {off:+.3f}")
