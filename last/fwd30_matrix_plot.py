"""fwd30_matrix_plot.py — heatmap of the (minute-of-day x day) momentum matrix.
Reads fwd<H>_matrix_<CODE><SUF>.csv from fwd30_matrix.py.
Color = strategy return in bps, clipped at +-CLIP for visibility (fat tails);
NaN cells (no signal / limit-up) drawn light gray. White line = lunch break.
Run: python3 fwd30_matrix_plot.py   (env: CODE, H, SUF=_pilot, CLIP)
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
_av = {f.name for f in fm.fontManager.ttflist}
for _f in ["Arial Unicode MS", "PingFang HK", "Heiti TC", "STHeiti", "Songti SC"]:
    if _f in _av:
        matplotlib.rcParams["font.sans-serif"] = [_f]; break
matplotlib.rcParams["axes.unicode_minus"] = False

D = "/Users/zhuisabella/xn/last"
CODE = os.environ.get("CODE", "IF0000")
H = int(os.environ.get("H", "30"))
SUF = os.environ.get("SUF", "")
CLIP = float(os.environ.get("CLIP", "30"))

m = pd.read_csv(f"{D}/fwd{H}_matrix_{CODE}{SUF}.csv", index_col="hm").drop(columns="t")
days = m.columns.tolist()
Z = m.to_numpy(float)

fig, ax = plt.subplots(figsize=(14, 6))
cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("0.88")
im = ax.imshow(np.clip(Z, -CLIP, CLIP), aspect="auto", cmap=cmap,
               vmin=-CLIP, vmax=CLIP, interpolation="nearest")
ax.axhline(119.5, color="white", lw=1.2)               # lunch break (t=120|121)
mstart = [i for i, d in enumerate(days) if i == 0 or d[:7] != days[i - 1][:7]]
ax.set_xticks(mstart); ax.set_xticklabels([days[i][:7] for i in mstart], fontsize=8)
ytick = [i for i, h in enumerate(m.index) if h.endswith(("00", "30"))]
ax.set_yticks(ytick); ax.set_yticklabels(m.index[ytick], fontsize=8)
ax.set_xlabel("交易日"); ax.set_ylabel("日内分钟 t（信号分钟）")
ax.set_title(f"{CODE} 动量矩阵热图：sign(第t分钟) × 未来{H}分钟收益（bps，截断±{CLIP:g}）"
             f"　白线=午休　灰=无信号/坏盘口", fontsize=11, fontweight="bold")
cb = fig.colorbar(im, ax=ax, pad=0.01); cb.set_label("bps")
fig.tight_layout(); fig.savefig(f"{D}/fig_fwd{H}_matrix_{CODE}{SUF}.png", dpi=135)
print(f"saved fig_fwd{H}_matrix_{CODE}{SUF}.png")
