"""crossimpact_plot.py — figures for Direction A (Cont-Cucuringu-Zhang 2023).
fig101: lead-lag return cross-correlation vs lag (per pair, W=1s) — peak location = who leads.
fig102: cross-OFI predictive R² (own vs full) per target & bin width, out-of-sample.
fig103: cross-OFI beta matrix heatmap (target r_i(t+1) ~ OFI_j(t), W=1s, train).
Run: python3 crossimpact_plot.py   (SUF=_pilot for pilot files)
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
D = "/Users/zhuisabella/xn/prediction"
SUF = os.environ.get("SUF", "")
NM = {"IC0000": "IC 中证500", "IF0000": "IF 沪深300", "IH0000": "IH 上证50", "IM0000": "IM 中证1000"}
CODES = ["IC0000", "IF0000", "IH0000", "IM0000"]
ll = pd.read_csv(f"{D}/xc_leadlag{SUF}.csv")
pr = pd.read_csv(f"{D}/xc_predict{SUF}.csv")
bt = pd.read_csv(f"{D}/xc_betas{SUF}.csv")

# fig101 — lead-lag cross-correlation curves (W=1s)
w = ll[ll.W == 1]
pairs = w[["a", "b"]].drop_duplicates().values
fig, axes = plt.subplots(2, 3, figsize=(13.5, 7))
for ax, (a, b) in zip(axes.ravel(), pairs):
    g = w[(w.a == a) & (w.b == b)].sort_values("lag")
    ax.axhline(0, color="0.6", lw=.6); ax.axvline(0, color="0.6", lw=.6)
    ax.plot(g.lag, g["corr"], marker="o", ms=5, lw=1.8, color="#4C72B0")
    pk = g.loc[g["corr"].abs().idxmax()]
    ax.set_title(f"{NM[a].split()[0]} vs {NM[b].split()[0]}  峰值@lag{int(pk['lag'])}", fontsize=10, fontweight="bold")
    ax.set_xlabel("lag ℓ (1s bins) —— ℓ>0: 左者领先", fontsize=8.5)
    ax.set_ylabel("corr(r_a(t), r_b(t+ℓ))", fontsize=8.5); ax.grid(True, alpha=.25)
fig.suptitle("图101　收益率跨合约领先-滞后（W=1s）—— 峰值几乎都在 lag0 = 同步共动，无秒级领先",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{D}/fig101_跨合约领先滞后{SUF}.png", dpi=135); plt.close(fig)

# fig102 — cross-OFI predictive R² (own vs full), OOS if present else train
fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
Ws = sorted(pr.W.unique())
usecol = "r2_oos" if pr["r2_oos"].notna().any() else "r2_train"
lab = "样本外" if usecol == "r2_oos" else "训练内"
for ax, W in zip(axes, Ws):
    s = pr[pr.W == W]
    x = np.arange(len(CODES)); wd = 0.36
    own = [s[(s.target == c) & (s.model == "own")][usecol].iloc[0] for c in CODES]
    full = [s[(s.target == c) & (s.model == "full")][usecol].iloc[0] for c in CODES]
    ax.bar(x - wd / 2, own, wd, color="0.6", label="仅自身OFI")
    ax.bar(x + wd / 2, full, wd, color="#c0392b", label="自身+跨合约OFI")
    ax.set_xticks(x); ax.set_xticklabels([NM[c].split()[0] for c in CODES], fontsize=9)
    ax.set_title(f"W={W}s", fontsize=10.5, fontweight="bold")
    ax.set_ylabel(f"预测 R²（{lab}）", fontsize=9); ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=.3)
fig.suptitle(f"图102　跨合约OFI提升下一区间收益预测（{lab}）—— 加入他人订单流，R² 翻 2–3 倍",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(f"{D}/fig102_跨合约预测R2{SUF}.png", dpi=135); plt.close(fig)

# fig103 — cross-OFI beta heatmap (W=1s)
b1 = bt[bt.W == 1]
M = np.full((len(CODES), len(CODES)), np.nan)
for _, r in b1.iterrows():
    M[CODES.index(r.target), CODES.index(r.source)] = r.beta
fig, ax = plt.subplots(figsize=(7.2, 6))
vmax = np.nanmax(np.abs(M))
im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
ax.set_xticks(range(len(CODES))); ax.set_xticklabels([NM[c].split()[0] for c in CODES])
ax.set_yticks(range(len(CODES))); ax.set_yticklabels([NM[c].split()[0] for c in CODES])
ax.set_xlabel("来源 OFI_j(t)", fontsize=10); ax.set_ylabel("目标 r_i(t+1)", fontsize=10)
for i in range(len(CODES)):
    for j in range(len(CODES)):
        if np.isfinite(M[i, j]):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                    fontsize=10, fontweight="bold" if i != j else "normal",
                    color="black" if abs(M[i, j]) < vmax * .6 else "white")
fig.colorbar(im, ax=ax, fraction=.046, pad=.04, label="β (tick per OFI unit)")
ax.set_title("图103　跨合约OFI系数矩阵（W=1s，对角=自身）\n小盘 IC↔IM 互驱最强", fontsize=11.5, fontweight="bold")
fig.tight_layout(); fig.savefig(f"{D}/fig103_跨合约OFI系数{SUF}.png", dpi=135); plt.close(fig)
print("saved fig101, fig102, fig103")
