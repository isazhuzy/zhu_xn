"""steps_plot.py — figures + readouts for the three intraminute experiments.
Run with system python3 (has matplotlib):  python3 steps_plot.py
Reads step1_startscan.csv, step2_twomin.csv, step3_extcurve.csv, step3_vtrade.csv.
COVERAGE CAP = 0.90: past it the within-minute tick count thins out and the signed
curve explodes spuriously (the fig26 dashed-tail artifact) — so trough/peak are read
only inside reliable coverage.
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

D = "/Users/zhuisabella/xn/intraminute"
FIG = f"{D}/figs"
NAME = {"IC0000": "IC 中证500", "IF0000": "IF 沪深300", "IH0000": "IH 上证50", "IM0000": "IM 中证1000"}
COL = {"IC0000": "#c0392b", "IF0000": "#e08a3c", "IH0000": "#27ae60", "IM0000": "#4C72B0"}
COV = 0.90

# ============================ STEP 1 : entry-second scan ============================
d1 = pd.read_csv(f"{D}/step1_startscan.csv")
_ym = d1.groupby(["year", "month"]).size().index.tolist()
_w0, _w1 = _ym[0], _ym[-1]
TAG = "（%d个月 %d-%02d..%d-%02d；阈值≥10 ticks/分钟）" % (len(_ym), _w0[0], _w0[1], _w1[0], _w1[1])
g1 = d1.groupby(["code", "S"]).agg(sm=("sum", "sum"), ss=("sumsq", "sum"), n=("n", "sum")).reset_index()
g1["mean"] = g1["sm"] / g1["n"]
g1["se"] = np.sqrt((g1["ss"] / g1["n"] - g1["mean"] ** 2) / g1["n"])
g1["t"] = g1["mean"] / g1["se"]

fig, ax = plt.subplots(figsize=(9, 5.4))
for code in NAME:
    s = g1[g1.code == code].sort_values("S")
    ax.plot(s["S"], s["mean"], color=COL[code], lw=1.8, marker="o", ms=5, label=NAME[code])
ax.axhline(0, color="0.55", lw=.7)
ax.set_xlabel("进场秒数 S（在第 M 分钟的第 S 秒进，到第 M+1 分钟的第 S 秒出；满一分钟持有）", fontsize=10)
ax.set_ylabel("动量P&L  d·(出场价−进场价)（指数点/笔）\n>0=动量/延续  <0=反转", fontsize=10)
ax.set_title(f"步骤1　进场时点扫描·前一分钟动量满分钟持有\n{TAG}", fontsize=13, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
fig.tight_layout(); fig.savefig(f"{FIG}/fig46_step1_进场秒扫描.png", dpi=130); plt.close(fig)
print("saved fig46")
print("\n=== STEP 1: mean pnl (pts/trade) by entry-second, t in () ===")
for code in NAME:
    s = g1[g1.code == code].sort_values("S")
    print(" ", NAME[code], " ".join(f"S{int(r.S)}={r['mean']:+.4f}(t{r.t:+.1f})" for _, r in s.iterrows()))


def reliable(s):
    """return (tick, mean, n, cutoff_tick) capping at COV*max coverage."""
    s = s.sort_values("tick")
    n = s["n"].to_numpy(); tk = s["tick"].to_numpy(); y = s["mean"].to_numpy()
    cut = int(tk[n >= COV * n.max()].max())
    return tk, y, n, cut


# ============================ STEP 2 : 2-minute-bar curve ============================
d2 = pd.read_csv(f"{D}/step2_twomin.csv")
g2 = d2.groupby(["code", "tick"]).agg(sm=("sum", "sum"), n=("n", "sum")).reset_index()
g2["mean"] = g2["sm"] / g2["n"]
fig, ax = plt.subplots(figsize=(9, 5.4))
print("\n=== STEP 2: 2-min-bar curve (within >=90% coverage) ===")
ylo = yhi = 0.0
for code in NAME:
    tk, y, n, cut = reliable(g2[g2.code == code])
    ax.plot(tk[:cut], y[:cut], color=COL[code], lw=1.8, label=NAME[code])
    ax.plot(tk[cut - 1:], y[cut - 1:], color=COL[code], lw=0.9, ls="--", alpha=0.30)
    yr = y[:cut]; ylo = min(ylo, yr.min()); yhi = max(yhi, yr.max())
    it = yr.argmin()
    print("  %s: trough y=%+.3f @tick%d(~%.0fs)  end(@cut tick%d) y=%+.3f"
          % (NAME[code], yr[it], tk[it], tk[it] * 0.5, cut, yr[-1]))
pad = (yhi - ylo) * 0.25
ax.set_ylim(ylo - pad, yhi + pad); ax.set_xlim(0, 245)
ax.axhline(0, color="0.55", lw=.7); ax.axvline(120, color="0.7", lw=.7, ls=":")
ax.set_xlabel("2分钟bar内 tick 序号（真实 tick；虚线=覆盖<90%，不可信尾部）", fontsize=10)
ax.set_ylabel("d·(价格−bar开盘)（指数点）", fontsize=10)
ax.set_title(f"步骤2　2分钟bar上的动量位移曲线（fig26的2分钟版）\n{TAG}", fontsize=13, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
fig.tight_layout(); fig.savefig(f"{FIG}/fig47_step2_2分钟bar曲线.png", dpi=130); plt.close(fig)
print("saved fig47")

# ============================ STEP 3 : extended curve + V-trade ============================
d3 = pd.read_csv(f"{D}/step3_extcurve.csv")
g3 = d3.groupby(["code", "tick"]).agg(sm=("sum", "sum"), n=("n", "sum")).reset_index()
g3["mean"] = g3["sm"] / g3["n"]
fig, ax = plt.subplots(figsize=(9, 5.4))
print("\n=== STEP 3: extended (minute M + M+1) signed curve, within >=90% coverage ===")
ylo = yhi = 0.0
for code in NAME:
    tk, y, n, cut = reliable(g3[g3.code == code])
    ax.plot(tk[:cut], y[:cut], color=COL[code], lw=1.8, label=NAME[code])
    ax.plot(tk[cut - 1:], y[cut - 1:], color=COL[code], lw=0.9, ls="--", alpha=0.30)
    yr = y[:cut]; ylo = min(ylo, yr.min()); yhi = max(yhi, yr.max())
    iF = yr.argmin()
    after = yr.copy(); after[:iF] = -9e9; iX = after.argmax()
    ax.scatter([tk[iF]], [yr[iF]], color=COL[code], marker="v", s=60, zorder=5)
    print("  %s: trough F* y=%+.3f @tick%d(~%.0fs)  peak X* y=%+.3f @tick%d(%s ~%.0fs)  rel-cut=tick%d"
          % (NAME[code], yr[iF], tk[iF], tk[iF] * 0.5, yr[iX], tk[iX],
             "M+1" if tk[iX] > 120 else "M", (tk[iX] % 120) * 0.5, cut))
pad = (yhi - ylo) * 0.25
ax.set_ylim(ylo - pad, yhi + pad); ax.set_xlim(0, 245)
ax.axhline(0, color="0.55", lw=.7)
ax.axvline(120, color="0.5", lw=1.0, ls=":")
ax.text(122, yhi + pad * 0.3, "分钟收盘→下一分钟", fontsize=8, color="0.4")
ax.set_xlabel("从第M分钟开盘起的 tick 序号（>120=进入第M+1分钟；虚线=覆盖<90%伪迹）", fontsize=10)
ax.set_ylabel("d·(价格−M分钟开盘)（指数点）", fontsize=10)
ax.set_title(f"步骤3　跨分钟延伸的动量位移曲线——趋势到哪里停？\n{TAG}", fontsize=13, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
fig.tight_layout(); fig.savefig(f"{FIG}/fig48_step3_跨分钟延伸曲线.png", dpi=130); plt.close(fig)
print("saved fig48")

# V-trade heatmaps + best
v = pd.read_csv(f"{D}/step3_vtrade.csv")
gv = v.groupby(["code", "F", "X"]).agg(sm=("sum", "sum"), ss=("sumsq", "sum"), n=("n", "sum")).reset_index()
gv["mean"] = gv["sm"] / gv["n"]
gv["se"] = np.sqrt((gv["ss"] / gv["n"] - gv["mean"] ** 2) / gv["n"])
gv["t"] = gv["mean"] / gv["se"]
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
print("\n=== STEP 3: V-trade (short 0->F, long F->X) best gross cell per contract ===")
for ax, code in zip(axes.ravel(), NAME):
    p = gv[gv.code == code].pivot(index="F", columns="X", values="mean")
    im = ax.imshow(p.values, cmap="RdBu_r", aspect="auto",
                   vmin=-abs(p.values).max(), vmax=abs(p.values).max())
    ax.set_xticks(range(len(p.columns))); ax.set_xticklabels(p.columns, fontsize=8)
    ax.set_yticks(range(len(p.index))); ax.set_yticklabels(p.index, fontsize=8)
    ax.set_xlabel("出场 tick X"); ax.set_ylabel("翻仓 tick F")
    b = gv[(gv.code == code) & (gv.X <= 140)].sort_values("mean", ascending=False).iloc[0]
    ax.set_title("%s  best F=%d X=%d  %+.3f pts (t=%.1f)" % (NAME[code], b["F"], b["X"], b["mean"], b["t"]),
                 fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046)
    print("  %s best: F=%d X=%d  gross=%+.3f pts  t=%+.2f  n=%d  (X=120 ~ minute close)"
          % (NAME[code], b["F"], b["X"], b["mean"], b["t"], b["n"]))
fig.suptitle(f"步骤3　分钟内V型交易毛利网格（翻仓F×出场X，指数点/笔，未计成本；best限X≤140可信覆盖）\n{TAG}",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(f"{FIG}/fig49_step3_V交易网格.png", dpi=130); plt.close(fig)
print("saved fig49")

# ============================ CROSS-MONTH STABILITY (full window) ============================
# Step 1: per-month hit-rate at the open entry (S=0).
print("\n=== STABILITY: step1 S=0 momentum pnl — fraction of months matching aggregate sign ===")
for code in NAME:
    s = d1[(d1.code == code) & (d1.S == 0)].copy()
    s["mm"] = s["sum"] / s["n"]
    agg = s["sum"].sum() / s["n"].sum()
    frac = (np.sign(s["mm"]) == np.sign(agg)).mean()
    print("  %s: agg=%+.4f pts  %.0f%% of %d months same sign" % (NAME[code], agg, 100 * frac, len(s)))

# Step 3: pick best (F,X) per contract on full aggregate, then per-month series.
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
print("\n=== STABILITY: step3 V-trade at the best (F,X) — per-month hit-rate & equity ===")
for ax, code in zip(axes.ravel(), NAME):
    b = gv[(gv.code == code) & (gv.X <= 140)].sort_values("mean", ascending=False).iloc[0]
    F, X = int(b["F"]), int(b["X"])
    m = v[(v.code == code) & (v.F == F) & (v.X == X)].copy().sort_values(["year", "month"])
    m["mm"] = m["sum"] / m["n"]                       # per-month gross pts/trade
    m["lab"] = m["year"].astype(str).str[2:] + "-" + m["month"].astype(str).str.zfill(2)
    aggm = m["sum"].sum() / m["n"].sum()
    hit = (np.sign(m["mm"]) == np.sign(aggm)).mean()
    cum = m["sum"].cumsum()                           # cumulative gross points (sum over trades)
    ax.bar(range(len(m)), m["mm"], color=COL[code], alpha=0.55, width=0.9)
    ax.axhline(aggm, color="k", lw=1.0, ls="--")
    ax2 = ax.twinx(); ax2.plot(range(len(m)), cum, color="0.2", lw=1.4)
    ax2.set_ylabel("累计毛利(点)", fontsize=8)
    step = max(1, len(m) // 12)
    ax.set_xticks(range(0, len(m), step)); ax.set_xticklabels(m["lab"].iloc[::step], rotation=90, fontsize=6)
    ax.axhline(0, color="0.55", lw=.6)
    ax.set_title("%s  F=%d X=%d  agg=%+.3f  %.0f%%月同号" % (NAME[code], F, X, aggm, 100 * hit),
                 fontsize=10, fontweight="bold")
    ax.set_ylabel("每月毛利(点/笔)", fontsize=8)
    print("  %s: best F=%d X=%d  agg=%+.4f pts  %.0f%% of %d months same sign  cum=%.1f pts"
          % (NAME[code], F, X, aggm, 100 * hit, len(m), cum.iloc[-1]))
fig.suptitle(f"步骤3　V型交易最优(F,X)·逐月毛利与累计（柱=每月点/笔；线=累计点）\n{TAG}",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(f"{FIG}/fig50_step3_V交易逐月稳定性.png", dpi=130); plt.close(fig)
print("saved fig50")
