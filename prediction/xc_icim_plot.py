"""xc_icim_plot.py — fig104: the IC↔IM small-cap zoom-in.
Left:  own vs pair R², IS vs OOS, per target (W=1s) — the pair effect holds out-of-sample.
Mid:   pairwise vs controlled(+IF/IH) cross-beta — ~80% survives ⇒ genuine small-cap sub-factor.
Right: IC↔IM return lead-lag IS vs OOS — peak at lag0, faint IM-leads-IC (lag-1>lag+1).
Run: python3 xc_icim_plot.py   (SUF=_pilot for pilot files)
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
p = pd.read_csv(f"{D}/xc_icim_predict{SUF}.csv")
ll = pd.read_csv(f"{D}/xc_icim_leadlag{SUF}.csv")
try:
    b = pd.read_csv(f"{D}/xc_betas{SUF}.csv"); b1 = b[b.W == 1]
    ctrl = {"IC": b1[(b1.target == "IC0000") & (b1.source == "IM0000")].beta.iloc[0],
            "IM": b1[(b1.target == "IM0000") & (b1.source == "IC0000")].beta.iloc[0]}
except Exception:
    ctrl = {"IC": np.nan, "IM": np.nan}

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# left: own vs pair, IS vs OOS (W=1s)
w = p[p.W == 1]
tgts = ["IC", "IM"]; x = np.arange(len(tgts)); wd = 0.2
own_is = [w[(w.target == t) & (w.model == "own")].r2_is.iloc[0] for t in tgts]
own_oos = [w[(w.target == t) & (w.model == "own")].r2_oos.iloc[0] for t in tgts]
pair_is = [w[(w.target == t) & (w.model == "pair")].r2_is.iloc[0] for t in tgts]
pair_oos = [w[(w.target == t) & (w.model == "pair")].r2_oos.iloc[0] for t in tgts]
ax1.bar(x - 1.5 * wd, own_is, wd, color="0.75", label="仅自身 IS")
ax1.bar(x - 0.5 * wd, own_oos, wd, color="0.5", label="仅自身 OOS")
ax1.bar(x + 0.5 * wd, pair_is, wd, color="#e08a3c", label="配对(+对方) IS")
ax1.bar(x + 1.5 * wd, pair_oos, wd, color="#c0392b", label="配对(+对方) OOS")
ax1.set_xticks(x); ax1.set_xticklabels(["目标 IC←IM", "目标 IM←IC"], fontsize=10)
ax1.set_ylabel("预测 R²（W=1s）", fontsize=10)
ax1.set_title("配对预测翻2.4倍，且样本外≈样本内", fontsize=11, fontweight="bold")
ax1.legend(fontsize=8.5); ax1.grid(True, axis="y", alpha=.3)

# mid: pairwise vs controlled cross-beta
pair_b = {t: w[(w.target == t) & (w.model == "pair")].cross_beta.iloc[0] for t in tgts}
xb = np.arange(len(tgts)); wb = 0.34
ax2.bar(xb - wb / 2, [pair_b[t] for t in tgts], wb, color="#c0392b", label="配对 β（未控制）")
ax2.bar(xb + wb / 2, [ctrl[t] for t in tgts], wb, color="#4C72B0", label="控制 IF/IH 后 β")
for i, t in enumerate(tgts):
    if np.isfinite(ctrl[t]):
        ax2.text(i, max(pair_b[t], ctrl[t]) + .003, f"{ctrl[t] / pair_b[t]:.0%} 存活",
                 ha="center", fontsize=9, fontweight="bold")
ax2.set_xticks(xb); ax2.set_xticklabels(["IM→IC", "IC→IM"], fontsize=10)
ax2.set_ylabel("跨合约 OFI 系数 β", fontsize=10)
ax2.set_title("控制大盘后 ~80% 存活\n= 真·小盘子因子（非市场beta）", fontsize=11, fontweight="bold")
ax2.legend(fontsize=8.5); ax2.grid(True, axis="y", alpha=.3)

# right: lead-lag IS vs OOS (W=1s)
w1 = ll[ll.W == 1]
for ph, c in [("IS", "#888"), ("OOS", "#c0392b")]:
    s = w1[w1.phase == ph].sort_values("lag")
    ax3.plot(s.lag, s["corr"], marker="o", ms=5, lw=1.8, color=c, label=ph)
ax3.axvline(0, color="0.6", lw=.6)
ax3.set_xlabel("lag ℓ (1s) — ℓ<0: IM 领先 IC", fontsize=9.5)
ax3.set_ylabel("corr(r_IC(t), r_IM(t+ℓ))", fontsize=10)
ax3.set_title("共动峰值在 lag0（IS/OOS 都稳）\n左略高于右 = IM 微弱领先 IC", fontsize=11, fontweight="bold")
ax3.legend(fontsize=9); ax3.grid(True, alpha=.25)

fig.suptitle("图104　小盘对 IC↔IM 深挖：跨合约预测真实、样本外稳定、~80%是小盘子因子",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(f"{D}/fig104_ICIM深挖{SUF}.png", dpi=135)
print("saved fig104")
