"""xc_icim_plot_2panel.py — fig104b: fig104 without the middle (controlled-beta) panel.
Left:  own vs pair R², IS vs OOS, per target (W=1s).
Right: IC↔IM return lead-lag IS vs OOS — peak at lag0, faint IM-leads-IC.
Run: python3 xc_icim_plot_2panel.py   (SUF=_pilot for pilot files)
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

fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(11, 5))

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
ax1.set_ylim(0, max(pair_is + pair_oos) * 1.32)
ax1.legend(fontsize=8.5, loc="upper right"); ax1.grid(True, axis="y", alpha=.3)

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

fig.suptitle("图104b　小盘对 IC↔IM 深挖：跨合约预测真实、样本外稳定",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(f"{D}/fig104b_ICIM深挖{SUF}.png", dpi=135)
print("saved fig104b")
