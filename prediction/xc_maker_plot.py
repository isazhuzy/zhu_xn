"""xc_maker_plot.py — fig106: the capstone. Market-making IC with three quote centers.
Left:  avg maker markout per fill (ticks) — mid vs micro vs cross — each signal layer
       lifts the per-fill edge by pulling toxic fills.
Right: fills accepted (fewer = more toxic-side pulling) with net total P&L annotated.
Run: python3 xc_maker_plot.py   (SUF=_pilot for pilot files)
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
r = pd.read_csv(f"{D}/xc_maker_results{SUF}.csv")
CEN = ["mid", "micro", "cross"]
LBL = {"mid": "中间价 mid\n(naive, 不skew)", "micro": "微观价格 P_micro\n(论文4)",
       "cross": "P_micro+跨合约漂移\n(论文4+5)"}
COL = {"mid": "0.6", "micro": "#4C72B0", "cross": "#c0392b"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))
TAUS = sorted(r.tau.unique())
x = np.arange(len(CEN)); wd = 0.24

# left: avg markout per fill (all), by center, grouped by tau
for j, tau in enumerate(TAUS):
    s = r[(r.tau == tau) & (r.side == "all")].set_index("center")
    vals = [s.loc[c, "avg_markout_tk"] for c in CEN]
    bars = ax1.bar(x + (j - 1) * wd, vals, wd, color=[COL[c] for c in CEN], alpha=[.5, .75, 1][j],
                   label=f"τ={tau}")
ax1.axhline(0, color="k", lw=1)
ax1.set_xticks(x); ax1.set_xticklabels([LBL[c] for c in CEN], fontsize=9)
ax1.set_ylabel("每笔成交 markout（tick）= 收半价差 − 逆选", fontsize=10)
ax1.set_title("三种报价中心的做市单笔毛收益（τ=skew阈值）\n每加一层信号，逆选更少、单笔更高", fontsize=11, fontweight="bold")
ax1.legend(fontsize=9, title="skew阈值"); ax1.grid(True, axis="y", alpha=.3)

# right: at tau=0.25, fills accepted + total P&L
tau0 = 0.25
s = r[(r.tau == tau0) & (r.side == "all")].set_index("center")
nf = [s.loc[c, "n_fills"] for c in CEN]
tp = [s.loc[c, "total_tk"] for c in CEN]
b = ax2.bar(x, [n / 1e6 for n in nf], color=[COL[c] for c in CEN])
for i, c in enumerate(CEN):
    ax2.text(i, nf[i] / 1e6 + max(nf) / 1e6 * .02,
             f"单笔{s.loc[c,'avg_markout_tk']:.2f}tk\n毛利{tp[i]/1e3:,.0f}k tk", ha="center", fontsize=8.5)
ax2.set_xticks(x); ax2.set_xticklabels([c for c in CEN], fontsize=10)
ax2.set_ylabel("接受成交笔数（百万，τ=0.25）", fontsize=10)
ax2.set_title("信号让你少接毒单：笔数↓ 但单笔↑\n（毛收益，未扣手续费/排队/离场成本）", fontsize=11, fontweight="bold")
ax2.grid(True, axis="y", alpha=.3)

fig.suptitle("图106　做市视角：用 P_micro(论文4)+跨合约漂移(论文5) 做市 IC — 宽价差从 taker 的敌人变 maker 的收入",
             fontsize=12.5, fontweight="bold")
fig.text(0.5, 0.005, "⚠ 毛收益，乐观假设：总在最优价成交、可在mid离场、无手续费/排队优先。相对比较(mid<micro<cross)稳健，绝对值偏乐观。",
         ha="center", fontsize=8.5, color="0.4")
fig.tight_layout(rect=(0, 0.03, 1, 0.95)); fig.savefig(f"{D}/fig106_做市资本之作{SUF}.png", dpi=135)
print("saved fig106")
