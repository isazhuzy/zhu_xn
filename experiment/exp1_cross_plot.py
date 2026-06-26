"""exp1_cross_plot.py — 方向1+2 分析: 跨合约 × 分年度的反转 hardline 稳定性 + 反转vs价差。
Run: python3 exp1_cross_plot.py   (reads exp1_cross.csv, exp1_cross_spread.csv; writes 2 figs)
 fig_cross_hardline.png : 4 合约，各 year(行)×k(列) 热力图(n=10)，色=均值符号 → hardline 跨年/跨合约稳不稳。
 fig_cross_spread.png   : 4 合约，逐年 反转幅度(n=10,k=30=6点) vs 该合约真实价差 → 在哪个合约/年净越过成本。
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

D = "/Users/zhuisabella/xn/experiment"
FIG = f"{D}/figs"
import os
os.makedirs(FIG, exist_ok=True)
NAME = {"IC0000": "IC 中证500", "IF0000": "IF 沪深300", "IH0000": "IH 上证50", "IM0000": "IM 中证1000"}
CODES = ["IC0000", "IF0000", "IH0000", "IM0000"]

d = pd.read_csv(f"{D}/exp1_cross.csv")
d["mean"] = d["s_sum"] / d["s_n"]
d["se"] = np.sqrt(np.maximum(d["s_ss"] / d["s_n"] - d["mean"] ** 2, 0) / d["s_n"])
d["t"] = d["mean"] / d["se"]
d["revhit"] = 100 * d["hits"] / d["s_n"]
sp = pd.read_csv(f"{D}/exp1_cross_spread.csv")
sp["spread"] = sp["sp_sum"] / sp["sp_n"]
spmap = {(r.code, r.year): r.spread for r in sp.itertuples()}
KT = sorted(d["k_ticks"].unique())

# ---------- Fig1: hardline 稳定性 (year × k at n=10) ----------
NREF = 10
fig, axes = plt.subplots(1, 4, figsize=(19, 6), constrained_layout=True)
vlim = 0.5
im = None
for ax, code in zip(axes, CODES):
    sub = d[(d.code == code) & (d.n == NREF)]
    yrs = sorted(sub["year"].unique())
    Z = sub.pivot(index="year", columns="k_ticks", values="mean").reindex(index=yrs, columns=KT)
    im = ax.imshow(Z.values, cmap="seismic", vmin=-vlim, vmax=vlim, aspect="auto", origin="upper")
    ax.set_xticks(range(len(KT))); ax.set_xticklabels(KT, fontsize=8)
    ax.set_yticks(range(len(yrs))); ax.set_yticklabels(yrs, fontsize=8)
    for ii in range(len(yrs)):
        for jj in range(len(KT)):
            v = Z.values[ii, jj]
            if np.isfinite(v):
                ax.text(jj, ii, (f"{v:+.2f}" if abs(v) < 10 else f"{v:+.0f}"),
                        ha="center", va="center", fontsize=6,
                        color="white" if abs(min(max(v, -vlim), vlim)) > 0.6 * vlim else "black")
    ax.set_title(NAME[code], fontsize=11)
    ax.set_xlabel("k 阈值（价格tick）", fontsize=9)
axes[0].set_ylabel("年份", fontsize=10)
fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, extend="both",
             label="均值符号化收益（点）红=趋势/蓝=反转")
fig.suptitle(f"方向1+2：反转 hardline 跨合约×跨年度稳定性（n={NREF}, x=10）—— 红转蓝的列稳不稳？  IM中证1000等",
             fontsize=13)
fig.savefig(f"{FIG}/fig_cross_hardline.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"saved {FIG}/fig_cross_hardline.png")

# ---------- Fig2: 反转幅度 vs 价差 逐年 ----------
NREF2, KREF = 10, 30      # n=10, k=30 tick = 6 点
fig, axes = plt.subplots(1, 4, figsize=(19, 5), sharey=False)
for ax, code in zip(axes, CODES):
    sub = d[(d.code == code) & (d.n == NREF2) & (d.k_ticks == KREF)].sort_values("year")
    yrs = sub["year"].tolist()
    revmag = (-sub["mean"]).clip(lower=-1).tolist()        # 逆向幅度=−均值（正=反转）
    thin = (sub["s_n"] < 30000).tolist()
    spr = [spmap.get((code, y), np.nan) for y in yrs]
    xp = range(len(yrs))
    bars = ax.bar(xp, revmag, color="#4C72B0", alpha=0.85, label="反转幅度(毛,点)")
    for b, th in zip(bars, thin):
        if th:
            b.set_color("#cccccc")
    ax.plot(xp, spr, color="#c0392b", lw=2, marker="o", ms=4, label="该合约真实价差(点)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(list(xp)); ax.set_xticklabels([str(y)[2:] for y in yrs], fontsize=7.5, rotation=0)
    ax.set_title(NAME[code], fontsize=11)
    ax.set_xlabel("年份", fontsize=9)
axes[0].set_ylabel("点（指数点）", fontsize=10)
axes[0].legend(fontsize=8, loc="upper right")
fig.suptitle("方向1+2：反转毛幅度（蓝, n=10 k=6点）vs 真实价差（红）逐年 —— 蓝高于红才有净空间（灰=样本<3万）",
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(f"{FIG}/fig_cross_spread.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"saved {FIG}/fig_cross_spread.png")

# ---------- 文字汇总 ----------
print("\n=== 各合约：平均价差 / 全期反转(n=10,k=6点) ===")
for code in CODES:
    s = d[(d.code == code) & (d.n == 10) & (d.k_ticks == 30)]
    agg_mean = s["s_sum"].sum() / s["s_n"].sum()
    agg_hit = 100 * s["hits"].sum() / s["s_n"].sum()
    msp = sp[sp.code == code]["spread"].mean()
    print(f"  {NAME[code]:>10}: 价差≈{msp:.2f}点  反转幅度≈{-agg_mean:.2f}点  胜率≈{agg_hit:.0f}%  "
          f"{'净>0?' if -agg_mean > msp else '净≈负(在价差内)'}")
