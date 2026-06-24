"""fig52_x140.py — two figures:
 fig52: per-contract V-trade but EXIT FIXED at X=140 (~10s into the next minute) —
        the 'hold into next minute' version. F* still = trough; 做多段 runs F*->140.
 fig53: gross-return comparison bar — 最优 X* vs X=140, split 做空段/做多段.
All GROSS (no costs). Momentum-normalised frame (curve = d*(price-open)).
Run: python3 fig52_x140.py
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
_av = {f.name for f in fm.fontManager.ttflist}
for _f in ["Arial Unicode MS", "PingFang HK", "Heiti TC", "STHeiti", "Songti SC"]:
    if _f in _av:
        matplotlib.rcParams["font.sans-serif"] = [_f]; break
matplotlib.rcParams["axes.unicode_minus"] = False

D = "/Users/zhuisabella/xn/intraminute"
NAME = {"IC0000": "IC 中证500", "IF0000": "IF 沪深300", "IH0000": "IH 上证50", "IM0000": "IM 中证1000"}
COL = {"IC0000": "#c0392b", "IF0000": "#e08a3c", "IH0000": "#27ae60", "IM0000": "#4C72B0"}
SHORT_BG = "#dce6f2"; LONG_BG = "#f3e7d3"
XFIX = 140

c = pd.read_csv(f"{D}/step3_extcurve.csv")
g = c.groupby(["code", "tick"]).agg(sm=("sum", "sum"), n=("n", "sum")).reset_index()
g["mean"] = g["sm"] / g["n"]

rows = {}
# ---------- fig52: X=140 panels ----------
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
for ax, code in zip(axes.ravel(), NAME):
    s = g[g.code == code].sort_values("tick")
    nmax = s["n"].max(); cut = int(s.loc[s.n >= 0.9 * nmax, "tick"].max())
    tk = s["tick"].to_numpy(); y = s["mean"].to_numpy(); ys = pd.Series(y, index=tk)
    F = int(ys.loc[10:110].idxmin()); yF = ys.loc[F]
    Xs = int(ys.loc[F + 1:cut].idxmax()); yXs = ys.loc[Xs]
    yX = ys.loc[XFIX]
    ret = yX - 2 * yF; short_pnl = -yF; long_pnl = yX - yF
    cov = 100 * s.loc[s.tick == XFIX, "n"].iloc[0] / nmax
    rows[code] = dict(F=F, Xs=Xs, ret_opt=yXs - 2 * yF, ret140=ret,
                      short=short_pnl, long_opt=yXs - yF, long140=long_pnl, cov=cov)
    drawcut = max(cut, XFIX + 5)
    sld = tk <= drawcut
    yr = y[tk <= max(cut, XFIX)]; lo, hi = yr.min(), yr.max(); pad = (hi - lo) * 0.42 + 0.01
    ylo, yhi = lo - pad, hi + pad
    ax.axvspan(0, F, color=SHORT_BG, zorder=0)
    ax.axvspan(F, XFIX, color=LONG_BG, zorder=0)
    solid = tk <= cut
    ax.plot(tk[solid], y[solid], color=COL[code], lw=2.2, zorder=4)
    ax.plot(tk[tk >= cut], y[tk >= cut], color=COL[code], lw=1.1, ls="--", alpha=0.45, zorder=4)
    ax.scatter([F], [yF], color=COL[code], marker="v", s=120, zorder=6, ec="k", lw=.6)
    ax.scatter([XFIX], [yX], color=COL[code], marker="^", s=130, zorder=6, ec="k", lw=.8)
    ax.annotate(f"翻仓 F*={F}（~{F*0.5:.0f}s）", (F, yF), textcoords="offset points",
                xytext=(6, -22), fontsize=8.5)
    ax.annotate(f"出场 X=140（下一分钟 ~10s）\n覆盖{cov:.0f}%", (XFIX, yX),
                textcoords="offset points", xytext=(-6, 10), fontsize=8.5, ha="right")
    ax.text(F / 2, yhi - (yhi - ylo) * 0.06, "① 做空 (−d)\n吃下跌段", ha="center", va="top",
            fontsize=9.5, fontweight="bold", color="#2c3e6b")
    ax.text((F + XFIX) / 2, yhi - (yhi - ylo) * 0.06, "② 做多 (+d) 吃回升段\n(持到下一分钟)",
            ha="center", va="top", fontsize=9.5, fontweight="bold", color="#8a5a1c")
    ax.axhline(0, color="0.45", lw=.7); ax.axvline(120, color="0.45", lw=1.0, ls=":")
    ax.text(120, ylo + (yhi - ylo) * 0.02, " 分钟收盘", fontsize=7.5, color="0.4")
    ax.set_ylim(ylo, yhi); ax.set_xlim(0, drawcut + 5)
    ax.set_xlabel("从分钟开盘起的 tick 序号（1 tick ≈ 0.5 秒；120 = 分钟收盘）", fontsize=9)
    ax.set_ylabel("动量框架位移  y = d·(价格 − 开盘)  （指数点）", fontsize=9)
    ax.set_title("%s    毛利 = %+.3f 点/笔   ( 做空段 %+.3f  +  做多段 %+.3f )"
                 % (NAME[code], ret, short_pnl, long_pnl), fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.22, zorder=1)
fig.suptitle("图52　四合约·分钟内 V 型交易（出场固定 X=140 = 持到下一分钟 ~10s 的版本）\n"
             "开盘 → ① 做空(−d) 吃下跌 → 谷底 F* 翻仓 → ② 做多(+d) 持到 X=140 → 离场　|　毛利 = y[140] − 2·y[F*]；76 月；未计成本",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(f"{D}/figs/fig52_四合约V型_出场140.png", dpi=135); plt.close(fig)
print("saved fig52")

# ---------- fig53: gross-return comparison bar ----------
fig, ax = plt.subplots(figsize=(10, 5.6))
labels = [NAME[c] for c in NAME]
x = np.arange(len(NAME)); w = 0.38
shorts = [rows[c]["short"] for c in NAME]
long_opt = [rows[c]["long_opt"] for c in NAME]
long_140 = [rows[c]["long140"] for c in NAME]
ax.bar(x - w / 2, shorts, w, color="#3b5b92", label="做空段（开盘→F*，吃下跌）")
ax.bar(x - w / 2, long_opt, w, bottom=shorts, color="#d6a24a", label="做多段（F*→最优X*，吃回升）")
ax.bar(x + w / 2, shorts, w, color="#3b5b92", alpha=0.45)
ax.bar(x + w / 2, long_140, w, bottom=shorts, color="#d6a24a", alpha=0.45,
       label="做多段（F*→X=140，多持到下一分钟）")
for i, c in enumerate(NAME):
    ax.text(x[i] - w / 2, rows[c]["ret_opt"] + 0.006, f"{rows[c]['ret_opt']:+.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.text(x[i] + w / 2, rows[c]["ret140"] + 0.006, f"{rows[c]['ret140']:+.3f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([f"{l}\n左=最优X*  右=X140" for l in labels], fontsize=9)
ax.axhline(0, color="0.5", lw=.7); ax.set_ylim(0, 0.40)
ax.set_ylabel("毛利（指数点 / 笔）= y[X] − 2·y[F*]", fontsize=10)
ax.set_title("图53　分钟内V型交易·毛利对比：最优出场 X* vs 出场=140（持到下一分钟）\n"
             "每合约左=最优X*、右=X=140；蓝=做空段，黄=做多段；76月；未计成本",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=8.5, loc="upper center", ncol=1, framealpha=0.95)
ax.grid(True, axis="y", alpha=0.25)
fig.tight_layout(); fig.savefig(f"{D}/figs/fig53_V型毛利对比.png", dpi=135); plt.close(fig)
print("saved fig53")
