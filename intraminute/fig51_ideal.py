"""fig51_ideal.py — per-contract V-trade at the tick-resolution ideal (F*,X*).
Momentum-normalised frame: curve y = d*(price-open), so prior-minute direction points UP.
In this frame the trade is literally: 做空(−d) from open->F* (capture the fall), then
做多(+d) from F*->X* (capture the recovery). Return = y[X*] - 2*y[F*].
Run: python3 fig51_ideal.py
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
SHORT_BG = "#dCe6f2"   # 做空段底色（中性蓝灰）
LONG_BG = "#f3e7d3"    # 做多段底色（中性米黄）

c = pd.read_csv(f"{D}/step3_extcurve.csv")
g = c.groupby(["code", "tick"]).agg(sm=("sum", "sum"), n=("n", "sum")).reset_index()
g["mean"] = g["sm"] / g["n"]

fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
for ax, code in zip(axes.ravel(), NAME):
    s = g[g.code == code].sort_values("tick")
    nmax = s["n"].max(); cut = int(s.loc[s.n >= 0.9 * nmax, "tick"].max())
    tk = s["tick"].to_numpy(); y = s["mean"].to_numpy()
    ys = pd.Series(y, index=tk)
    F = int(ys.loc[10:110].idxmin()); yF = ys.loc[F]
    X = int(ys.loc[F + 1:cut].idxmax()); yX = ys.loc[X]
    ret = yX - 2 * yF; short_pnl = -yF; long_pnl = yX - yF
    sld = tk <= cut
    yr = y[sld]; lo, hi = yr.min(), yr.max(); pad = (hi - lo) * 0.42 + 0.01
    ylo, yhi = lo - pad, hi + pad
    # 两段持仓底色
    ax.axvspan(0, F, color=SHORT_BG, alpha=0.9, zorder=0)
    ax.axvspan(F, X, color=LONG_BG, alpha=0.9, zorder=0)
    # 曲线
    ax.plot(tk[sld], y[sld], color=COL[code], lw=2.2, zorder=4)
    ax.plot(tk[tk >= cut], y[tk >= cut], color=COL[code], lw=0.9, ls="--", alpha=0.3, zorder=4)
    # 翻仓点 / 出场点
    ax.scatter([F], [yF], color=COL[code], marker="v", s=120, zorder=6, ec="k", lw=.6)
    ax.scatter([X], [yX], color=COL[code], marker="^", s=120, zorder=6, ec="k", lw=.6)
    ax.annotate(f"翻仓 F*={F}（~{F*0.5:.0f}s）\n平空·反手做多", (F, yF),
                textcoords="offset points", xytext=(6, -28), fontsize=8.5)
    ax.annotate(f"出场 X*={X}（~{(X%120 if X>120 else X)*0.5:.0f}s{'·下一分钟' if X>120 else ''}）\n平多·离场",
                (X, yX), textcoords="offset points", xytext=(-6, 10), fontsize=8.5, ha="right")
    # 段标签
    ax.text(F / 2, yhi - (yhi - ylo) * 0.06, "① 做空 (−d)\n吃下跌段", ha="center", va="top",
            fontsize=9.5, fontweight="bold", color="#2c3e6b")
    ax.text((F + X) / 2, yhi - (yhi - ylo) * 0.06, "② 做多 (+d)\n吃回升段", ha="center", va="top",
            fontsize=9.5, fontweight="bold", color="#8a5a1c")
    ax.axhline(0, color="0.45", lw=.7); ax.axvline(120, color="0.45", lw=1.0, ls=":")
    ax.text(120, ylo + (yhi - ylo) * 0.02, " 分钟收盘", fontsize=7.5, color="0.4")
    ax.set_ylim(ylo, yhi); ax.set_xlim(0, max(cut + 8, 130))
    ax.set_xlabel("从分钟开盘起的 tick 序号（1 tick ≈ 0.5 秒；120 = 分钟收盘）", fontsize=9)
    ax.set_ylabel("动量框架位移  y = d·(价格 − 开盘)  （指数点）", fontsize=9)
    ax.set_title("%s    毛利 = %+.3f 点/笔   ( 做空段 %+.3f  +  做多段 %+.3f )"
                 % (NAME[code], ret, short_pnl, long_pnl), fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.22, zorder=1)
fig.suptitle("图51　四合约·分钟内 V 型交易的 tick 级最优（动量框架：已按上一分钟方向归一，故曲线先跌后回升）\n"
             "做法：开盘 → ① 做空(−d) 吃下跌 → 谷底 F* 翻仓 → ② 做多(+d) 吃回升 → X* 离场　|　毛利 = y[X*] − 2·y[F*]；76 月；阈值≥10ticks；未计成本",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(f"{D}/figs/fig51_四合约最优V型.png", dpi=135); plt.close(fig)
print("saved fig51")
