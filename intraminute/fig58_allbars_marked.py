"""fig58_allbars_marked.py — fig47 data (ALL bars, MIN_TICKS=10), drawn full length but with
the line fading as coverage thins, and a marker on EACH line at the fig47 90%-coverage cutoff
(where fig47's solid line ended). Reads step2_twomin.csv. Run: python3 fig58_allbars_marked.py
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
g = pd.read_csv(f"{D}/step2_twomin.csv").groupby(["code", "tick"]).agg(
    sm=("sum", "sum"), n=("n", "sum")).reset_index()
g["mean"] = g["sm"] / g["n"]

fig, ax = plt.subplots(figsize=(11, 6.2))
ylo = yhi = 0.0
print("ALL bars (MIN_TICKS=10) — 每条线标出 fig47 的 90% 覆盖终点")
for code in NAME:
    s = g[g.code == code].sort_values("tick")
    tk = s["tick"].to_numpy(); y = s["mean"].to_numpy(); n = s["n"].to_numpy().astype(float)
    nmax = n.max(); cov = n / nmax
    cut90 = int(tk[cov >= .90].max())          # fig47 solid end (bold ends here)
    cut60 = int(tk[cov >= .60].max())          # dotted starts where cov drops below 60%
    cov240 = 100 * n[tk == 240][0] / nmax if (tk == 240).any() else 0.0
    # zone A: cov>=90 bold ; B: 60-90 medium ; C: <60 dotted (drawn to the end ~240)
    a = tk <= cut90
    b = (tk >= cut90) & (tk <= cut60)
    c = tk >= cut60
    ax.plot(tk[a], y[a], color=COL[code], lw=2.4, label=NAME[code])
    ax.plot(tk[b], y[b], color=COL[code], lw=1.3, alpha=0.6)
    ax.plot(tk[c], y[c], color=COL[code], lw=0.9, ls=":", alpha=0.4)
    # marker at fig47 cutoff
    yc = y[tk == cut90][0]
    ax.scatter([cut90], [yc], s=90, color=COL[code], ec="k", lw=.8, zorder=6)
    ax.annotate(f"fig47止于 tick{cut90}\n(90%覆盖)", (cut90, yc), textcoords="offset points",
                xytext=(6, 10 if code != "IM0000" else -26), fontsize=8, color=COL[code], fontweight="bold")
    ylo = min(ylo, y[tk <= cut60].min()); yhi = max(yhi, y[tk <= cut60].max())
    print(f"  {NAME[code]}: 90%覆盖→tick{cut90} | 60%(点线起)→tick{cut60} | 到tick240的覆盖={cov240:.1f}%")
pad = (yhi - ylo) * 0.18
ax.set_ylim(ylo - pad, yhi + pad); ax.set_xlim(0, 235)
ax.axhline(0, color="0.55", lw=.7)
ax.axvline(200, color="#888", lw=1.0, ls="-"); ax.text(201, yhi, " tick200", fontsize=8, color="#666")
ax.set_xlabel("2分钟bar内 tick 序号（全部bar；●=fig47的90%覆盖终点；粗实线≥90% · 细线60–90% · 点线<60%覆盖）", fontsize=9.5)
ax.set_ylabel("d·(价格 − bar开盘)（指数点）", fontsize=10)
ax.set_title("图58　2分钟bar动量曲线·全部bar全程画出，但随样本变薄而淡化\n"
             "●标出 fig47 各线 90% 覆盖的终点(=之前转虚线处)；76月", fontsize=12, fontweight="bold")
ax.legend(fontsize=9, loc="lower left"); ax.grid(True, alpha=0.25)
fig.tight_layout(); fig.savefig(f"{D}/figs/fig58_全bar_标注fig47终点.png", dpi=135); plt.close(fig)
print("saved fig58")
