"""fig54_gross_opt.py — standalone GROSS-return bar for the fig51 case (optimal exit X*).
Stacked 做空段(-y[F*]) + 做多段(y[X*]-y[F*]) per contract. All gross (no costs).
Run: python3 fig54_gross_opt.py
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
c = pd.read_csv(f"{D}/step3_extcurve.csv")
g = c.groupby(["code", "tick"]).agg(sm=("sum", "sum"), n=("n", "sum")).reset_index()
g["mean"] = g["sm"] / g["n"]

R = {}
for code in NAME:
    s = g[g.code == code].sort_values("tick"); nmax = s["n"].max()
    cut = int(s.loc[s.n >= 0.9 * nmax, "tick"].max())
    y = s.set_index("tick")["mean"]
    F = int(y.loc[10:110].idxmin()); yF = y.loc[F]
    X = int(y.loc[F + 1:cut].idxmax()); yX = y.loc[X]
    R[code] = dict(F=F, X=X, short=-yF, long=yX - yF, tot=yX - 2 * yF)

fig, ax = plt.subplots(figsize=(10, 5.8))
x = np.arange(len(NAME))
shorts = [R[c]["short"] for c in NAME]
longs = [R[c]["long"] for c in NAME]
b1 = ax.bar(x, shorts, 0.55, color="#3b5b92", label="做空段（开盘 → F*，吃下跌）")
b2 = ax.bar(x, longs, 0.55, bottom=shorts, color="#d6a24a", label="做多段（F* → X*，吃回升）")
for i, code in enumerate(NAME):
    ax.text(i, shorts[i] / 2, f"{shorts[i]:+.3f}", ha="center", va="center", fontsize=9, color="w", fontweight="bold")
    ax.text(i, shorts[i] + longs[i] / 2, f"{longs[i]:+.3f}", ha="center", va="center", fontsize=9, color="#5a3a0c")
    ax.text(i, R[code]["tot"] + 0.008, f"合计 {R[code]['tot']:+.3f}", ha="center", fontsize=10.5, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f"{NAME[c]}\nF*={R[c]['F']}(~{R[c]['F']*0.5:.0f}s) · X*={R[c]['X']}(~{(R[c]['X']%120 if R[c]['X']>120 else R[c]['X'])*0.5:.0f}s)"
                    for c in NAME], fontsize=8.5)
ax.axhline(0, color="0.5", lw=.7); ax.set_ylim(0, max(R[c]["tot"] for c in NAME) * 1.18)
ax.set_ylabel("毛利（指数点 / 笔）= y[X*] − 2·y[F*]", fontsize=10)
ax.set_title("图54　分钟内V型交易·最优出场 X* 的毛利（对应 fig51）\n"
             "开盘做空吃下跌 + 谷底翻多吃回升；76月；阈值≥10ticks；未计成本",
             fontsize=12.5, fontweight="bold")
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, axis="y", alpha=0.25)
fig.tight_layout(); fig.savefig(f"{D}/figs/fig54_V型毛利_最优出场.png", dpi=135); plt.close(fig)
print("saved fig54")
for c in NAME:
    print("  %s: 做空%+.3f + 做多%+.3f = 合计 %+.3f 点/笔" % (NAME[c], R[c]["short"], R[c]["long"], R[c]["tot"]))
