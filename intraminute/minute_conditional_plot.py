"""minute_conditional_plot.py — per-contract momentum P&L by prior-move-size bucket,
in-sample vs out-of-sample. If conditioning worked, the IS pattern would repeat OOS.
Run: python3 minute_conditional_plot.py"""
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

OUT = "/Users/zhuisabella/xn/intraminute/figs"
NAME = {"IC0000": "IC 中证500", "IF0000": "IF 沪深300", "IH0000": "IH 上证50", "IM0000": "IM 中证1000"}
df = pd.read_csv("/Users/zhuisabella/xn/intraminute/minute_conditional_isoos.csv")
df["pnl"] = np.sign(df.dprev) * df.ret
df["absd"] = df.dprev.abs()
pct = df.groupby(["sample", "code", "ym"])["absd"].rank(pct=True)
df["Q"] = np.ceil(pct * 5).clip(1, 5).astype(int)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
qs = [1, 2, 3, 4, 5]
for ax, code in zip(axes.ravel(), NAME):
    isv = [df[(df.code == code) & (df["sample"] == "IS") & (df.Q == q)]["pnl"].mean() for q in qs]
    oos = [df[(df.code == code) & (df["sample"] == "OOS") & (df.Q == q)]["pnl"].mean() for q in qs]
    x = np.arange(5)
    ax.bar(x - 0.2, isv, 0.4, color="#4C72B0", label="样本内 IS")
    ax.bar(x + 0.2, oos, 0.4, color="#c0392b", label="样本外 OOS")
    ax.axhline(0, color="0.4", lw=.7)
    ax.set_xticks(x); ax.set_xticklabels(["Q1\n小动", "Q2", "Q3", "Q4", "Q5\n大动"], fontsize=8)
    ax.set_title(NAME[code], fontsize=12, fontweight="bold")
    ax.set_ylabel("动量P&L 均值 (指数点/分钟)", fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.25)
fig.suptitle("图33　条件化:动量P&L 按上一分钟波幅分桶 · 样本内 vs 样本外（6+6个平静月）",
             fontsize=14, fontweight="bold")
fig.text(0.5, 0.005, "假设:上一分钟动得越大→反转越强。若成立,IS(蓝)的形态应在OOS(红)重现。实际OOS几乎全部塌到≈0,"
         "IS里的形态(如IC的Q1负/Q5正)样本外不复现→条件化也无稳健edge。", ha="center", fontsize=8.5, color="0.4")
fig.tight_layout(rect=(0, 0.03, 1, 0.96))
fig.savefig(f"{OUT}/fig33_条件化_波幅分桶.png", dpi=130); plt.close(fig)
print("saved fig33")
