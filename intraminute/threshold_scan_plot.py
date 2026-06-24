"""threshold_scan_plot.py — mean return vs tick threshold (L>=T), step1 & step3.
Reads threshold_scan.csv (per-contract L-bucketed sums). For each threshold T we pool
all buckets with Lbucket>=T -> mean, t, n. Run: python3 threshold_scan_plot.py
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
THR = list(range(10, 130, 10))   # thresholds T to evaluate (L>=T)

d = pd.read_csv(f"{D}/threshold_scan.csv")


def cum(sub, pre):
    """for each threshold T, pool buckets Lbucket>=T."""
    out = []
    for T in THR:
        s = sub[sub.Lbucket >= T]
        n = s[f"{pre}_n"].sum()
        if n < 50:
            out.append((T, np.nan, np.nan, n)); continue
        sm = s[f"{pre}_sum"].sum(); ss = s[f"{pre}_ss"].sum()
        mean = sm / n; sd = np.sqrt(max(ss / n - mean ** 2, 0))
        out.append((T, mean, mean / (sd / np.sqrt(n)), n))
    return pd.DataFrame(out, columns=["T", "mean", "t", "n"])


fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
titles = {"s1": "步骤1  (S=0 满分钟动量, <0=反转)", "s3": "步骤3  (V型: 做空→做多, >0=毛利)"}
for ax, pre in zip(axes, ["s1", "s3"]):
    print(f"\n=== {titles[pre]} :  mean return  vs  threshold L>=T  (t in (), n) ===")
    for code in NAME:
        r = cum(d[d.code == code], pre)
        ax.plot(r["T"], r["mean"], color=COL[code], lw=1.9, marker="o", ms=4, label=NAME[code])
        best = r.loc[r["mean"].abs().idxmax()] if pre == "s1" else r.loc[r["mean"].idxmax()]
        print("  %s:  " % NAME[code] + "  ".join(
            f"T{int(x['T'])}={x['mean']:+.3f}(t{x['t']:+.1f})" for _, x in r.iterrows()))
        print("      -> 最优 T=%d  mean=%+.3f  t=%.1f  n=%d" % (best["T"], best["mean"], best["t"], best["n"]))
    ax.axhline(0, color="0.55", lw=.7)
    ax.set_xlabel("tick 阈值 T（只保留该分钟 tick 数 L≥T 的交易）", fontsize=10)
    ax.set_ylabel("平均毛利（指数点 / 笔）", fontsize=10)
    ax.set_title(titles[pre], fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
fig.suptitle("图55　提高 tick 阈值 L≥T 对收益的影响（76月；横轴右移=只做更活跃的分钟）",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(f"{D}/figs/fig55_tick阈值扫描.png", dpi=135); plt.close(fig)
print("\nsaved fig55")
