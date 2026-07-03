"""ofi_plot.py — figures for the OFI analysis. Run: python3 ofi_plot.py
 fig61: binned dP vs OFI (10s buckets) per contract, with fitted line, R2, D.
 fig62: R2 vs interval length — OFI (contemporaneous) vs TI vs OFI(predictive).
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

D = "/Users/zhuisabella/xn/prediction"
NAME = {"IC0000": "IC 中证500", "IF0000": "IF 沪深300", "IH0000": "IH 上证50", "IM0000": "IM 中证1000"}
COL = {"IC0000": "#c0392b", "IF0000": "#e08a3c", "IH0000": "#27ae60", "IM0000": "#4C72B0"}
res = pd.read_csv(f"{D}/ofi_results.csv")
sc = pd.read_csv(f"{D}/ofi_scatter.csv")

# fig61: binned scatter dP vs OFI (10s)
fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
for ax, code in zip(axes.ravel(), NAME):
    s = sc[sc.code == code]
    r = res[(res.code == code) & (res.B == 20)].iloc[0]
    ax.scatter(s["OFI"], s["dP"], s=18, color=COL[code], alpha=0.8)
    xs = np.linspace(s["OFI"].min(), s["OFI"].max(), 50)
    ax.plot(xs, xs * r["coef_ofi"], color="k", lw=1.3, ls="--")
    ax.axhline(0, color="0.6", lw=.6); ax.axvline(0, color="0.6", lw=.6)
    ax.set_title(f"{NAME[code]}   R²={r['r2_ofi']:.2f}  D≈{r['D']:.0f}  斜率=1/2D",
                 fontsize=10.5, fontweight="bold")
    ax.set_xlabel("OFI（10s 净订单流不平衡）", fontsize=9)
    ax.set_ylabel("ΔP（10s 中间价变动，tick）", fontsize=9)
    ax.grid(True, alpha=0.25)
fig.suptitle("图61　订单流不平衡 OFI 同期解释中间价变动（10s；2024-06；分箱均值）\n"
             "ΔP = OFI/(2D) + ε —— 近乎线性、无参数（斜率即 1/2D）", fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(f"{D}/fig61_OFI同期散点.png", dpi=135); plt.close(fig)
print("saved fig61")

# fig62: R2 vs interval — OFI vs TI vs predictive
fig, ax = plt.subplots(figsize=(10, 6))
for code in NAME:
    s = res[res.code == code].sort_values("secs")
    ax.plot(s["secs"], s["r2_ofi"], color=COL[code], lw=2, marker="o", label=f"{NAME[code]} · OFI同期")
    ax.plot(s["secs"], s["r2_ti"], color=COL[code], lw=1.3, ls="--", marker="s", ms=4, alpha=0.7)
    ax.plot(s["secs"], s["r2_pred"], color=COL[code], lw=1.0, ls=":", marker="^", ms=4, alpha=0.6)
ax.set_xscale("log"); ax.set_xticks([0.5, 2, 10, 60]); ax.set_xticklabels(["0.5s", "2s", "10s", "60s"])
ax.set_xlabel("聚合区间长度", fontsize=11)
ax.set_ylabel("R²", fontsize=11)
ax.set_title("图62　R² 对比：OFI同期(实线) vs 交易不平衡TI(虚线) vs OFI预测下一区间(点线)\n"
             "2024-06；区间越长同期R²越高；预测R²≈0", fontsize=12, fontweight="bold")
ax.legend(fontsize=8.5, loc="center left"); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(f"{D}/fig62_R2对比.png", dpi=135); plt.close(fig)
print("saved fig62")
