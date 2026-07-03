"""ofi_full_plot.py — full-window (2020-2026) figures. Run: python3 ofi_full_plot.py
 fig63: binned dP vs OFI (10s) per contract.  fig64: R2 vs interval (OFI/TI/pred).
 fig65: per-month R2(OFI,10s) over 2020-2026 (stability).
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
res = pd.read_csv(f"{D}/ofi_results_full.csv"); sc = pd.read_csv(f"{D}/ofi_scatter_full.csv")
mon = pd.read_csv(f"{D}/ofi_permonth.csv")

# fig63 scatter
fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
for ax, code in zip(axes.ravel(), NAME):
    s = sc[sc.code == code].sort_values("OFI"); r = res[(res.code == code) & (res.B == 20)].iloc[0]
    ax.scatter(s["OFI"], s["dP"], s=14, color=COL[code], alpha=0.7)
    xs = np.linspace(s["OFI"].min(), s["OFI"].max(), 50); ax.plot(xs, xs * r["coef_ofi"], "k--", lw=1.2)
    ax.axhline(0, color="0.6", lw=.6); ax.axvline(0, color="0.6", lw=.6)
    ax.set_title(f"{NAME[code]}   R²={r['r2_ofi']:.2f}  D≈{r['D']:.1f}", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("OFI (10s)", fontsize=9); ax.set_ylabel("ΔP (tick,10s)", fontsize=9); ax.grid(True, alpha=0.25)
fig.suptitle("图63　OFI 同期解释中间价变动（10s；全样本 2020-2026；分箱均值）\nΔP = OFI/(2D) + ε",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(f"{D}/fig63_OFI同期散点_全样本.png", dpi=135); plt.close(fig)

# fig64 R2 vs interval
fig, ax = plt.subplots(figsize=(10, 6))
for code in NAME:
    s = res[res.code == code].sort_values("secs")
    ax.plot(s.secs, s.r2_ofi, color=COL[code], lw=2, marker="o", label=f"{NAME[code]} OFI同期")
    ax.plot(s.secs, s.r2_ti, color=COL[code], lw=1.2, ls="--", marker="s", ms=4, alpha=0.7)
    ax.plot(s.secs, s.r2_pred, color=COL[code], lw=1.0, ls=":", marker="^", ms=4, alpha=0.6)
ax.set_xscale("log"); ax.set_xticks([0.5, 2, 10, 60]); ax.set_xticklabels(["0.5s", "2s", "10s", "60s"])
ax.set_xlabel("聚合区间", fontsize=11); ax.set_ylabel("R²", fontsize=11)
ax.set_title("图64　R²：OFI同期(实线) vs 交易不平衡TI(虚线) vs OFI预测下一区间(点线)\n全样本 2020-2026",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=8.5, loc="center left"); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(f"{D}/fig64_R2对比_全样本.png", dpi=135); plt.close(fig)

# fig65 per-month stability
mon = mon.dropna(subset=["r2_ofi"]).copy()
mon["t"] = mon.year + (mon.month - 1) / 12.0
fig, ax = plt.subplots(figsize=(12, 5.5))
for code in NAME:
    s = mon[mon.code == code].sort_values("t")
    ax.plot(s.t, s.r2_ofi, color=COL[code], lw=1.5, marker="o", ms=3, label=NAME[code])
ax.set_ylim(0, 0.85); ax.set_xlabel("年份", fontsize=11); ax.set_ylabel("R²(OFI, 10s)", fontsize=11)
ax.axhline(mon.r2_ofi.mean(), color="0.5", ls="--", lw=1, label=f"均值 {mon.r2_ofi.mean():.2f}")
ax.set_title("图65　OFI同期R²(10s) 的逐月稳定性（2020-2026）—— 关系跨时间稳定", fontsize=12, fontweight="bold")
ax.legend(fontsize=9, ncol=5, loc="lower center"); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(f"{D}/fig65_R2逐月稳定性.png", dpi=135); plt.close(fig)
print("saved fig63, fig64, fig65")
