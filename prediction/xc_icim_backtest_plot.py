"""xc_icim_backtest_plot.py — fig105: does IM's flow make money trading IC?
Left:  per-trade P&L vs threshold, own vs pair signal, five cost tiers (W=1s, 1s hold).
Right: cumulative NET profit (净利润, CNY, 1 lot, q=0.1, taker1+平昨) — own vs pair.
Run: python3 xc_icim_backtest_plot.py   (SUF=_pilot for pilot files)
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
r = pd.read_csv(f"{D}/xc_icim_bt_results{SUF}.csv")
dly = pd.read_csv(f"{D}/xc_icim_bt_daily{SUF}.csv")

TIER = [("mid0", "无摩擦 mid进出（信号价值）", "#2c7fb8", "-"),
        ("taker0", "吃单·同bin", "#7fcdbb", "--"),
        ("taker1", "吃单·延迟1bin", "0.5", "--"),
        ("fee_yz", "＋平昨手续费", "#e08a3c", ":"),
        ("fee_jt", "＋平今手续费", "#c0392b", ":")]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.6))

# left — per-trade by tier; pair=solid marker, own=hollow
for tier, lab, c, ls in TIER:
    sp = r[(r.signal == "pair") & (r.tier == tier)].sort_values("q")
    so = r[(r.signal == "own") & (r.tier == tier)].sort_values("q")
    ax1.plot(sp.q, sp.avg_ticks, marker="o", ms=6, lw=1.8, color=c, ls=ls, label=f"pair · {lab}")
    ax1.plot(so.q, so.avg_ticks, marker="o", ms=6, mfc="none", lw=1.0, color=c, ls=ls, alpha=.6)
ax1.axhline(0, color="k", lw=1)
ax1.set_xscale("log"); ax1.set_xticks([0.05, 0.1, 0.2, 0.5]); ax1.set_xticklabels(["0.05", "0.1", "0.2", "0.5"])
ax1.set_xlabel("信号阈值 q (tick)", fontsize=10); ax1.set_ylabel("每笔平均损益 (tick)", fontsize=10)
ax1.set_title("每笔损益：实心=pair(+IM)  空心=own\n信号价值 pair>own，但跨IC宽价差全砸到负", fontsize=11, fontweight="bold")
ax1.legend(fontsize=7.5, ncol=1); ax1.grid(True, alpha=.25)

# right — cumulative NET profit (净利润)
dly["day"] = pd.to_datetime(dly.gidday, unit="ns")
for sig, c in [("own", "0.5"), ("pair", "#c0392b")]:
    s = dly[dly.signal == sig].sort_values("day")
    if not len(s):
        continue
    mu = s.pnl_cny.mean(); shp = mu / s.pnl_cny.std() * np.sqrt(244) if s.pnl_cny.std() > 0 else np.nan
    ax2.plot(s.day, s.pnl_cny.cumsum() / 1e4, lw=1.8, color=c,
             label=f"{'pair(+IM)' if sig=='pair' else 'own(仅IC)'}  日均{mu:,.0f}元 Sharpe={shp:.1f}")
ax2.axhline(0, color="k", lw=1)
ax2.set_xlabel("日期", fontsize=10); ax2.set_ylabel("累计净利润（万元，单手）", fontsize=10)
ax2.set_title("净利润曲线（q=0.1；吃单+延迟+平昨费；2025-26样本外）\n加IM让信号更准，但净利润仍深度为负", fontsize=11, fontweight="bold")
ax2.legend(fontsize=9); ax2.grid(True, alpha=.3)

fig.suptitle("图105　用 IM 的订单流交易 IC：信号真的更准，但 taker 净利润仍是死刑", fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{D}/fig105_ICIM回测净利润{SUF}.png", dpi=135)
print("saved fig105")
