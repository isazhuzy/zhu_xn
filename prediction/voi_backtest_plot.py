"""voi_backtest_plot.py — figures for the threshold-strategy backtest.
fig84: per-trade P&L (ticks) vs threshold, by execution/cost tier.
fig85: cumulative daily P&L (CNY, 1 lot, q=0.5, taker1+平昨 fees).
Run: python3 voi_backtest_plot.py   (SUF=_pilot for pilot files)
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
NAME = {"IC0000": "IC 中证500", "IF0000": "IF 沪深300", "IH0000": "IH 上证50", "IM0000": "IM 中证1000"}
COL = {"IC0000": "#c0392b", "IF0000": "#e08a3c", "IH0000": "#27ae60", "IM0000": "#4C72B0"}
res = pd.read_csv(f"{D}/bt_results{SUF}.csv"); dly = pd.read_csv(f"{D}/bt_daily{SUF}.csv")

TIER = [("mid0", "无摩擦（mid进出）＝信号价值", "#2c7fb8", "-"),
        ("taker0", "吃单·同快照", "#7fcdbb", "--"),
        ("taker1", "吃单·延迟500ms", "0.45", "--"),
        ("fee_yz", "＋手续费(开0.23+平昨0.23bp)", "#e08a3c", ":"),
        ("fee_jt", "＋手续费(开0.23+平今3.45bp)", "#c0392b", ":")]

fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
for ax, code in zip(axes.ravel(), NAME):
    for tier, lab, c, ls in TIER:
        s = res[(res.code == code) & (res.tier == tier)].sort_values("q")
        ax.plot(s.q, s.avg_ticks, marker="o", ms=4, lw=1.7, color=c, ls=ls, label=lab)
    ax.axhline(0, color="k", lw=1)
    ax.set_xscale("log"); ax.set_xticks([0.1, 0.2, 0.5, 1, 2]); ax.set_xticklabels(["0.1", "0.2", "0.5", "1", "2"])
    n5 = res[(res.code == code) & (res.tier == "mid0") & (res.q == 0.5)]
    ttl = NAME[code] + (f"   (q=0.5: {int(n5.n_trades.iloc[0]):,}笔)" if len(n5) else "")
    ax.set_title(ttl, fontsize=10.5, fontweight="bold")
    ax.set_xlabel("信号阈值 q (tick)", fontsize=9); ax.set_ylabel("每笔平均损益 (tick)", fontsize=9)
    ax.legend(fontsize=7.5); ax.grid(True, alpha=.25)
fig.suptitle("图84　阈值策略每笔损益：信号价值为正，但跨价差+手续费把它埋掉\n"
             "（模型B k=20 冻结系数；2025-26 样本外；非重叠持仓10s）",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(f"{D}/fig84_回测每笔损益{SUF}.png", dpi=135); plt.close(fig)

dly["day"] = pd.to_datetime(dly.day)
fig, ax = plt.subplots(figsize=(12, 6))
for code in NAME:
    s = dly[dly.code == code].sort_values("day")
    if not len(s):
        continue
    mu, sd = s.pnl_cny.mean(), s.pnl_cny.std()
    shp = mu / sd * np.sqrt(244) if sd > 0 else np.nan
    ax.plot(s.day, s.pnl_cny.cumsum() / 1e4, color=COL[code], lw=1.6,
            label=f"{NAME[code]}  日均{mu:,.0f}元  Sharpe={shp:.1f}")
ax.axhline(0, color="k", lw=1)
ax.set_xlabel("日期", fontsize=11); ax.set_ylabel("累计损益（万元，单手）", fontsize=11)
ax.set_title("图85　净损益曲线（q=0.5；吃单+延迟500ms+平昨费率；2025-26 样本外）",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=.3)
fig.tight_layout(); fig.savefig(f"{D}/fig85_回测累计损益{SUF}.png", dpi=135); plt.close(fig)
print("saved fig84, fig85")
