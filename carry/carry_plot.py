"""carry_plot.py — fig201-204 for the basis-carry study (see carry_ddb.py).
fig201 贴水历史: term-structure slope carry (1↔2), 60d MA, per product.
fig202 实现carry累计: cumulative realized roll-cycle carry per product (+ann/t stats).
fig203 分红季节性: mean realized carry by expiry calendar month.
fig204 入场时机: entry basis vs realized carry; deep-vs-shallow split table printed.
Run: python3 carry_plot.py
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
D = "/Users/zhuisabella/xn/carry"
NM = {"IC": "IC 中证500", "IF": "IF 沪深300", "IH": "IH 上证50", "IM": "IM 中证1000"}
CL = {"IC": "#d62728", "IF": "#1f77b4", "IH": "#2ca02c", "IM": "#9467bd"}

slope = pd.read_csv(f"{D}/carry_slope.csv", parse_dates=["date"])
real = pd.read_csv(f"{D}/carry_realized.csv", parse_dates=["entry_date", "exit_date"])

# fig201 — slope carry history
fig, ax = plt.subplots(figsize=(13, 5))
for p in NM:
    s = slope[(slope["product"] == p) & (slope.pair == "1-2")].set_index("date")["ann_pct"]
    if not len(s):
        continue
    ax.plot(s.rolling(60, min_periods=20).mean(), lw=1.2, color=CL[p], label=NM[p])
ax.axhline(0, color="gray", lw=0.8)
ax.set_ylabel("年化 slope carry (%)  =(F近−F远)/F远·年化")
ax.set_title("fig201 — 股指期货期限结构 carry（近月↔次月，60日均线）：正=贴水结构=做多远月吃升水")
ax.legend(); fig.tight_layout(); fig.savefig(f"{D}/fig201_贴水历史.png", dpi=130)

# fig202 — cumulative realized carry
fig, ax = plt.subplots(figsize=(13, 5))
for p in NM:
    r = real[real["product"] == p].sort_values("exit_date")
    if not len(r):
        continue
    ann = r.carry_pct.mean() * 12
    t = r.carry_pct.mean() / (r.carry_pct.std() / np.sqrt(len(r)))
    ax.plot(r.exit_date, r.carry_pct.cumsum(), lw=1.4, color=CL[p],
            label=f"{NM[p]}  年化{ann:+.1f}%  t={t:.1f}")
ax.axhline(0, color="gray", lw=0.8)
ax.set_ylabel("累计实现 carry (%)（期货滚动 − 现货锚）")
ax.set_title("fig202 — 实现的吃贴水收益：每月滚动持有次月合约至交割，相对现货指数的超额")
ax.legend(); fig.tight_layout(); fig.savefig(f"{D}/fig202_实现carry累计.png", dpi=130)

# fig203 — seasonality by expiry month
fig, ax = plt.subplots(figsize=(11, 4.5))
wd = 0.2
for i, p in enumerate(NM):
    r = real[real["product"] == p]
    m = r.groupby("exp_month")["carry_pct"].mean()
    ax.bar(np.array(m.index) + (i - 1.5) * wd, m.values, width=wd, color=CL[p], label=NM[p])
ax.axhline(0, color="gray", lw=0.8)
ax.set_xticks(range(1, 13)); ax.set_xlabel("到期月份"); ax.set_ylabel("平均实现 carry (%/滚动月)")
ax.set_title("fig203 — carry 的分红季节性：5–8月到期的窗口 carry 明显更高（那部分=分红补偿，非纯风险溢价）")
ax.legend(); fig.tight_layout(); fig.savefig(f"{D}/fig203_分红季节性.png", dpi=130)

# fig204 — entry basis vs realized carry + timing split
fig, ax = plt.subplots(figsize=(7.5, 6))
for p in NM:
    r = real[real["product"] == p]
    ax.scatter(r.entry_basis_pct, r.carry_pct, s=14, alpha=0.6, color=CL[p], label=NM[p])
lim = np.array([-6, 3])
ax.plot(lim, -lim, "k--", lw=0.8, label="机械换算线 carry=−entry basis")
ax.set_xlabel("入场基差 (次月F − 现货锚)/现货锚 (%)"); ax.set_ylabel("该窗口实现 carry (%)")
ax.set_title("fig204 — 入场贴水深度 vs 实现carry")
ax.legend(); fig.tight_layout(); fig.savefig(f"{D}/fig204_入场时机.png", dpi=130)

print("=== per-year annualized realized carry (%) ===")
real["yr"] = real.exit_date.dt.year
print(real.pivot_table(index="yr", columns="product", values="carry_pct",
                       aggfunc=lambda s: s.mean() * 12).round(1).to_string())
print("\n=== timing: split by entry basis (per-product expanding median) ===")
for p in NM:
    r = real[real["product"] == p].sort_values("entry_date").copy()
    med = r.entry_basis_pct.expanding(12).median().shift(1)
    deep = r.carry_pct[r.entry_basis_pct < med]; shal = r.carry_pct[r.entry_basis_pct >= med]
    if len(deep) > 5:
        print(f"{p}: deep-discount entries mean {deep.mean():+.2f}%/mo (n={len(deep)})   "
              f"shallow {shal.mean():+.2f}%/mo (n={len(shal)})")
print("\n=== correlation(entry basis, realized carry) ===")
for p in NM:
    r = real[real["product"] == p]
    print(p, f"{np.corrcoef(r.entry_basis_pct, r.carry_pct)[0,1]:+.2f}")
