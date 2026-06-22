"""
subminute_demo.py — sub-minute entry-timing demo on the open (38-day sample).

Idea: a minute ≈ 120 ticks (~2 snapshots/sec). Watch the FIRST N ticks of
minute 1, FADE that move (up -> short, down -> long), enter at the Nth tick,
and hold to a chosen exit. Sweep the entry tick N (8 / 10 / 12 / ...).

  signal(N) = mid[tick N] - mid[tick 1]      # move over the first N ticks
  pos       = -sign(signal)                  # reversal / fade
  entry     = mid[tick N]
  exit_min1 = mid[last tick of 09:30–09:31]  (≈ 09:31 close)
  exit_min2 = mid[last tick of 09:31–09:32]  (≈ 09:32 close)
  pnl_bp    = pos * (exit/entry - 1) * 1e4

Mid price = (bid+ask)/2; gross (mid-to-mid, no cost). N is the Nth AVAILABLE
tick (days have 110–120), so it is robust to the tick-count wobble.
Note: N≈120 with exit_min2 reproduces the original 09:31 contrarian trade.

Run:  /usr/local/bin/python3.14 subminute_demo.py
"""
import os
import datetime as dt
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

SRC = "/Users/zhuisabella/xn/ticker/open_breakdown/open_ticks_2months.csv"
OUT = "/Users/zhuisabella/xn/ticker/figs_conclusion"
PRODUCTS = ["IC0000", "IF0000", "IH0000", "IM0000"]
NAME = {"IC0000": "IC 中证500", "IF0000": "IF 沪深300",
        "IH0000": "IH 上证50", "IM0000": "IM 中证1000"}
COL = {"IC0000": "#c0392b", "IF0000": "#e08a3c", "IH0000": "#27ae60", "IM0000": "#4C72B0"}
N_LIST = [2, 4, 6, 8, 10, 12, 15, 20, 30, 45, 60, 90, 118]


def build_records(df):
    """(code, day) -> dict(m1=mid array of minute 1, ex1, ex2)."""
    df = df.copy()
    df["mid"] = (df["m_nBidPrice"] + df["m_nAskPrice"]) / 2.0
    df["day"] = df["m_nDatetime"].dt.normalize()
    df["t"] = df["m_nDatetime"].dt.time
    recs = {}
    for (code, day), g in df.groupby(["code", "day"], sort=False):
        g = g.sort_values("m_nDatetime")
        t = g["t"].values
        mid = g["mid"].values
        in1 = (t >= dt.time(9, 30)) & (t < dt.time(9, 31))
        in2 = (t >= dt.time(9, 31)) & (t < dt.time(9, 32))
        m1, m2 = mid[in1], mid[in2]
        if len(m1) < 3 or len(m2) < 1:
            continue
        recs[(code, day)] = {"m1": m1, "ex1": m1[-1], "ex2": m2[-1]}
    return recs


def stats(pnls):
    a = np.asarray(pnls, float)
    n = len(a)
    if n == 0:
        return None
    t = a.mean() / (a.std(ddof=1) / np.sqrt(n)) if n > 1 and a.std(ddof=1) > 0 else np.nan
    return dict(n=n, win=round((a > 0).mean(), 3), mean_bp=round(a.mean(), 2),
                total_bp=round(a.sum(), 1), t=round(t, 2))


def sweep(recs):
    rows = []
    for code in PRODUCTS:
        keys = [k for k in recs if k[0] == code]
        for N in N_LIST:
            p1, p2 = [], []
            for k in keys:
                r = recs[k]; m1 = r["m1"]
                if len(m1) < N:
                    continue
                entry = m1[N - 1]
                sig = m1[N - 1] - m1[0]
                if sig == 0:
                    continue
                pos = -np.sign(sig)
                if N < len(m1):                       # a later tick exists in min1
                    p1.append(pos * (r["ex1"] / entry - 1) * 1e4)
                p2.append(pos * (r["ex2"] / entry - 1) * 1e4)
            for exit_lbl, p in [("到09:31收盘", p1), ("到09:32收盘", p2)]:
                st = stats(p)
                if st:
                    rows.append({"product": code, "entry_tick": N,
                                 "exit": exit_lbl, **st})
    return pd.DataFrame(rows)


def plot(tab):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), sharex=True)
    exits = ["到09:31收盘", "到09:32收盘"]
    for j, ex in enumerate(exits):
        for p in PRODUCTS:
            s = tab[(tab["product"] == p) & (tab["exit"] == ex)].sort_values("entry_tick")
            axes[0, j].plot(s["entry_tick"], s["mean_bp"], "-o", color=COL[p], ms=5, label=NAME[p])
            axes[1, j].plot(s["entry_tick"], s["win"] * 100, "-o", color=COL[p], ms=5, label=NAME[p])
        axes[0, j].axhline(0, color="k", lw=.8)
        axes[1, j].axhline(50, color="k", lw=1, ls=":")
        axes[0, j].set_title(f"持有 {ex}", fontsize=12)
        axes[0, j].grid(alpha=.3); axes[1, j].grid(alpha=.3)
        axes[1, j].set_xlabel("进场 tick 序号 N（约 N/2 秒；一分钟≈120 ticks）")
    axes[0, 0].set_ylabel("每笔平均收益（bp，毛）")
    axes[1, 0].set_ylabel("逆势胜率（%）")
    axes[0, 1].legend(fontsize=8.5, loc="upper right")
    fig.suptitle("图7　开盘前 N 个 tick 反转择时（38天演示）：进场 tick 序号 N 的扫描",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, 0.01,
             "信号=前N个tick的涨跌方向、反向开仓(涨则空/跌则多)，mid成交、毛收益；"
             "38天小样本仅作机制演示，最佳N不代表可外推。",
             ha="center", fontsize=8.5, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(f"{OUT}/fig7_subminute_entry_demo.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(SRC, dtype={"code": "string"}, parse_dates=["m_nDatetime"])
    recs = build_records(df)
    tab = sweep(recs)
    tab.to_csv(f"{OUT.replace('figs_conclusion','open_breakdown')}/subminute_demo.csv", index=False)
    pd.set_option("display.width", 200, "display.max_rows", 400)
    for ex in ["到09:31收盘", "到09:32收盘"]:
        print(f"\n===== 进场tick扫描 | 持有{ex} | 反转 | 38天 | 每笔bp & 胜率 =====")
        print(tab[tab.exit == ex].drop(columns="exit").to_string(index=False))
    plot(tab)
    print("\nsaved fig ->", f"{OUT}/fig7_subminute_entry_demo.png")
