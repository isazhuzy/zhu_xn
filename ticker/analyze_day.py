"""
analyze_day.py — whole-day momentum/reversal profile for the 38-day window.

Momentum signal at every minute (session-aware, same as matrix.py):
  sig=close-prev, fwd=next/close-1, pos=sign(sig), r=pos*fwd if |sig|>thr*0.2
  r>0  => momentum (trend continuation) made money
  r<0  => reversal (the move faded)   <- the open's 09:31 is the extreme case

Outputs (Chinese-labelled):
  fig8  intraday heatmap: 合约 × 30分钟时段, 动量每笔平均收益 (bp)
  fig9  平均日内累计动量收益曲线 (per-minute, with lunch gap)
  open_breakdown/day_buckets.csv

Run:  /usr/local/bin/python3.14 analyze_day.py
"""
import os
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

SRC = "/Users/zhuisabella/xn/ticker/open_breakdown/day_bars_2months.csv"
OUT = "/Users/zhuisabella/xn/ticker/figs_conclusion"
PRODUCTS = ["IC0000", "IF0000", "IH0000", "IM0000"]
NAME = {"IC0000": "IC 中证500", "IF0000": "IF 沪深300",
        "IH0000": "IH 上证50", "IM0000": "IM 中证1000"}
COL = {"IC0000": "#c0392b", "IF0000": "#e08a3c", "IH0000": "#27ae60", "IM0000": "#4C72B0"}
TICK, THR = 0.2, 10
AM_END, PM_START = 11 * 60 + 30, 13 * 60      # minutes-since-midnight


def build(path, thr=THR):
    df = pd.read_csv(path)
    df["d"] = pd.to_datetime(df["d"])
    ts = pd.to_datetime(df["mn"])
    df["hhmm"] = ts.dt.hour * 60 + ts.dt.minute
    df["tod"] = ts.dt.strftime("%H:%M")
    df = df[(df["hhmm"] <= AM_END) | (df["hhmm"] >= PM_START)]      # drop any lunch stragglers
    df["session"] = np.where(df["hhmm"] <= AM_END, "AM", "PM")
    df = df.sort_values(["code", "d", "hhmm"])
    g = df.groupby(["code", "d", "session"])
    df["prev"] = g["close"].shift(1)
    df["next"] = g["close"].shift(-1)
    sig = df["close"] - df["prev"]
    fwd = df["next"] / df["close"] - 1.0
    keep = sig.abs() > thr * TICK
    df["r"] = np.where(keep, np.sign(sig) * fwd, 0.0)
    df["active"] = keep & df["next"].notna()
    return df


def bucket_table(df):
    df = df.copy()
    df["bucket"] = (df["hhmm"] // 30 * 30).map(lambda m: f"{m//60:02d}:{m%60:02d}")
    act = df[df["r"] != 0]
    g = act.groupby(["code", "bucket"])["r"]
    out = pd.DataFrame({"active": g.size(), "hit": g.apply(lambda s: (s > 0).mean()),
                        "mean_bp": g.mean() * 1e4, "total_bp": g.sum() * 1e4})
    return out.reset_index()


def fig_heatmap(tab):
    buckets = [b for b in sorted(tab["bucket"].unique())
               if tab[tab.bucket == b]["active"].sum() >= 20]      # drop near-empty (11:30/15:00)
    M = (tab.pivot(index="code", columns="bucket", values="mean_bp")
         .reindex(PRODUCTS)[buckets])
    fig, ax = plt.subplots(figsize=(12, 4.2))
    vmax = 1.2   # values are all sub-bp; tighten the scale so structure shows
    im = ax.imshow(M.values, cmap="RdYlGn_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(buckets)), buckets)
    ax.set_yticks(range(4), [NAME[p] for p in PRODUCTS])
    ax.set_xlabel("日内时段（30分钟）"); ax.set_ylabel("合约")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8)
    cb = fig.colorbar(im, ax=ax, pad=.01)
    cb.set_label("动量每笔平均收益（bp）　红=正(动量/趋势延续) / 绿=负(反转)")
    ax.set_title("图8　全日动量/反转地图（38天，@10跳）：各时段动量信号的每笔平均收益（bp）",
                 fontsize=13, fontweight="bold", pad=18)
    fig.text(0.5, 0.02, "注：30分钟时段会稀释1分钟级现象——开盘强反转只集中在09:31这一分钟，"
             "在09:30时段里被其余分钟摊薄。毛收益。", ha="center", fontsize=8.5, color="0.35")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(f"{OUT}/fig8_全日动量地图.png", dpi=150); plt.close(fig)


def fig_cumulative(df):
    fig, ax = plt.subplots(figsize=(12.5, 6))
    for p in PRODUCTS:
        s = df[df.code == p].groupby("hhmm")["r"].mean()           # mean incl. zeros
        am = s[s.index <= AM_END].sort_index()
        pm = s[s.index >= PM_START].sort_index()
        cum_am = am.cumsum() * 1e4
        cum_pm = (am.sum() + pm.cumsum()) * 1e4
        ax.plot(am.index, cum_am.values, color=COL[p], lw=1.7, label=NAME[p])
        ax.plot(pm.index, cum_pm.values, color=COL[p], lw=1.7)
    ax.axhline(0, color="k", lw=.8)
    ax.axvspan(AM_END, PM_START, color="0.9")                       # lunch
    ax.text((AM_END + PM_START) / 2, ax.get_ylim()[1] * .9, "午休", ha="center", fontsize=9)
    ticks = [570, 600, 630, 660, 690, 780, 810, 840, 870, 900]
    ax.set_xticks(ticks, [f"{m//60:02d}:{m%60:02d}" for m in ticks])
    ax.set_xlabel("日内时间")
    ax.set_ylabel("平均累计动量收益（bp，按交易日平均后累加）")
    ax.set_title("图9　平均日内累计动量收益（38天，@10跳）：早盘动量占优(上行)、午后反转占优(下行)",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=.3); ax.legend(loc="best")
    fig.text(0.5, 0.01, "曲线下行=动量亏损(反转占优)，上行=动量盈利。开盘第一分钟的反转表现为左端的向下台阶。"
             "毛收益。", ha="center", fontsize=8.5, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(f"{OUT}/fig9_日内累计曲线.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    df = build(SRC)
    print("days:", df["d"].nunique(), "| rows:", len(df),
          "| minutes/day(IC):", df[df.code == "IC0000"].groupby("d").size().median())
    tab = bucket_table(df)
    tab.to_csv("/Users/zhuisabella/xn/ticker/open_breakdown/day_buckets.csv", index=False)
    pd.set_option("display.width", 200, "display.max_rows", 200)
    print("\n=== 30分钟时段 动量画像（@10跳，38天）===")
    print(tab.round({"hit": 3, "mean_bp": 2, "total_bp": 1}).to_string(index=False))
    fig_heatmap(tab); fig_cumulative(df)
    print("\nsaved fig8_全日动量地图.png, fig9_日内累计曲线.png ->", OUT)
