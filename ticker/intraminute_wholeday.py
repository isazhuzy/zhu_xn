"""
intraminute_wholeday.py — intra-minute fade applied to EVERY minute of the day.

For each minute M of the session (whole day, all 38 days), split M into its ticks,
watch the first N, FADE that move, enter at tick N, exit at M's close:
    sig   = mid[tick N] - mid[tick 1]          (first N ticks of minute M)
    pos   = -sign(sig)                         (reversal / fade)
    entry = mid[tick N]
    exit  = mid[last tick of minute M]
    pnl   = pos * (exit/entry - 1) * 1e4       (bp, gross, mid-to-mid)

Sweep N = 8,10,...,30. Each minute is one trade -> ~9000 trades/contract over the
2 months. Effect size = per-trade mean; significance = t on the 38 DAILY means
(honest: intraday & cross-contract trades are correlated).

Run:  /usr/local/bin/python3.14 intraminute_wholeday.py   (loads the 450MB tick CSVs)
"""
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

DIR = "/Users/zhuisabella/xn/ticker"
OUT = f"{DIR}/figs_conclusion"
FILES = {"IC0000": f"{DIR}/IC_20230104_20230304.csv",
         "IF0000": f"{DIR}/IF_20230104_20230304.csv",
         "IH0000": f"{DIR}/IH_20230104_20230304.csv",
         "IM0000": f"{DIR}/IM_20230104_20230304.csv"}
NAME = {"IC0000": "IC 中证500", "IF0000": "IF 沪深300",
        "IH0000": "IH 上证50", "IM0000": "IM 中证1000"}
COL = {"IC0000": "#c0392b", "IF0000": "#e08a3c", "IH0000": "#27ae60", "IM0000": "#4C72B0"}
N_LIST = [8, 10, 12, 14, 16, 18, 20, 22, 26, 30]


def contract_trades(path, code):
    """-> long df: day, mkey(minute), N, pnl  for every session minute."""
    df = pd.read_csv(path, usecols=["code", "m_nDatetime", "m_nBidPrice", "m_nAskPrice"],
                     dtype={"code": "string"})
    df = df[df["code"] == code]
    ts = pd.to_datetime(df["m_nDatetime"])
    mid = (df["m_nBidPrice"].to_numpy() + df["m_nAskPrice"].to_numpy()) / 2.0
    df = pd.DataFrame({"ts": ts.to_numpy(), "mid": mid}).sort_values("ts")
    df["mkey"] = df["ts"].dt.floor("min")
    tod = df["mkey"].dt.hour * 60 + df["mkey"].dt.minute
    df = df[((tod >= 570) & (tod <= 690)) | ((tod >= 780) & (tod <= 900))]   # session minutes
    df["idx"] = df.groupby("mkey").cumcount()
    base = pd.DataFrame({"first": df[df["idx"] == 0].set_index("mkey")["mid"],
                         "last": df.groupby("mkey")["mid"].last(),
                         "nt": df.groupby("mkey")["mid"].size()})
    out = []
    for N in N_LIST:
        eN = df[df["idx"] == N - 1].set_index("mkey")["mid"].rename("entry")
        b = base.join(eN, how="inner")
        b = b[b["nt"] > N]
        sig = b["entry"] - b["first"]
        b = b[sig != 0]
        pos = -np.sign(b["entry"] - b["first"])
        pnl = pos * (b["last"] / b["entry"] - 1.0) * 1e4
        out.append(pd.DataFrame({"day": b.index.normalize(), "mkey": b.index,
                                 "N": N, "pnl": pnl.to_numpy()}))
    return pd.concat(out, ignore_index=True)


def agg(sub):
    daily = sub.groupby("day")["pnl"].mean()
    t = daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily))) if len(daily) > 1 else np.nan
    return dict(n=len(sub), win=round((sub["pnl"] > 0).mean(), 3),
                mean_bp=round(daily.mean(), 3), t=round(t, 2))


def main():
    parts = []
    for code, path in FILES.items():
        print("processing", code, "...", flush=True)
        t = contract_trades(path, code); t["code"] = code
        parts.append(t)
    L = pd.concat(parts, ignore_index=True)

    rows = []
    for N in N_LIST:
        for code in FILES:
            rows.append({"group": code, "N": N, **agg(L[(L.code == code) & (L.N == N)])})
        # equal-weight basket: per (day,minute) mean across contracts, then daily
        bk = L[L.N == N].groupby(["day", "mkey"])["pnl"].mean().reset_index()
        rows.append({"group": "组合(4合约等权)", "N": N, **agg(bk)})
    tab = pd.DataFrame(rows)
    tab.to_csv(f"{DIR}/open_breakdown/intraminute_wholeday.csv", index=False)
    pd.set_option("display.width", 200, "display.max_rows", 200)
    print("\n=== 全日逐分钟 反转择时：每分钟前N个tick定方向，持有到该分钟收盘（38天）===")
    print("（mean_bp=每笔毛收益；t=按38个交易日聚合的t值）\n")
    print(tab.to_string(index=False))
    plot(tab)


def plot(tab):
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    for code in FILES:
        s = tab[tab.group == code].sort_values("N")
        axes[0].plot(s["N"], s["mean_bp"], "-o", color=COL[code], ms=5, lw=1.2, alpha=.7, label=NAME[code])
        axes[1].plot(s["N"], s["win"] * 100, "-o", color=COL[code], ms=5, lw=1.2, alpha=.7, label=NAME[code])
    b = tab[tab.group == "组合(4合约等权)"].sort_values("N")
    axes[0].plot(b["N"], b["mean_bp"], "-s", color="k", lw=2.6, ms=8, label="组合(等权)", zorder=5)
    for _, r in b.iterrows():
        axes[0].annotate(f"t={r['t']:.1f}", (r["N"], r["mean_bp"]),
                         textcoords="offset points", xytext=(0, 9), ha="center",
                         fontsize=8, fontweight="bold")
    axes[1].plot(b["N"], b["win"] * 100, "-s", color="k", lw=2.6, ms=8, label="组合(等权)")
    axes[0].axhline(0, color="0.5", lw=.8); axes[1].axhline(50, color="0.5", lw=1, ls=":")
    axes[0].set_ylabel("每笔平均收益（bp，毛）"); axes[1].set_ylabel("逆势胜率（%）")
    axes[1].set_xlabel("进场 tick 序号 N（约 N/2 秒进场；持有到该分钟收盘）")
    axes[1].set_xticks(N_LIST)
    for ax in axes:
        ax.grid(alpha=.3); ax.legend(fontsize=8, ncol=2)
    fig.suptitle("图11　全日逐分钟「前N个tick反转」择时（38天，全时段聚合）：进场tick N 扫描",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, 0.01, "每个交易分钟都视作一次小开盘：看前N个tick方向、反向开仓、持有到该分钟收盘。"
             "约9000笔/合约；t按38个交易日聚合（已考虑日内/合约间相关）。毛收益。",
             ha="center", fontsize=8.5, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(f"{OUT}/fig11_全日逐分钟tick扫描.png", dpi=150); plt.close(fig)
    print("\nsaved fig11_全日逐分钟tick扫描.png ->", OUT)


if __name__ == "__main__":
    main()
