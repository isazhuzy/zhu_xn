"""
fullminute_extended.py — FULL-MINUTE-hold reversal, shifted by N ticks, on a
SELECTIVELY EXTENDED span (not all 11 years — sampled windows across years).

"Shift everything back by N ticks": the whole 1-minute window slides by N ticks, so
the hold is always a full minute (not the shrinking 120-N remainder).
  anchor E   = the N-th tick of minute M
  signal     = mid(E) - mid(N-th tick of M-1)   # trailing full minute at the anchor
  pos        = -sign(signal)                     # 做反转
  exit       = mid(N-th tick of M+1)             # one full minute later
  pnl        = pos * (exit/E - 1) * 1e4          (bp, gross)
N=1 ≈ standard clock-minute reversal. Needs M-1,M,M+1 consecutive in-session.

Selective sample (fast): ~3-week windows across years + the original 2-month 2023.
IM0000 (中证1000) only exists from 2022-07, so it's absent in the early windows.

Compute only (.venv); fullminute_extended_plot.py (3.14) draws fig13.
Run:  /Users/zhuisabella/xn/.venv/bin/python fullminute_extended.py   (sandbox off)
"""
import numpy as np
import pandas as pd
import dolphindb as ddb

from ddb_config import HOST, PORT, USER, PW
CODES = ["IC0000", "IF0000", "IH0000", "IM0000"]
N_LIST = [1, 2, 4, 8, 12, 20, 30]
PERIODS = [("2016-06", "2016.06.01", "2016.06.21"),
           ("2019-06", "2019.06.01", "2019.06.21"),
           ("2021-06", "2021.06.01", "2021.06.21"),
           ("2023-orig", "2023.01.04", "2023.03.04"),
           ("2024-06", "2024.06.03", "2024.06.21"),
           ("2025-06", "2025.06.03", "2025.06.20")]
OUTCSV = "/Users/zhuisabella/xn/ticker/open_breakdown/fullminute_extended.csv"


def fetch(sess, code, start, end):
    q = f"""
    pt=loadTable("dfs://hft_future_ts","TickPartitioned")
    select m_nDatetime as ts, (m_nBidPrice+m_nAskPrice)/2.0 as mid
    from pt where code_init=`{code[:2]}, m_nDatetime>={start}T00:00:00,
          m_nDatetime<={end}T23:59:59, code=`{code},
          minute(m_nDatetime) between 09:30m:15:00m
    """
    return sess.run(q)


def trades(df):
    df = df.sort_values("ts").copy()
    df["mkey"] = df["ts"].dt.floor("min")
    tod = df["mkey"].dt.hour * 60 + df["mkey"].dt.minute
    df = df[((tod >= 570) & (tod <= 690)) | ((tod >= 780) & (tod <= 900))]
    if df.empty:
        return pd.DataFrame(columns=["day", "N", "pnl"])
    df["idx"] = df.groupby("mkey").cumcount()
    per = pd.DataFrame(index=pd.Index(sorted(df["mkey"].unique()), name="mkey"))
    for N in N_LIST:
        per[f"e{N}"] = df[df["idx"] == N - 1].set_index("mkey")["mid"]
    per["day"] = per.index.normalize()
    per["tod"] = per.index.hour * 60 + per.index.minute
    per["session"] = np.where(per["tod"] <= 690, "AM", "PM")
    per = per.sort_index()
    g = per.groupby(["day", "session"])
    cprev = (per["tod"] - g["tod"].shift(1) == 1)
    cnext = (g["tod"].shift(-1) - per["tod"] == 1)
    out = []
    for N in N_LIST:
        e = per[f"e{N}"]
        pe, ne = g[f"e{N}"].shift(1), g[f"e{N}"].shift(-1)
        sig = e - pe
        valid = e.notna() & pe.notna() & ne.notna() & cprev & cnext & (sig != 0)
        pos = -np.sign(sig[valid])
        pnl = pos * (ne[valid] / e[valid] - 1.0) * 1e4
        out.append(pd.DataFrame({"day": per["day"][valid].to_numpy(),
                                 "N": N, "pnl": pnl.to_numpy()}))
    return pd.concat(out, ignore_index=True)


def agg(sub):
    daily = sub.groupby("day")["pnl"].mean()
    t = daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily))) if len(daily) > 1 else np.nan
    return dict(n=len(sub), win=round((sub["pnl"] > 0).mean(), 3),
                mean_bp=round(daily.mean(), 3), t=round(t, 2))


if __name__ == "__main__":
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    rows = []
    for plabel, s, e in PERIODS:
        L = []
        present = []
        for code in CODES:
            df = fetch(sess, code, s, e)
            if len(df) == 0:
                continue
            t = trades(df)
            if len(t):
                t["code"] = code; L.append(t); present.append(code[:2])
        if not L:
            print(f"{plabel}: no data", flush=True); continue
        LL = pd.concat(L, ignore_index=True)
        ndays = LL["day"].nunique()
        for N in N_LIST:
            bk = LL[LL.N == N].groupby(["day"])["pnl"].mean().reset_index()
            rows.append({"period": plabel, "N": N, "days": ndays,
                         "contracts": "+".join(present), **agg(bk)})
        print(f"{plabel}: {ndays} days, contracts {present}", flush=True)
    sess.close()
    tab = pd.DataFrame(rows)
    tab.to_csv(OUTCSV, index=False)
    pd.set_option("display.width", 220, "display.max_rows", 200)
    print("\n=== 全分钟持有反转(整体平移N tick)：各抽样时段 × 进场延迟N（组合等权 basket）===")
    print("（mean_bp=每笔毛收益；t=按交易日聚合；N=1≈分钟边界反转）\n")
    print(tab.to_string(index=False))
    print("\nsaved", OUTCSV)
