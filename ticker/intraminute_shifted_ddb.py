"""
intraminute_shifted_ddb.py — "whole-minute / shifted grid" variant (2-month window).

Difference vs intraminute_wholeday_ddb.py: there, entry = tick N, exit = THAT minute's
close, so the holding shrinks as N grows (120-N ticks). Here we shift the ENTIRE
1-minute grid back by N ticks, so the hold is always a FULL minute and N is a pure
phase offset:

    entry  = mid at tick N of minute M
    signal = mid(tick N of M) - mid(tick N of M-1)     # previous full minute (shifted)
    pos    = -sign(signal)                              # 做反转 / reversal
    exit   = mid at tick N of minute M+1                # one full minute later
    pnl    = pos * (exit/entry - 1) * 1e4               (bp, gross)

N=1 ≈ the standard clean-minute reversal. M-1,M,M+1 must be consecutive in-session.
Compute only (.venv); intraminute_shifted_plot.py (3.14) draws the comparison fig.
Run:  /Users/zhuisabella/xn/.venv/bin/python intraminute_shifted_ddb.py   (sandbox off)
"""
import numpy as np
import pandas as pd
import dolphindb as ddb

from ddb_config import HOST, PORT, USER, PW
START, END = "2023.01.04", "2023.03.04"
CODES = ["IC0000", "IF0000", "IH0000", "IM0000"]
N_LIST = [1, 2, 4, 8, 10, 12, 14, 16, 18, 20, 22, 26, 30]
OUTCSV = "/Users/zhuisabella/xn/ticker/open_breakdown/intraminute_shifted.csv"


def fetch(sess, code):
    q = f"""
    pt=loadTable("dfs://hft_future_ts","TickPartitioned")
    select m_nDatetime as ts, (m_nBidPrice+m_nAskPrice)/2.0 as mid
    from pt where code_init=`{code[:2]}, m_nDatetime>={START}T00:00:00,
          m_nDatetime<={END}T23:59:59, code=`{code},
          minute(m_nDatetime) between 09:30m:15:00m
    """
    return sess.run(q)


def contract_trades(df, code):
    df = df.sort_values("ts").copy()
    df["mkey"] = df["ts"].dt.floor("min")
    tod = df["mkey"].dt.hour * 60 + df["mkey"].dt.minute
    df = df[((tod >= 570) & (tod <= 690)) | ((tod >= 780) & (tod <= 900))]
    df["idx"] = df.groupby("mkey").cumcount()

    meta = pd.DataFrame({"nt": df.groupby("mkey").size()}).sort_index()
    meta["day"] = meta.index.normalize()
    meta["tod"] = meta.index.hour * 60 + meta.index.minute
    meta["session"] = np.where(meta["tod"] <= 690, "AM", "PM")

    out = []
    for N in N_LIST:
        midN = df[df["idx"] == N - 1].set_index("mkey")["mid"].rename("midN")
        m = meta.join(midN, how="left").sort_index()
        g = m.groupby(["day", "session"])
        m["prev"] = g["midN"].shift(1)        # tick N of M-1  (signal start)
        m["nextv"] = g["midN"].shift(-1)      # tick N of M+1  (exit)
        m["ptod"] = g["tod"].shift(1)
        m["ntod"] = g["tod"].shift(-1)
        consec = (m["tod"] - m["ptod"] == 1) & (m["ntod"] - m["tod"] == 1)
        v = m[consec & m["midN"].notna() & m["prev"].notna() & m["nextv"].notna()].copy()
        sig = v["midN"] - v["prev"]
        v = v[sig != 0]
        pos = -np.sign(v["midN"] - v["prev"])
        pnl = pos * (v["nextv"] / v["midN"] - 1.0) * 1e4
        out.append(pd.DataFrame({"day": v["day"].to_numpy(), "mkey": v.index,
                                 "N": N, "pnl": pnl.to_numpy()}))
    return pd.concat(out, ignore_index=True)


def agg(sub):
    daily = sub.groupby("day")["pnl"].mean()
    t = daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily))) if len(daily) > 1 else np.nan
    return dict(n=len(sub), win=round((sub["pnl"] > 0).mean(), 3),
                mean_bp=round(daily.mean(), 3), t=round(t, 2))


if __name__ == "__main__":
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    parts = []
    for code in CODES:
        df = fetch(sess, code)
        print(f"fetched {code}: {df.shape[0]} ticks", flush=True)
        t = contract_trades(df, code); t["code"] = code
        parts.append(t)
    sess.close()
    L = pd.concat(parts, ignore_index=True)

    rows = []
    for N in N_LIST:
        for code in CODES:
            rows.append({"group": code, "N": N, **agg(L[(L.code == code) & (L.N == N)])})
        bk = L[L.N == N].groupby(["day", "mkey"])["pnl"].mean().reset_index()
        rows.append({"group": "组合(4合约等权)", "N": N, **agg(bk)})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUTCSV, index=False)
    pd.set_option("display.width", 200, "display.max_rows", 300)
    print("\n=== 整分钟/平移版：第N tick进场，持有满一分钟到下一分钟第N tick（38天，反转）===")
    print("（N=相位偏移；持有时长恒定=满分钟；mean_bp=每笔毛收益；t=按交易日聚合）\n")
    print(tab.to_string(index=False))
    print("\nsaved", OUTCSV)
