"""
intraminute_wholeday_ddb.py — previous-minute reversal, FULL-MINUTE hold, entry shifted by N ticks.

Design (per user):
  signal = C(M-1) - C(M-2)            # previous minute's move (same 1-min signal as matrix.py)
  pos    = -sign(signal)             # 做反转
  entry  = mid at the N-th tick of minute M
  exit   = mid at the N-th tick of minute M+1     # <-- whole-minute hold; the window
                                                  #     [tickN(M), tickN(M+1)] is just shifted
                                                  #     back by N ticks, length stays ~1 minute
  pnl    = pos * (exit/entry - 1) * 1e4           (bp, gross)

So N changes ONLY the intra-minute entry timing, NOT the holding length (was 120-N ticks before).
Per-contract only (no equal-weight basket). t = t-stat on the 38 DAILY means.
Run:  /Users/zhuisabella/xn/.venv/bin/python intraminute_wholeday_ddb.py   (sandbox off)
"""
import numpy as np
import pandas as pd
import dolphindb as ddb

from ddb_config import HOST, PORT, USER, PW
START, END = "2023.01.04", "2023.03.04"
CODES = ["IC0000", "IF0000", "IH0000", "IM0000"]
N_LIST = [1, 2, 4, 8, 10, 12, 14, 16, 18, 20, 22, 26, 30]
OUTCSV = "/Users/zhuisabella/xn/ticker/open_breakdown/intraminute_wholeday.csv"


def fetch(sess, code, start=START, end=END):
    q = f"""
    pt=loadTable("dfs://hft_future_ts","TickPartitioned")
    select m_nDatetime as ts, (m_nBidPrice+m_nAskPrice)/2.0 as mid
    from pt
    where code_init=`{code[:2]}, m_nDatetime>={start}T00:00:00, m_nDatetime<={end}T23:59:59,
          code=`{code}, minute(m_nDatetime) between 09:30m:15:00m
    """
    return sess.run(q)


def contract_trades(df, code):
    df = df.sort_values("ts").copy()
    df["mkey"] = df["ts"].dt.floor("min")
    tod = df["mkey"].dt.hour * 60 + df["mkey"].dt.minute
    df = df[((tod >= 570) & (tod <= 690)) | ((tod >= 780) & (tod <= 900))]
    df["idx"] = df.groupby("mkey").cumcount()

    per = pd.DataFrame({"last": df.groupby("mkey")["mid"].last(),
                        "nt": df.groupby("mkey")["mid"].size()})
    per["day"] = per.index.normalize()
    per["tod"] = per.index.hour * 60 + per.index.minute
    per["session"] = np.where(per["tod"] <= 690, "AM", "PM")
    per = per.sort_values(["day", "session", "tod"])
    gp = per.groupby(["day", "session"])
    c1, c2 = gp["last"].shift(1), gp["last"].shift(2)
    t1, t2, tn = gp["tod"].shift(1), gp["tod"].shift(2), gp["tod"].shift(-1)
    per["signal"] = np.where((per["tod"] - t1 == 1) & (t1 - t2 == 1), c1 - c2, np.nan)
    per["pos"] = -np.sign(per["signal"])
    per["next_consec"] = (tn - per["tod"] == 1)

    out = []
    for N in N_LIST:
        eN = df[df["idx"] == N - 1].groupby("mkey")["mid"].first()     # tick-N mid per minute
        per["entry"] = per.index.map(eN)
        per["exit"] = per.groupby(["day", "session"])["entry"].shift(-1)   # tick-N of NEXT minute
        ok = (per["next_consec"] & per["signal"].notna() & (per["signal"] != 0)
              & per["entry"].notna() & per["exit"].notna())
        b = per[ok]
        pnl = b["pos"] * (b["exit"] / b["entry"] - 1.0) * 1e4
        out.append(pd.DataFrame({"day": b["day"].to_numpy(), "N": N, "pnl": pnl.to_numpy()}))
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

    rows = [{"group": code, "N": N, **agg(L[(L.code == code) & (L.N == N)])}
            for N in N_LIST for code in CODES]
    tab = pd.DataFrame(rows)
    tab.to_csv(OUTCSV, index=False)
    pd.set_option("display.width", 200, "display.max_rows", 300)
    print("\n=== 上一分钟信号做反转，第N tick进场，持有整一分钟(出场=下一分钟第N tick)｜2个月｜各合约 ===")
    print("（mean_bp=每笔毛收益；t=按交易日聚合；N=进场延迟 tick）\n")
    print(tab.to_string(index=False))
    print("\nsaved", OUTCSV)
