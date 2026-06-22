"""
plan2_threshold_ddb.py — is there a "prime" price-tick threshold? (open minutes, 2 months)

Trade (open minutes 09:30–09:59, valid trade minutes 09:32–09:59):
  signal = C(M-1) - C(M-2)            # previous-minute move (close-to-close)
  pos    = -sign(signal)             # 做反转 / reversal
  entry  = first mid of minute M  ;  exit = last mid of minute M  (1-min hold, bar-level)
  ret    = pos * (exit/entry - 1) * 1e4   (bp, gross)

Dead-band sweep k = 0..40 ticks: keep only trades with |signal| > k*0.2.
Objective = MEAN per trade. Overfitting guard: also compute the curve on two
independent day-halves (alternating days). Per contract × {full,H1,H2} × k:
n, mean_bp, hit, t(daily-aggregated). Saves CSV; plan2_plot.py draws it.

Run:  /Users/zhuisabella/xn/.venv/bin/python plan2_threshold_ddb.py   (sandbox off)
"""
import numpy as np
import pandas as pd
import dolphindb as ddb

from ddb_config import HOST, PORT, USER, PW
START, END = "2023.01.04", "2023.03.04"
CODES = ["IC0000", "IF0000", "IH0000", "IM0000"]
KS = list(range(0, 41))
TICK = 0.2
OUTCSV = "/Users/zhuisabella/xn/ticker/open_breakdown/plan2_threshold.csv"


def fetch(sess, code, start=START, end=END):
    q = f"""
    pt=loadTable("dfs://hft_future_ts","TickPartitioned")
    select m_nDatetime as ts, (m_nBidPrice+m_nAskPrice)/2.0 as mid
    from pt where code_init=`{code[:2]}, m_nDatetime>={start}T00:00:00,
          m_nDatetime<={end}T23:59:59, code=`{code},
          minute(m_nDatetime) between 09:30m:09:59m
    """
    return sess.run(q)


def build_trades(df):
    """open-minute reversal trades: one row per valid trade minute."""
    df = df.sort_values("ts").copy()
    df["mkey"] = df["ts"].dt.floor("min")
    per = pd.DataFrame({"first": df.groupby("mkey")["mid"].first(),
                        "last": df.groupby("mkey")["mid"].last()})
    per["day"] = per.index.normalize()
    per["tod"] = per.index.hour * 60 + per.index.minute
    per = per.sort_values(["day", "tod"])
    g = per.groupby("day")
    c1, c2 = g["last"].shift(1), g["last"].shift(2)
    t1, t2 = g["tod"].shift(1), g["tod"].shift(2)
    consec = (per["tod"] - t1 == 1) & (t1 - t2 == 1)
    sig = np.where(consec, c1 - c2, np.nan)
    per["sig_ticks"] = np.abs(sig) / TICK
    pos = -np.sign(sig)
    per["ret"] = pos * (per["last"] / per["first"] - 1.0) * 1e4
    out = per[per["sig_ticks"].notna() & (per["sig_ticks"] > 0)][["day", "ret", "sig_ticks"]]
    return out.reset_index(drop=True)


def agg(sub):
    n = len(sub)
    if n == 0:
        return None
    daily = sub.groupby("day")["ret"].mean()
    t = daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily))) if len(daily) > 1 and daily.std(ddof=1) > 0 else np.nan
    return dict(n=n, mean_bp=round(sub["ret"].mean(), 3),
                hit=round((sub["ret"] > 0).mean(), 3), t=round(t, 2))


if __name__ == "__main__":
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    rows = []
    for code in CODES:
        tr = build_trades(fetch(sess, code))
        days = sorted(tr["day"].unique())
        h1 = set(days[::2]); h2 = set(days[1::2])          # alternating-day halves
        tr["half"] = np.where(tr["day"].isin(h1), "H1", "H2")
        print(f"{code}: {len(tr)} trades, {len(days)} days", flush=True)
        for k in KS:
            for half, sub in [("full", tr), ("H1", tr[tr.half == "H1"]),
                              ("H2", tr[tr.half == "H2"])]:
                st = agg(sub[sub["sig_ticks"] > k])
                if st:
                    rows.append({"code": code, "half": half, "k": k, **st})
    sess.close()
    tab = pd.DataFrame(rows)
    tab.to_csv(OUTCSV, index=False)
    pd.set_option("display.width", 200, "display.max_rows", 60)
    print("\n=== k* (mean_bp 最大) per contract (full sample) ===")
    f = tab[tab.half == "full"]
    for code in CODES:
        s = f[f.code == code]
        best = s.loc[s["mean_bp"].idxmax()]
        print(f"{code}: k*={int(best['k'])}  mean_bp={best['mean_bp']}  n={int(best['n'])}  t={best['t']}"
              f"   | k=0: mean={s[s.k==0]['mean_bp'].iloc[0]} n={int(s[s.k==0]['n'].iloc[0])}")
    print("\nsaved", OUTCSV)
