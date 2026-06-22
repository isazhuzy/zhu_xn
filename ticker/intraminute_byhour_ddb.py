"""
intraminute_byhour_ddb.py — prev-minute reversal by time of day, FINAL design.

Same trade as intraminute_wholeday_ddb.py (signal=C(M-1)-C(M-2), fade, enter at tick
N=10, hold a FULL minute = exit at tick N of next minute), broken out by 30-min time
bucket, PER CONTRACT (no basket). Shows where in the day the reversal is +/-.

Compute only (DolphinDB on .venv); intraminute_byhour_plot.py (3.14) draws fig12.
Run:  /Users/zhuisabella/xn/.venv/bin/python intraminute_byhour_ddb.py   (sandbox off)
"""
import numpy as np
import pandas as pd
import dolphindb as ddb

from ddb_config import HOST, PORT, USER, PW
START, END = "2023.01.04", "2023.03.04"
CODES = ["IC0000", "IF0000", "IH0000", "IM0000"]
N = 10
OUTCSV = "/Users/zhuisabella/xn/ticker/open_breakdown/intraminute_byhour.csv"


def fetch(sess, code):
    q = f"""
    pt=loadTable("dfs://hft_future_ts","TickPartitioned")
    select m_nDatetime as ts, (m_nBidPrice+m_nAskPrice)/2.0 as mid
    from pt where code_init=`{code[:2]}, m_nDatetime>={START}T00:00:00,
          m_nDatetime<={END}T23:59:59, code=`{code},
          minute(m_nDatetime) between 09:30m:15:00m
    """
    return sess.run(q)


def trades(df):
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
    eN = df[df["idx"] == N - 1].groupby("mkey")["mid"].first()
    per["entry"] = per.index.map(eN)
    per["exit"] = per.groupby(["day", "session"])["entry"].shift(-1)
    ok = (per["next_consec"] & per["signal"].notna() & (per["signal"] != 0)
          & per["entry"].notna() & per["exit"].notna())
    b = per[ok]
    pnl = b["pos"] * (b["exit"] / b["entry"] - 1.0) * 1e4
    return pd.DataFrame({"day": b["day"].to_numpy(), "bucket": (b["tod"] // 30 * 30).to_numpy(),
                         "pnl": pnl.to_numpy()})


def agg(sub):
    daily = sub.groupby("day")["pnl"].mean()
    t = daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily))) if len(daily) > 1 else np.nan
    return dict(n=len(sub), mean_bp=round(daily.mean(), 3), t=round(t, 2))


if __name__ == "__main__":
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    parts = []
    for code in CODES:
        df = fetch(sess, code); t = trades(df); t["code"] = code
        print(f"{code}: {len(t)} trades", flush=True)
        parts.append(t)
    sess.close()
    L = pd.concat(parts, ignore_index=True)
    L["blabel"] = L["bucket"].map(lambda m: f"{m//60:02d}:{m%60:02d}")

    rows = []
    for (blab, code), g in L.groupby(["blabel", "code"]):
        if g["pnl"].count() < 200:
            continue
        rows.append({"bucket": blab, "code": code, **agg(g)})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUTCSV, index=False)
    pd.set_option("display.width", 200, "display.max_rows", 200)
    print(f"\n=== 反转(上一分钟信号,整分钟持有,N={N}) 每笔mean_bp ｜ 时段 × 合约 ===")
    print(tab.pivot(index="bucket", columns="code", values="mean_bp").round(3).to_string())
    print("\nsaved", OUTCSV)
