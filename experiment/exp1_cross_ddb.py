"""exp1_cross_ddb.py — 方向1+2: EXP1 价格脉冲反转，跨合约 × 分年度 + 各合约真实价差。
对 IC/IF/IH/IM 四个合约、各自全历史（IC/IF/IH 2015-2026, IM 2022-2026），按【年】分组，
算 EXP1 n×k（固定 x=10）的符号化收益 + 反转命中数；同时累计每合约每年的【平均价差】。
源头滤坏报价（bid>0 & ask>0）+ 单点尖峰去除（despike）。
Output:
  exp1_cross.csv        : code,year,n,k_ticks,k_pts, s_sum,s_ss,s_n,hits(rev=signed<0)
  exp1_cross_spread.csv : code,year, sp_sum,sp_n  (平均价差 = sp_sum/sp_n, 指数点)
Run: /Users/zhuisabella/xn/.venv/bin/python exp1_cross_ddb.py   (sandbox OFF)
"""
import calendar
import numpy as np
import pandas as pd
import dolphindb as ddb
from ddb_config import HOST, PORT, USER, PW

TICK = 0.2
X = 10
NS = [1, 2, 3, 5, 10, 20, 30]
KTICKS = [0, 2, 4, 6, 8, 13, 20, 30]
KS = [kt * TICK for kt in KTICKS]
CAP = 4.0
RANGE = {"IC": (2015, 2026), "IF": (2015, 2026), "IH": (2015, 2026), "IM": (2022, 2026)}
OUT = "/Users/zhuisabella/xn/experiment/exp1_cross.csv"
OUT_SP = "/Users/zhuisabella/xn/experiment/exp1_cross_spread.csv"


def fetch(sess, code, start, end):
    q = f"""
    pt=loadTable("dfs://hft_future_ts","TickPartitioned")
    select m_nDatetime as ts, (m_nBidPrice+m_nAskPrice)/2.0 as mid,
           (m_nAskPrice-m_nBidPrice) as spread
    from pt where code_init=`{code[:2]}, m_nDatetime>={start}T00:00:00,
          m_nDatetime<={end}T23:59:59, code=`{code},
          m_nBidPrice>0, m_nAskPrice>0,
          minute(m_nDatetime) between 09:30m:15:00m
    """
    return sess.run(q)


def session_blocks(df):
    df = df.drop_duplicates("ts").sort_values("ts").copy()
    tod = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    df = df[((tod >= 570) & (tod <= 690)) | ((tod >= 780) & (tod <= 900))]
    if df.empty:
        return
    df["day"] = df["ts"].dt.normalize()
    df["session"] = np.where(tod[df.index] <= 690, "AM", "PM")
    for _, g in df.groupby(["day", "session"], sort=False):
        yield g["mid"].to_numpy(float), g["spread"].to_numpy(float)


def despike(mid):
    m = mid.astype(float).copy()
    for _ in range(2):
        if len(m) < 3:
            break
        dp = m[1:-1] - m[:-2]; dn = m[1:-1] - m[2:]
        spk = (np.abs(dp) > CAP) & (np.abs(dn) > CAP) & (np.sign(dp) == np.sign(dn))
        idx = np.where(spk)[0] + 1
        if not idx.size:
            break
        m[idx] = 0.5 * (m[idx - 1] + m[idx + 1])
    return m


def exp1(mid, acc, code, yr):
    mid = despike(mid)
    L = len(mid)
    for n in NS:
        lo, hi = n, L - X
        if hi <= lo:
            continue
        i = np.arange(lo, hi)
        back = mid[i] - mid[i - n]
        fwd = mid[i + X] - mid[i]
        signed = np.sign(back) * fwd
        ab = np.abs(back)
        for kt, k in zip(KTICKS, KS):
            m = ab > k if k > 0 else np.ones(len(i), bool)
            s = signed[m]
            if not s.size:
                continue
            a = acc.setdefault((code, yr, n, kt), [0.0, 0.0, 0, 0])
            a[0] += s.sum(); a[1] += (s * s).sum(); a[2] += s.size
            a[3] += int((s < 0).sum())          # 反转命中（signed<0）


if __name__ == "__main__":
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    acc, spacc = {}, {}
    for code2, (y0, y1) in RANGE.items():
        code = code2 + "0000"
        for yr in range(y0, y1 + 1):
            for mo in range(1, 13):
                last = calendar.monthrange(yr, mo)[1]
                start, end = f"{yr}.{mo:02d}.01", f"{yr}.{mo:02d}.{last:02d}"
                df = fetch(sess, code, start, end)
                if not len(df):
                    continue
                for mid, spread in session_blocks(df):
                    if len(mid) > max(NS) + X:
                        exp1(mid, acc, code, yr)
                    sp = spacc.setdefault((code, yr), [0.0, 0])
                    sp[0] += float(np.nansum(spread)); sp[1] += int(np.isfinite(spread).sum())
            print(f"{code} {yr} done", flush=True)
    sess.close()
    rows = [{"code": c, "year": y, "n": n, "k_ticks": kt, "k_pts": round(kt * TICK, 2),
             "s_sum": a[0], "s_ss": a[1], "s_n": a[2], "hits": a[3]}
            for (c, y, n, kt), a in acc.items()]
    pd.DataFrame(rows).sort_values(["code", "year", "n", "k_ticks"]).to_csv(OUT, index=False)
    sp_rows = [{"code": c, "year": y, "sp_sum": s[0], "sp_n": s[1]} for (c, y), s in spacc.items()]
    pd.DataFrame(sp_rows).sort_values(["code", "year"]).to_csv(OUT_SP, index=False)
    print(f"saved {OUT} ({len(rows)}) , {OUT_SP} ({len(sp_rows)})")
