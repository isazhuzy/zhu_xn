"""exp2_cross_ddb.py — 成交量爆发后【反转占比】，跨合约 × 分年度（频率，不是均值）。
对 IC/IF/IH/IM 各自全历史，按年分组：成交量爆发(norm 量比 spike=(V_t1/t1)/(V_t2/t2))后，
看未来 x=20 tick 的方向 signed=sign(mid[i]-mid[i-t1])*(mid[i+x]-mid[i])，统计【反转占比 = signed<0 的比例】。
源头滤坏tick(bid>0&ask>0) + despike。
Output exp2_cross.csv: code,year,t1,t2,r,x, n_peak, g_sum,g_ss, rhits(signed<0)
Run: /Users/zhuisabella/xn/.venv/bin/python exp2_cross_ddb.py   (sandbox OFF)
"""
import calendar
import numpy as np
import pandas as pd
import dolphindb as ddb
from ddb_config import HOST, PORT, USER, PW

PAIRS = [(10, 60), (20, 120)]            # (t1,t2) 持续放量窗口
RS = [1.5, 2.0, 3.0, 5.0]                # norm 爆发强度阈值（几倍平时）
X = 20                                    # 向前 20 tick ≈ 10s
CAP = 4.0
RANGE = {"IC": (2015, 2026), "IF": (2015, 2026), "IH": (2015, 2026), "IM": (2022, 2026)}
OUT = "/Users/zhuisabella/xn/experiment/exp2_cross.csv"


def fetch(sess, code, start, end):
    q = f"""
    pt=loadTable("dfs://hft_future_ts","TickPartitioned")
    select m_nDatetime as ts, (m_nBidPrice+m_nAskPrice)/2.0 as mid, m_iVolume as vol
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
        yield g["mid"].to_numpy(float), g["vol"].to_numpy(float)


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


def exp2(mid, vol, acc, code, yr):
    mid = despike(mid)
    L = len(mid)
    cs = np.concatenate([[0.0], np.cumsum(vol)])
    def trailsum(t):
        s = np.full(L, np.nan)
        s[t - 1:] = cs[t:L + 1] - cs[0:L - t + 1]
        return s
    for (t1, t2) in PAIRS:
        v1, v2 = trailsum(t1), trailsum(t2)
        with np.errstate(divide="ignore", invalid="ignore"):
            spike = (v1 / t1) / (v2 / t2)
        lo, hi = t2, L - X
        if hi <= lo:
            continue
        i = np.arange(lo, hi)
        signed = np.sign(mid[i] - mid[i - t1]) * (mid[i + X] - mid[i])
        spi = spike[i]
        for r in RS:
            m = spi > r
            if not m.any():
                continue
            g = signed[m]
            a = acc.setdefault((code, yr, t1, t2, r), [0, 0.0, 0.0, 0])
            a[0] += int(g.size); a[1] += g.sum(); a[2] += (g * g).sum()
            a[3] += int((g < 0).sum())


if __name__ == "__main__":
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    acc = {}
    for code2, (y0, y1) in RANGE.items():
        code = code2 + "0000"
        for yr in range(y0, y1 + 1):
            for mo in range(1, 13):
                last = calendar.monthrange(yr, mo)[1]
                start, end = f"{yr}.{mo:02d}.01", f"{yr}.{mo:02d}.{last:02d}"
                df = fetch(sess, code, start, end)
                if not len(df):
                    continue
                for mid, vol in session_blocks(df):
                    if len(mid) > max(p[1] for p in PAIRS) + X:
                        exp2(mid, vol, acc, code, yr)
            print(f"{code} {yr} done", flush=True)
    sess.close()
    rows = [{"code": c, "year": y, "t1": t1, "t2": t2, "r": r, "x": X,
             "n_peak": a[0], "g_sum": a[1], "g_ss": a[2], "rhits": a[3]}
            for (c, y, t1, t2, r), a in acc.items()]
    pd.DataFrame(rows).sort_values(["code", "year", "t1", "r"]).to_csv(OUT, index=False)
    print(f"saved {OUT} ({len(rows)})")
