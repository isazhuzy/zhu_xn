"""ofi_full_ddb.py — OFI analysis over the FULL 2020-01..2026-05 window (76 months).
Chunks by month (can't pull 76mo in one query) and accumulates OLS moments so the pooled
R2 / slope / depth D are exact. Also stores per-month R2 (stability) and a binned scatter.

Models per (code, bucket B):
  OFI contemporaneous :  dP_k    ~ OFI_k
  TI  contemporaneous :  dP_k    ~ TI_k
  OFI predictive      :  dP_{k+1}~ OFI_k
Pooled OLS from summed moments [n, Sx, Sy, Sxy, Sxx, Syy]:
  slope=(n*Sxy-Sx*Sy)/(n*Sxx-Sx^2);  R2=(n*Sxy-Sx*Sy)^2/((n*Sxx-Sx^2)(n*Syy-Sy^2));  D=1/(2*slope)
Run: /Users/zhuisabella/xn/.venv/bin/python ofi_full_ddb.py   (sandbox OFF)
"""
import calendar
import numpy as np, pandas as pd, dolphindb as ddb
from ddb_config import HOST, PORT, USER, PW
CODES = ["IC0000", "IF0000", "IH0000", "IM0000"]
TICK = 0.2
BUCKETS = [1, 4, 20, 120]                    # 0.5s, 2s, 10s, 60s
BW, CAP = 10, 600                            # scatter bin width / range (10s bucket)
WINDOWS = [(y, m) for y in range(2020, 2027) for m in range(1, 13)
           if (2020, 1) <= (y, m) <= (2026, 5)]
OUT_RES = "/Users/zhuisabella/xn/prediction/ofi_results_full.csv"
OUT_MON = "/Users/zhuisabella/xn/prediction/ofi_permonth.csv"
OUT_BIN = "/Users/zhuisabella/xn/prediction/ofi_scatter_full.csv"


def fetch(sess, code, start, end):
    q = f"""
    pt=loadTable("dfs://hft_future_ts","TickPartitioned")
    select m_nDatetime as ts, m_nBidPrice as pb, m_nBidVolume as qb,
           m_nAskPrice as pa, m_nAskVolume as qa,
           m_nActBidVolume as actb, m_nActAskVolume as acta
    from pt where code_init=`{code[:2]}, code=`{code},
          m_nDatetime>={start}T00:00:00, m_nDatetime<={end}T23:59:59,
          minute(m_nDatetime) between 09:30m:15:00m
    """
    return sess.run(q)


def prep(df):
    df = df.drop_duplicates("ts").sort_values("ts").copy()
    df = df[(df.pb > 0) & (df.pa > 0) & (df.pa >= df.pb)]
    tod = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    df = df[((tod >= 570) & (tod <= 690)) | ((tod >= 780) & (tod <= 900))]
    if df.empty:
        return df
    df["day"] = df["ts"].dt.normalize()
    df["sess"] = np.where(tod <= 690, "AM", "PM")
    g = df.groupby(["day", "sess"], sort=False)
    pb1, qb1 = g["pb"].shift(1), g["qb"].shift(1)
    pa1, qa1 = g["pa"].shift(1), g["qa"].shift(1)
    bid = np.where(df.pb >= pb1, df.qb, 0.0) - np.where(df.pb <= pb1, qb1, 0.0)
    ask = np.where(df.pa <= pa1, df.qa, 0.0) - np.where(df.pa >= pa1, qa1, 0.0)
    df["ofi"] = np.where(pb1.notna().to_numpy(), bid - ask, np.nan)
    df["ti"] = (df["actb"] - df["acta"]).astype(float)
    df["mid_tk"] = (df.pb + df.pa) / (2 * TICK)
    df["dP"] = g["mid_tk"].diff()
    df["gid"] = g.ngroup(); df["idx"] = g.cumcount()
    return df


def bucketize(df, B):
    d = df.dropna(subset=["ofi", "dP"]).copy()
    d["bkt"] = d["idx"] // B
    a = d.groupby(["gid", "bkt"], sort=True).agg(
        OFI=("ofi", "sum"), TI=("ti", "sum"), dP=("dP", "sum")).reset_index()
    a["dP_next"] = a.groupby("gid")["dP"].shift(-1)
    return a


def moments(x, y):
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    return np.array([len(x), x.sum(), y.sum(), (x * y).sum(), (x * x).sum(), (y * y).sum()])


def stats(M):
    n, Sx, Sy, Sxy, Sxx, Syy = M
    denx = n * Sxx - Sx * Sx; deny = n * Syy - Sy * Sy
    if n < 100 or denx <= 0 or deny <= 0:
        return np.nan, np.nan, int(n)
    slope = (n * Sxy - Sx * Sy) / denx
    r2 = (n * Sxy - Sx * Sy) ** 2 / (denx * deny)
    return slope, r2, int(n)


if __name__ == "__main__":
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    acc = {}                     # (code,B,model) -> moment vector
    scat = {}                    # (code,bin) -> [sum_dP, count]
    monrows = []
    for yr, mo in WINDOWS:
        last = calendar.monthrange(yr, mo)[1]
        start, end = f"{yr}.{mo:02d}.01", f"{yr}.{mo:02d}.{last:02d}"
        for code in CODES:
            df = fetch(sess, code, start, end)
            if not len(df):
                continue
            df = prep(df)
            if df.empty:
                continue
            for B in BUCKETS:
                a = bucketize(df, B)
                for model, x, y in [("ofi", a.OFI, a.dP), ("ti", a.TI, a.dP),
                                    ("pred", a.OFI, a.dP_next)]:
                    k = (code, B, model)
                    acc[k] = acc.get(k, np.zeros(6)) + moments(x.to_numpy(), y.to_numpy())
                if B == 20:
                    _, r2m, nm = stats(moments(a.OFI.to_numpy(), a.dP.to_numpy()))
                    monrows.append(dict(code=code, year=yr, month=mo, r2_ofi=r2m, n=nm))
                    aa = a[np.isfinite(a.OFI) & np.isfinite(a.dP) & (a.OFI.abs() <= CAP)]
                    b = (np.round(aa.OFI / BW) * BW).astype(int)
                    for bin_c, dp in zip(b.to_numpy(), aa.dP.to_numpy()):
                        s = scat.setdefault((code, int(bin_c)), [0.0, 0])
                        s[0] += dp; s[1] += 1
        print(f"{yr}-{mo:02d} done", flush=True)
    sess.close()

    rows = []
    for code in CODES:
        for B in BUCKETS:
            so, r2o, n = stats(acc[(code, B, "ofi")])
            st, r2t, _ = stats(acc[(code, B, "ti")])
            sp, r2p, _ = stats(acc[(code, B, "pred")])
            D = 1.0 / (2 * so) if np.isfinite(so) and so != 0 else np.nan
            rows.append(dict(code=code, B=B, secs=B * 0.5, n=n, r2_ofi=r2o, r2_ti=r2t,
                             coef_ofi=so, D=D, r2_pred=r2p))
    pd.DataFrame(rows).to_csv(OUT_RES, index=False)
    pd.DataFrame(monrows).to_csv(OUT_MON, index=False)
    pd.DataFrame([{"code": c, "OFI": b, "dP": v[0] / v[1], "n": v[1]}
                  for (c, b), v in scat.items() if v[1] >= 50]).to_csv(OUT_BIN, index=False)
    print(f"saved {OUT_RES}, {OUT_MON}, {OUT_BIN}")
