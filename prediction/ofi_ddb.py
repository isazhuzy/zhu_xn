"""ofi_ddb.py — Cont-Kukanov-Stoikov (2014) Order Flow Imbalance on CFFEX index futures.
L1-snapshot OFI built from best bid/ask (price, size); trade imbalance TI from aggressor
volumes. Regress mid-price change (in ticks) on OFI and on TI, contemporaneously and with
a one-step lag (the 'prediction' test). Small window first: 2024-06, all 4 contracts.

Per-snapshot OFI increment (Cont et al., L1 version), for consecutive snapshots n-1 -> n:
  bid = q^b_n * 1[Pb_n >= Pb_{n-1}]  -  q^b_{n-1} * 1[Pb_n <= Pb_{n-1}]
  ask = q^a_n * 1[Pa_n <= Pa_{n-1}]  -  q^a_{n-1} * 1[Pa_n >= Pa_{n-1}]
  OFI_n = bid - ask
Aggregate over a bucket of B snapshots: OFI_k = sum(OFI_n). Model: dP_k = OFI_k/(2D) + e_k.
TI_k = sum(ActBidVol - ActAskVol) = M^b - M^s;  dP_k = TI_k/(2D) + eta_k.
Run: /Users/zhuisabella/xn/.venv/bin/python ofi_ddb.py   (sandbox OFF)
"""
import numpy as np, pandas as pd, dolphindb as ddb
from ddb_config import HOST, PORT, USER, PW
CODES = ["IC0000", "IF0000", "IH0000", "IM0000"]
TICK = 0.2                                  # CFFEX index-futures tick size (index points)
BUCKETS = [1, 4, 20, 120]                   # snapshots per bucket: 0.5s, 2s, 10s, 60s
START, END = "2024.06.01", "2024.06.30"
OUT_RES = "/Users/zhuisabella/xn/prediction/ofi_results.csv"
OUT_BIN = "/Users/zhuisabella/xn/prediction/ofi_scatter.csv"   # binned dP vs OFI at 10s


def fetch(sess, code):
    q = f"""
    pt=loadTable("dfs://hft_future_ts","TickPartitioned")
    select m_nDatetime as ts, m_nBidPrice as pb, m_nBidVolume as qb,
           m_nAskPrice as pa, m_nAskVolume as qa,
           m_nActBidVolume as actb, m_nActAskVolume as acta
    from pt where code_init=`{code[:2]}, code=`{code},
          m_nDatetime>={START}T00:00:00, m_nDatetime<={END}T23:59:59,
          minute(m_nDatetime) between 09:30m:15:00m
    """
    return sess.run(q)


def reg(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 100 or x.std() == 0:
        return np.nan, np.nan, len(x)
    b1, b0 = np.polyfit(x, y, 1)
    r2 = np.corrcoef(x, y)[0, 1] ** 2
    return b1, r2, len(x)


def prep(df):
    df = df.drop_duplicates("ts").sort_values("ts").copy()
    df = df[(df.pb > 0) & (df.pa > 0) & (df.pa >= df.pb)]        # sane quotes
    tod = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    df = df[((tod >= 570) & (tod <= 690)) | ((tod >= 780) & (tod <= 900))]
    df["day"] = df["ts"].dt.normalize()
    df["sess"] = np.where(tod <= 690, "AM", "PM")
    g = df.groupby(["day", "sess"], sort=False)
    pb1, qb1 = g["pb"].shift(1), g["qb"].shift(1)
    pa1, qa1 = g["pa"].shift(1), g["qa"].shift(1)
    bid = np.where(df.pb >= pb1, df.qb, 0.0) - np.where(df.pb <= pb1, qb1, 0.0)
    ask = np.where(df.pa <= pa1, df.qa, 0.0) - np.where(df.pa >= pa1, qa1, 0.0)
    df["ofi"] = bid - ask                                       # NaN where pb1 is NaN (session start)
    df["ofi"] = np.where(pb1.notna().to_numpy(), df["ofi"], np.nan)
    df["ti"] = (df["actb"] - df["acta"]).astype(float)
    df["mid_tk"] = (df.pb + df.pa) / (2 * TICK)                 # mid price in ticks
    df["dP"] = g["mid_tk"].diff()                               # per-snapshot mid change (ticks)
    df["gid"] = g.ngroup()
    df["idx"] = g.cumcount()
    return df


def bucketize(df, B):
    d = df.dropna(subset=["ofi", "dP"]).copy()
    d["bkt"] = d["idx"] // B
    agg = d.groupby(["gid", "bkt"], sort=True).agg(
        OFI=("ofi", "sum"), TI=("ti", "sum"), dP=("dP", "sum")).reset_index()
    return agg


if __name__ == "__main__":
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    rows, binrows = [], []
    for code in CODES:
        df = prep(fetch(sess, code))
        for B in BUCKETS:
            a = bucketize(df, B)
            b_ofi, r2_ofi, n = reg(a["OFI"].to_numpy(), a["dP"].to_numpy())
            b_ti, r2_ti, _ = reg(a["TI"].to_numpy(), a["dP"].to_numpy())
            D = 1.0 / (2 * b_ofi) if np.isfinite(b_ofi) and b_ofi != 0 else np.nan
            # predictive: OFI_k -> dP_{k+1} (within same gid)
            a2 = a.copy(); a2["dP_next"] = a2.groupby("gid")["dP"].shift(-1)
            _, r2_pred, npred = reg(a2["OFI"].to_numpy(), a2["dP_next"].to_numpy())
            rows.append(dict(code=code, B=B, secs=B * 0.5, n=n, r2_ofi=r2_ofi, r2_ti=r2_ti,
                             coef_ofi=b_ofi, D=D, r2_pred=r2_pred))
            print(f"{code} B={B:>3}({B*0.5:>4.1f}s) n={n:>6}  R2(OFI)={r2_ofi:.3f}  "
                  f"R2(TI)={r2_ti:.3f}  D={D:>7.1f}  R2(pred,OFI->next)={r2_pred:.4f}", flush=True)
            if B == 20:                                        # save 10s binned scatter
                a = a[np.isfinite(a.OFI) & np.isfinite(a.dP)]
                q = pd.qcut(a["OFI"], 40, duplicates="drop")
                bb = a.groupby(q, observed=True).agg(OFI=("OFI", "mean"), dP=("dP", "mean"),
                                                     n=("dP", "size")).reset_index(drop=True)
                bb["code"] = code; binrows.append(bb)
    sess.close()
    pd.DataFrame(rows).to_csv(OUT_RES, index=False)
    pd.concat(binrows).to_csv(OUT_BIN, index=False)
    print(f"\nsaved {OUT_RES} and {OUT_BIN}")
