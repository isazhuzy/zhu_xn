"""step3_vtrade_ddb.py — GENUINE intraminute V-trade + extended (cross-boundary) curve.
INDEPENDENT experiment (not a continuation): fig26 shows, in the momentum frame, the
signed displacement DROPS through the first part of the minute (trough ~sec 20-30) then
RISES, and the rise extends past the minute close into the next minute. So:
  - SHORT (position -d) from the open to a flip tick F  (capture the down-leg)
  - LONG  (position +d) from F to an exit tick X        (capture the up-leg; X may be >120)
  d = sign(close_{M-1} - close_{M-2}).

To see WHERE the trend stops we build the EXTENDED signed curve over minute M concatenated
with minute M+1 (signed by d_M, aligned by tick from M's open, ~240 ticks). Then F*=trough,
X*=peak (read off the aggregate later). Gross per-minute V-trade pnl for (F,X) tick pair:
  pnl = -d*(comb[F]-comb[0]) + d*(comb[X]-comb[F]) = (-2*y[F] + y[X]) ; y = d*(comb-comb[0]).

Outputs (small window first):
  step3_extcurve.csv : code,year,month,tick,sum,n           (extended signed curve)
  step3_vtrade.csv   : code,year,month,F,X,sum,sumsq,n       (V-trade pnl grid for t-stats)
Run:  /Users/zhuisabella/xn/.venv/bin/python step3_vtrade_ddb.py   (sandbox OFF)
"""
import os
import calendar
import numpy as np
import pandas as pd
import dolphindb as ddb

from ddb_config import HOST, PORT, USER, PW
CODES = ["IC0000", "IF0000", "IH0000", "IM0000"]
MIN_TICKS = 10                       # threshold: minute M must have >=10 ticks
F_GRID = [30, 40, 50, 60, 70]                       # flip tick (short->long); ~sec 15..35
X_GRID = [60, 80, 100, 110, 120, 140, 160, 180, 200, 220, 240]   # exit tick; >120 = into next min
SMALL = os.environ.get("SMALL", "1") == "1"    # SMALL=0 -> full 76-month window
WINDOWS = ([(2024, 4), (2024, 5), (2024, 6)] if SMALL else
           [(y, m) for y in range(2020, 2027) for m in range(1, 13)
            if (2020, 1) <= (y, m) <= (2026, 5)])
OUT_CURVE = "/Users/zhuisabella/xn/intraminute/step3_extcurve.csv"
OUT_VT = "/Users/zhuisabella/xn/intraminute/step3_vtrade.csv"


def fetch(sess, code, start, end):
    q = f"""
    pt=loadTable("dfs://hft_future_ts","TickPartitioned")
    select m_nDatetime as ts, (m_nBidPrice+m_nAskPrice)/2.0 as mid
    from pt where code_init=`{code[:2]}, m_nDatetime>={start}T00:00:00,
          m_nDatetime<={end}T23:59:59, code=`{code},
          minute(m_nDatetime) between 09:30m:15:00m
    """
    return sess.run(q)


def run(df):
    df = df.drop_duplicates("ts").sort_values("ts").copy()
    df["mkey"] = df["ts"].dt.floor("min")
    tod = df["mkey"].dt.hour * 60 + df["mkey"].dt.minute
    df = df[((tod >= 570) & (tod <= 690)) | ((tod >= 780) & (tod <= 900))]
    if df.empty:
        return None

    per = pd.DataFrame({"last": df.groupby("mkey")["mid"].last()})
    per["tod"] = per.index.hour * 60 + per.index.minute
    per["day"] = per.index.normalize()
    per["session"] = np.where(per["tod"] <= 690, "AM", "PM")
    per = per.sort_values(["day", "session", "tod"])
    g = per.groupby(["day", "session"])
    c1, c2 = g["last"].shift(1), g["last"].shift(2)
    t1, t2 = g["tod"].shift(1), g["tod"].shift(2)
    per["d"] = np.where((per["tod"] - t1 == 1) & (t1 - t2 == 1), np.sign(c1 - c2), np.nan)
    dmap = per["d"].to_dict()
    todmap = per["tod"].to_dict()
    daymap = per["day"].to_dict()
    sesmap = per["session"].to_dict()

    paths = {mkey: grp["mid"].to_numpy() for mkey, grp in df.groupby("mkey")}

    vecs = []                                   # extended signed displacement vectors
    vt = {(F, X): [0.0, 0.0, 0] for F in F_GRID for X in X_GRID}   # sum,sumsq,n
    for mkey, pm in paths.items():
        d = dmap.get(mkey, np.nan)
        if not np.isfinite(d) or d == 0 or len(pm) < MIN_TICKS:
            continue
        nxt = mkey + pd.Timedelta(minutes=1)
        pn = paths.get(nxt)
        if pn is None or daymap.get(nxt) != daymap.get(mkey) \
                or sesmap.get(nxt) != sesmap.get(mkey) \
                or todmap.get(nxt, -99) - todmap.get(mkey, 0) != 1:
            continue
        comb = np.concatenate([pm, pn])
        y = d * (comb - comb[0])                # extended signed curve, this minute
        vecs.append(y)
        for F in F_GRID:
            if len(comb) <= F:
                continue
            yF = y[F]                           # y[F] = d*(comb[F]-comb[0])
            for X in X_GRID:
                if len(comb) <= X:
                    continue
                p = -2.0 * yF + y[X]            # short 0->F then long F->X
                rec = vt[(F, X)]
                rec[0] += p; rec[1] += p * p; rec[2] += 1
    if not vecs:
        return None
    L = max(len(v) for v in vecs)
    M = np.full((len(vecs), L), np.nan)
    for i, v in enumerate(vecs):
        M[i, :len(v)] = v
    return np.nanmean(M, axis=0), np.sum(~np.isnan(M), axis=0), len(vecs), vt


if __name__ == "__main__":
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    crows, vrows = [], []
    for yr, mo in WINDOWS:
        last = calendar.monthrange(yr, mo)[1]
        start, end = f"{yr}.{mo:02d}.01", f"{yr}.{mo:02d}.{last:02d}"
        for code in CODES:
            df = fetch(sess, code, start, end)
            if not len(df):
                print(f"  {yr}-{mo:02d} {code}: no data", flush=True); continue
            res = run(df)
            if res is None:
                continue
            mean, n, nsamp, vt = res
            print(f"  {yr}-{mo:02d} {code}: {nsamp} minute-samples, extlen {len(mean)}", flush=True)
            for i, (mv, nv) in enumerate(zip(mean, n), start=1):
                crows.append({"code": code, "year": yr, "month": mo,
                              "tick": i, "sum": mv * int(nv), "n": int(nv)})
            for (F, X), (s, ss, nn) in vt.items():
                vrows.append({"code": code, "year": yr, "month": mo,
                              "F": F, "X": X, "sum": s, "sumsq": ss, "n": nn})
        print(f"{yr}-{mo:02d} done", flush=True)
    sess.close()
    pd.DataFrame(crows).to_csv(OUT_CURVE, index=False)
    pd.DataFrame(vrows).to_csv(OUT_VT, index=False)
    print(f"saved {OUT_CURVE} and {OUT_VT}")
