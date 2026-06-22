"""minute_curves_raw_multi_ddb.py — fig20-style RAW (un-normalized) grand-average curves
across several (year, month) windows. Buy-and-hold, aligned by real tick index (no interp),
same sample filter as fig19/fig20. One CSV per month -> raw_months/minute_curves_raw_<YYYY_MM>.csv.
Run:  /Users/zhuisabella/xn/.venv/bin/python minute_curves_raw_multi_ddb.py   (sandbox off)
"""
import os
import calendar
import numpy as np
import pandas as pd
import dolphindb as ddb

from ddb_config import HOST, PORT, USER, PW
CODES = ["IC0000", "IF0000", "IH0000", "IM0000"]
MIN_TICKS = 2          # was 10; lowered to the math floor (need >=2 ticks for a path)
# full monthly coverage: every month 2020-01 .. 2026-05
WINDOWS = [(y, m) for y in range(2020, 2027) for m in range(1, 13)
           if (2020, 1) <= (y, m) <= (2026, 5)]
SKIP_EXISTING = True
OUTDIR = "/Users/zhuisabella/xn/intraminute/raw_months"
os.makedirs(OUTDIR, exist_ok=True)


def fetch(sess, code, start, end):
    q = f"""
    pt=loadTable("dfs://hft_future_ts","TickPartitioned")
    select m_nDatetime as ts, (m_nBidPrice+m_nAskPrice)/2.0 as mid
    from pt where code_init=`{code[:2]}, m_nDatetime>={start}T00:00:00,
          m_nDatetime<={end}T23:59:59, code=`{code},
          minute(m_nDatetime) between 09:30m:15:00m
    """
    return sess.run(q)


def raw_curve(df):
    df = df.drop_duplicates("ts").sort_values("ts").copy()   # dedupe (fixes 2024-02 doubled rows)
    df["mkey"] = df["ts"].dt.floor("min")
    tod = df["mkey"].dt.hour * 60 + df["mkey"].dt.minute
    df = df[((tod >= 570) & (tod <= 690)) | ((tod >= 780) & (tod <= 900))]
    if df.empty:
        return None
    per = pd.DataFrame({"last": df.groupby("mkey")["mid"].last()})
    per["day"] = per.index.normalize()
    per["tod"] = per.index.hour * 60 + per.index.minute
    per["session"] = np.where(per["tod"] <= 690, "AM", "PM")
    per = per.sort_values(["day", "session", "tod"])
    g = per.groupby(["day", "session"])
    c1, c2 = g["last"].shift(1), g["last"].shift(2)
    t1, t2 = g["tod"].shift(1), g["tod"].shift(2)
    per["d"] = np.where((per["tod"] - t1 == 1) & (t1 - t2 == 1), np.sign(c1 - c2), np.nan)
    dmap = per["d"].to_dict()

    vecs = []
    for mkey, grp in df.groupby("mkey"):
        d = dmap.get(mkey, np.nan)
        if not np.isfinite(d) or d == 0:
            continue
        path = grp["mid"].to_numpy()
        if len(path) < MIN_TICKS:
            continue
        vecs.append(d * (path - path[0]))     # momentum timing: signal ±1 × displacement
    if not vecs:
        return None
    L = max(len(v) for v in vecs)
    M = np.full((len(vecs), L), np.nan)
    for i, v in enumerate(vecs):
        M[i, :len(v)] = v
    return np.nanmean(M, axis=0), np.sum(~np.isnan(M), axis=0), len(vecs)


if __name__ == "__main__":
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    for (yr, mo) in WINDOWS:
        out = f"{OUTDIR}/minute_curves_raw_{yr}_{mo:02d}.csv"
        if SKIP_EXISTING and os.path.exists(out):
            print(f"\n=== {yr}-{mo:02d}  already exists — skip ===", flush=True)
            continue
        last = calendar.monthrange(yr, mo)[1]
        start, end = f"{yr}.{mo:02d}.01", f"{yr}.{mo:02d}.{last:02d}"
        print(f"\n=== {yr}-{mo:02d}  ({start}..{end}) ===", flush=True)
        rows = []
        for code in CODES:
            df = fetch(sess, code, start, end)
            days = df['ts'].dt.normalize().nunique() if len(df) else 0
            if not len(df):
                print(f"  {code}: no data — skip", flush=True); continue
            res = raw_curve(df)
            if res is None:
                print(f"  {code}: {len(df)} ticks but no valid minutes — skip", flush=True); continue
            mean, n, nsamp = res
            print(f"  {code}: {len(df)} ticks, {days} days, {nsamp} minute-samples", flush=True)
            for i, (mv, nv) in enumerate(zip(mean, n), start=1):
                rows.append({"code": code, "tick": i, "mean": mv, "n": int(nv)})
        if rows:
            out = f"{OUTDIR}/minute_curves_raw_{yr}_{mo:02d}.csv"
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"  saved {out}", flush=True)
    sess.close()
    print("\nALL DONE")
