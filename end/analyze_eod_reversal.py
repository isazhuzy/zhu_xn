"""
analyze_eod_reversal.py — end-of-day reversal after a >=1% intraday rally.

Question: on a day where price has risen >=1% from the open at some point, is
there a specific time-of-day (before the 15:00 close) after which the market
tends to give back the gain (reversal into the close)?

Design (event-conditional, day-clustered — avoids pseudo-replication from
autocorrelated within-day minutes):
  open(day)   = close at 09:30
  cumret(day,t) = close(day,t)/open(day) - 1
  cond(day,t) = cumret(day,t) >= THRESH   (day is up >=1% AS OF minute t)
  fwd(day,t)  = close(day,15:00)/close(day,t) - 1     (forward return, t -> close)

For each contract and each minute-of-day t, take the sample of days where
cond(day,t) holds, and summarize fwd(day,t)*1e4 (bp): n, mean, win-rate (frac
fwd<0, i.e. "reversed"), t-stat (day-level, one obs/day). Scanning t across the
session shows whether/where the reversal signal concentrates, especially in
the last hour before close (13:00-14:55) which is the user's specific ask.

Run:  /Users/zhuisabella/xn/.venv/bin/python analyze_eod_reversal.py
"""
import numpy as np
import pandas as pd

IN = "/Users/zhuisabella/xn/end/day_minutes_full.csv"
THRESH = 0.01
CODES = ["IC0000", "IF0000", "IH0000", "IM0000"]
OUT_SCAN = "/Users/zhuisabella/xn/end/scan_by_minute.csv"
OUT_YEAR = "/Users/zhuisabella/xn/end/scan_by_year.csv"


def load():
    df = pd.read_csv(IN)
    df["d"] = pd.to_datetime(df["d"])
    mn = pd.to_datetime(df["mn"])
    df["tod"] = mn.dt.hour * 60 + mn.dt.minute
    df["year"] = df["d"].dt.year
    return df


def build_panel(df, code):
    sub = df[df["code"] == code]
    piv = sub.pivot_table(index="d", columns="tod", values="close")
    piv = piv.sort_index()
    return piv


def day_stats(piv):
    """Return open, close(15:00), and cumret matrix (day x tod)."""
    open_ = piv[570]  # 09:30
    close_eod = piv[900]  # 15:00
    cumret = piv.div(open_, axis=0) - 1.0
    fwd = close_eod.to_numpy()[:, None] / piv.to_numpy() - 1.0
    fwd = pd.DataFrame(fwd, index=piv.index, columns=piv.columns)
    return open_, close_eod, cumret, fwd


def scan_minute(cumret, fwd, tod_list, year=None, years_of=None):
    rows = []
    for t in tod_list:
        if t not in cumret.columns or t not in fwd.columns:
            continue
        x = cumret[t]
        y = fwd[t] * 1e4
        mask = x >= THRESH
        if years_of is not None and year is not None:
            mask = mask & (years_of == year)
        yy = y[mask].dropna()
        if len(yy) < 5:
            continue
        m = yy.mean()
        s = yy.std(ddof=1)
        t_stat = m / (s / np.sqrt(len(yy))) if s > 0 else np.nan
        rows.append(dict(tod=t, n=len(yy), mean_bp=round(m, 3),
                          win_reversal=round((yy < 0).mean(), 3), t=round(t_stat, 2)))
    return pd.DataFrame(rows)


def tod_to_hm(t):
    return f"{t // 60:02d}:{t % 60:02d}"


if __name__ == "__main__":
    df = load()
    all_tod = sorted(df["tod"].unique())
    # afternoon focus per the question: before 15:00, i.e. tod in [780, 899] (13:00..14:59)
    afternoon = [t for t in all_tod if 780 <= t <= 899]

    scans = []
    for code in CODES:
        piv = build_panel(df, code)
        open_, close_eod, cumret, fwd = day_stats(piv)
        s = scan_minute(cumret, fwd, afternoon)
        s["code"] = code
        # how many days ever qualify (day is up>=1% at ANY point in the session)?
        ever = (cumret[all_tod] >= THRESH).any(axis=1).sum()
        s["days_ever_up1pct"] = ever
        scans.append(s)
        print(f"\n=== {code} : afternoon scan (n days up>=1% at any point = {ever}) ===")
        s2 = s.copy()
        s2["hm"] = s2["tod"].map(tod_to_hm)
        print(s2[["hm", "n", "mean_bp", "win_reversal", "t"]].to_string(index=False))

    full = pd.concat(scans, ignore_index=True)
    full["hm"] = full["tod"].map(tod_to_hm)
    full.to_csv(OUT_SCAN, index=False)
    print("\nsaved", OUT_SCAN)

    # cross-year robustness on each contract's single most negative/strongest bp minute
    print("\n=== cross-year robustness of best minute per contract ===")
    year_rows = []
    for code in CODES:
        piv = build_panel(df, code)
        open_, close_eod, cumret, fwd = day_stats(piv)
        best = full[full.code == code].sort_values("t").iloc[0]
        t_best = int(best["tod"])
        years_of = pd.Series(cumret.index.year, index=cumret.index)
        yrs = sorted(years_of.unique())
        for yr in yrs:
            s = scan_minute(cumret, fwd, [t_best], year=yr, years_of=years_of)
            if len(s) == 0:
                continue
            s["code"] = code
            s["year"] = yr
            s["hm"] = tod_to_hm(t_best)
            year_rows.append(s)
        print(f"{code}: best minute = {tod_to_hm(t_best)} (full-history mean={best.mean_bp}bp t={best.t}, n={best.n})")
    yeartab = pd.concat(year_rows, ignore_index=True)
    yeartab.to_csv(OUT_YEAR, index=False)
    print(yeartab[["code", "year", "hm", "n", "mean_bp", "win_reversal", "t"]].to_string(index=False))
    print("\nsaved", OUT_YEAR)
