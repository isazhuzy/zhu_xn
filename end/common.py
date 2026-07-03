"""Shared panel-building helpers for the xn/end reversal studies."""
import numpy as np
import pandas as pd

IN = "/Users/zhuisabella/xn/end/day_minutes_full.csv"
CODES = ["IC0000", "IF0000", "IH0000", "IM0000"]


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
    return piv.sort_index()


def day_stats(piv):
    """open, close(15:00), cumret (day x tod), fwd (day x tod, t->close)."""
    open_ = piv[570]        # 09:30
    close_eod = piv[900]    # 15:00
    cumret = piv.div(open_, axis=0) - 1.0
    fwd = close_eod.to_numpy()[:, None] / piv.to_numpy() - 1.0
    fwd = pd.DataFrame(fwd, index=piv.index, columns=piv.columns)
    return open_, close_eod, cumret, fwd


def running_max(cumret):
    """running intraday max of cumret up to and including t, per day."""
    return cumret.cummax(axis=1)


def rolling_change(cumret, back_min=30):
    """cumret(t) - cumret(t-back_min): how much of the move happened recently."""
    cols = sorted(cumret.columns)
    shifted = cumret.reindex(columns=[c - back_min for c in cols])
    shifted.columns = cols
    return cumret - shifted


def day_range(piv):
    """(intraday max - intraday min) / open, one value per day."""
    open_ = piv[570]
    return (piv.max(axis=1) - piv.min(axis=1)) / open_


def path_ratio(piv):
    """路程/位移比 (path-length / displacement), real-time-computable (only uses
    data up to t, no look-ahead): path(t) = cumsum of |step-to-step price
    change| from 09:30 to t; displacement(t) = |price(t) - open| (net move
    from open), both in raw price points so they're on the same scale.
    Ratio >= 1 always; 1 = perfectly straight-line move to t, larger = more
    back-and-forth (noisy) for the same net gain. NaN/inf where displacement
    ~ 0 (guarded by the caller's cumret>=thresh mask anyway)."""
    steps = piv.diff(axis=1).abs()
    path = steps.cumsum(axis=1)
    disp = piv.sub(piv[570], axis=0).abs()
    ratio = path / disp
    return ratio.replace([np.inf, -np.inf], np.nan)


def scan_minute(mask, fwd, tod_list, day_filter=None):
    """day-clustered per-minute scan: mask/fwd are day x tod DataFrames."""
    rows = []
    for t in tod_list:
        if t not in mask.columns or t not in fwd.columns:
            continue
        m = mask[t].fillna(False)
        if day_filter is not None:
            m = m & day_filter
        y = (fwd[t] * 1e4)[m].dropna()
        if len(y) < 5:
            continue
        mean = y.mean()
        s = y.std(ddof=1)
        t_stat = mean / (s / np.sqrt(len(y))) if s > 0 else np.nan
        rows.append(dict(tod=t, n=len(y), mean_bp=round(mean, 3),
                          win_reversal=round((y < 0).mean(), 3), t=round(t_stat, 2)))
    return pd.DataFrame(rows)


def tod_to_hm(t):
    return f"{t // 60:02d}:{t % 60:02d}"


def year_breakdown(mask, fwd, t_best, years_index):
    rows = []
    for yr in sorted(years_index.unique()):
        yr_mask = years_index == yr
        s = scan_minute(mask, fwd, [t_best], day_filter=yr_mask)
        if len(s):
            s["year"] = yr
            rows.append(s)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
