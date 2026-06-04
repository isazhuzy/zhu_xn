"""
trend_research.py
=================
Extensions and corrections to the commodity trend-following metric suite.

Context
-------
The original suite scores each commodity on four descriptive axes:
    trend Sharpe        reward (but see note below)
    noise ratio         signal/noise smoothness     (low = smoother)
    drawdown structure  path cleanliness            (low = cleaner)
    gap ratio           continuity / jump risk      (low = continuous)

This module keeps that spirit but adds the pieces that turn a *descriptive*
study into an *actionable* one, and fixes several issues that bias the
original conclusions:

  ISSUE 1  `compute_trend_sharpe` measures the buy-&-hold Sharpe of the
           underlying, NOT the Sharpe of a trend strategy. A market can
           trend hard in BOTH directions (great for trend following) yet
           have a near-zero buy-&-hold Sharpe. -> `tsmom_sharpe` below.

  ISSUE 2  `noise ratio` docstring says ATR14/price but the code uses a
           20d return std. Also move20 can be ~0, exploding the ratio.
           -> documented + winsorized in `noise_ratio`.

  ISSUE 3  `drawdown_structure_ratio` divides by net move; a contract that
           round-trips (ends near its start) gives net~0 -> ratio blows up
           or is dropped, silently biasing the cross-section toward
           one-directional names. -> floored in `drawdown_structure_ratio`.

  ISSUE 4  `gap ratio` uses a fixed 2% threshold. High-vol contracts will
           look "jumpy" purely because they are volatile, not discontinuous.
           -> `gap_ratio` normalizes by each contract's own volatility.

  ISSUE 5  No sample-length control. ~70% of the 62 Wind indices have far
           less than the full 2010-2026 history; several short, recent
           contracts dominate the naive Sharpe ranking on tiny samples.
           -> `min_obs` filtering throughout.

  ISSUE 6  No out-of-sample validation. A metric is only useful for
           SELECTION if it predicts future behaviour. -> `predictive_validity`.

Run directly:  python trend_research.py /path/to/wind.xlsx
"""

from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import math

try:
    from scipy.stats import spearmanr
except Exception:                                   # graceful degrade
    spearmanr = None


# --------------------------------------------------------------------------- #
#  Data loading (Wind 品种指数 export: 4 metadata rows, then Date + tickers)   #
# --------------------------------------------------------------------------- #
def load_wind_prices(path: str, skiprows: int = 4) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=0, skiprows=skiprows)
    raw = raw.rename(columns={raw.columns[0]: "Date"})
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date"]).set_index("Date").sort_index()
    return raw.apply(pd.to_numeric, errors="coerce")


# --------------------------------------------------------------------------- #
#  1. PROPER trend-strategy Sharpe  (the metric the original "trend Sharpe"    #
#     should arguably be).  Time-series momentum, vol-targeted, no look-ahead. #
# --------------------------------------------------------------------------- #
def tsmom_sharpe(
    close: pd.Series,
    lookback: int = 120,
    vol_lookback: int = 60,
    vol_target: float = 0.15,
    max_leverage: float = 3.0,
) -> float:
    """
    Sharpe of a vol-targeted time-series-momentum strategy on one contract.

    position_t = sign(return over `lookback`) * (vol_target / realized_vol),
    capped at `max_leverage`, lagged one day to remove look-ahead.
    """
    close = close.dropna()
    if len(close) < lookback + vol_lookback + 40:
        return np.nan
    ret = close.pct_change()
    signal = np.sign(close.pct_change(lookback))
    realized_vol = ret.rolling(vol_lookback).std() * np.sqrt(252)
    scale = (vol_target / realized_vol).clip(upper=max_leverage)
    strat = ((signal * scale).shift(1) * ret).dropna()
    if len(strat) < 60 or strat.std() == 0:
        return np.nan
    return np.sqrt(252) * strat.mean() / strat.std()


def tsmom_sharpes(df: pd.DataFrame, min_obs: int = 500, **kw) -> pd.Series:
    out = {c: tsmom_sharpe(df[c], **kw) for c in df.columns
           if df[c].notna().sum() >= min_obs}
    return pd.Series(out).dropna().sort_values()


# --------------------------------------------------------------------------- #
#  2. DIRECT trendiness statistic — variance ratio (Lo–MacKinlay style).       #
#     VR>1 => positive autocorrelation (trending); VR<1 => mean-reverting.     #
# --------------------------------------------------------------------------- #
def variance_ratio(close: pd.Series, q: int = 10) -> float:
    logp = np.log(close.dropna())
    r1 = logp.diff().dropna()
    if len(r1) < q * 5:
        return np.nan
    var_1 = r1.var(ddof=1)
    var_q = logp.diff(q).dropna().var(ddof=1)
    return var_q / (q * var_1) if var_1 > 0 else np.nan


def variance_ratios(df: pd.DataFrame, q: int = 10, min_obs: int = 500) -> pd.Series:
    out = {c: variance_ratio(df[c], q) for c in df.columns
           if df[c].notna().sum() >= min_obs}
    return pd.Series(out).dropna().sort_values()


# --------------------------------------------------------------------------- #
#  3. Corrected versions of the original descriptive metrics                   #
# --------------------------------------------------------------------------- #
def noise_ratio(close: pd.Series, winsor: float = 0.99) -> float:
    """20d vol / 20d directional move, winsorized to tame move20~0 blow-ups."""
    c = close.dropna()
    move20 = c.pct_change(20).abs()
    ratio = (c.pct_change().rolling(20).std() / move20)
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    if ratio.empty:
        return np.nan
    return ratio.clip(upper=ratio.quantile(winsor)).median()


def drawdown_structure_ratio(close: pd.Series, net_floor: float = 0.05) -> float:
    """
    |max drawdown| / |net move|, with the denominator floored so contracts
    that round-trip (net~0) are not assigned an exploding/NaN score.
    """
    c = close.dropna()
    if len(c) < 2:
        return np.nan
    nav = c / c.iloc[0]
    max_dd = abs((nav / nav.cummax() - 1).min())
    net_move = max(abs(c.iloc[-1] / c.iloc[0] - 1), net_floor)
    return max_dd / net_move


def gap_ratio(close: pd.Series, k: float = 3.0, vol_lookback: int = 60) -> float:
    """
    Vol-normalized gap ratio: fraction of days whose move exceeds k * its own
    trailing daily vol. Removes the 'high vol == jumpy' confound of a fixed %.
    """
    c = close.dropna()
    ret = c.pct_change()
    rolling_sd = ret.rolling(vol_lookback).std()
    return (ret.abs() > k * rolling_sd).mean()


# --------------------------------------------------------------------------- #
#  4. Composite trend-friendliness score (cross-sectional z, sign-aligned)     #
# --------------------------------------------------------------------------- #
def composite_score(df: pd.DataFrame, min_obs: int = 500) -> pd.DataFrame:
    cols = [c for c in df.columns if df[c].notna().sum() >= min_obs]
    m = pd.DataFrame({
        "tsmom":  pd.Series({c: tsmom_sharpe(df[c]) for c in cols}),
        "vr":     pd.Series({c: variance_ratio(df[c]) for c in cols}),
        "noise":  pd.Series({c: noise_ratio(df[c]) for c in cols}),
        "dd":     pd.Series({c: drawdown_structure_ratio(df[c]) for c in cols}),
        "gap":    pd.Series({c: gap_ratio(df[c]) for c in cols}),
    }).dropna()

    # sign: + means more trend-friendly
    signs = {"tsmom": 1, "vr": 1, "noise": -1, "dd": -1, "gap": -1}
    z = (m - m.mean()) / m.std()
    for k, s in signs.items():
        z[k] *= s
    m["score"] = z.mean(axis=1)
    return m.sort_values("score", ascending=False)


# --------------------------------------------------------------------------- #
#  5. THE validation step: does an in-sample metric predict OOS trend profit?  #
# --------------------------------------------------------------------------- #
def _spearman(x, y):
    """Spearman rank correlation + approximate two-sided p-value (no scipy)."""
    rx = pd.Series(np.asarray(x, float)).rank().to_numpy()
    ry = pd.Series(np.asarray(y, float)).rank().to_numpy()
    rho = np.corrcoef(rx, ry)[0, 1]
    n = len(rx)
    if n > 2 and abs(rho) < 1:
        t = rho * math.sqrt((n - 2) / (1 - rho ** 2))
        # normal approximation to the t-tail; fine for n >= ~30
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    else:
        p = float("nan")
    return rho, p


def predictive_validity(df: pd.DataFrame, split: str = "2018-12-31",
                        min_obs: int = 400) -> pd.DataFrame:
    IS, OOS = df.loc[:split], df.loc[split:]
    rows = []
    for c in df.columns:
        if IS[c].dropna().shape[0] < min_obs or OOS[c].dropna().shape[0] < min_obs:
            continue
        rows.append(dict(
            t=c,
            is_vr=variance_ratio(IS[c]),
            is_noise=noise_ratio(IS[c]),
            is_dd=drawdown_structure_ratio(IS[c]),
            is_tsmom=tsmom_sharpe(IS[c]),
            oos_tsmom=tsmom_sharpe(OOS[c]),
        ))
    R = pd.DataFrame(rows).dropna()
    specs = [("is_vr", 1), ("is_noise", -1), ("is_dd", -1), ("is_tsmom", 1)]
    res = []
    for col, sign in specs:
        rho, p = _spearman(sign * R[col], R["oos_tsmom"])
        res.append(dict(metric=col, rank_corr=rho, p_value=p, n=len(R)))
    return pd.DataFrame(res)

# --------------------------------------------------------------------------- #
#  6. Diversified basket vs single-name selection                              #
# --------------------------------------------------------------------------- #
def tsmom_daily(close, lookback=120, vol_lookback=60, vol_target=0.15, max_leverage=3.0):
    ret = close.pct_change()
    signal = np.sign(close.pct_change(lookback))
    rv = ret.rolling(vol_lookback).std() * np.sqrt(252)
    return (signal * (vol_target / rv).clip(upper=max_leverage)).shift(1) * ret


def basket_vs_single(df: pd.DataFrame, start: str = "2019-01-01", min_obs: int = 500):
    cols = [c for c in df.columns if df[c].loc[start:].dropna().shape[0] >= min_obs]
    sr = pd.DataFrame({c: tsmom_daily(df[c]) for c in cols}).loc[start:]
    basket = sr.mean(axis=1).dropna()
    sharpe = lambda x: np.sqrt(252) * x.mean() / x.std()
    singles = sr.apply(lambda x: sharpe(x.dropna()))
    return {
        "basket_sharpe": sharpe(basket),
        "avg_single_sharpe": singles.mean(),
        "basket_annvol": basket.std() * np.sqrt(252),
        "avg_single_annvol": (sr.std() * np.sqrt(252)).mean(),
        "n_contracts": len(cols),
    }


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "wind.xlsx"
    df = load_wind_prices(path)
    print(f"Loaded {df.shape[1]} contracts, {df.shape[0]} days "
          f"({df.index.min().date()} -> {df.index.max().date()})\n")

    print(">>> TSMOM strategy Sharpe (top 8)")
    print(tsmom_sharpes(df).tail(8).round(2), "\n")

    print(">>> Variance ratio (q=10), most trending (top 6)")
    print(variance_ratios(df).tail(6).round(2), "\n")

    print(">>> Composite trend-friendliness score (top 8)")
    print(composite_score(df)["score"].head(8).round(2), "\n")

    print(">>> Predictive validity (IS<=2018 metric -> OOS>=2019 TSMOM Sharpe)")
    print(predictive_validity(df).round(3), "\n")

    print(">>> Diversified basket vs single-name (2019+)")
    for k, v in basket_vs_single(df).items():
        print(f"    {k:20s} {v:.3f}" if isinstance(v, float) else f"    {k:20s} {v}")
