import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
containing all the metrics computation
"""
#===========================================================
# trend sharpe
#===========================================================
def compute_trend_sharpe(
    df: pd.DataFrame
) -> pd.Series:
    """
    Annualized Sharpe Ratio.

    Sharpe =
    sqrt(252) * mean(daily return)
                / std(daily return)
    
    returns scale linearly with time, volatility scales with the sqrt of time
    """

    ret_1d = df.pct_change() #computing returns (P_t -P_{t-1})/P_{t-1}

    sharpe = (
        np.sqrt(252)
        * ret_1d.mean()
        / ret_1d.std()
    )

    return sharpe.sort_values()

def compute_trend_sharpe_period(
    df: pd.DataFrame,
    start_year: int,
    end_year: int
) -> pd.Series:
    """
    Compute trend Sharpe over a specific period.
    """

    period_df = df.loc[
        f"{start_year}-01-01":
        f"{end_year}-12-31"
    ]

    return compute_trend_sharpe(
        period_df
    )

def compute_yearly_sharpe(
    close: pd.Series
    ) -> pd.Series:
    """
    Compute yearly Sharpe ratio
    for one commodity.
    """

    results = {}
    years = sorted(
        close.index.year.unique()
    )

    for year in years:
        year_data = (
            close[
                close.index.year == year
            ]
        )
        if len(year_data) < 20:
            continue
        ret = year_data.pct_change()
        sharpe = (
            np.sqrt(252)
            * ret.mean()
            / ret.std()
        )
        results[year] = sharpe

    return pd.Series(results)
#===========================================================
# noise ratio
#===========================================================
def compute_noise_ratio(
    df: pd.DataFrame
) -> pd.Series:
    """
    Compute close-only noise ratio.

    Lower = smoother trend.
    Higher = noisier trend.

    noise ratio = ATR percentage / 20-day directional move
    ATR % = ATR_{14} / price
    directional move = abs(P_t - P_{t-20} / P_{t-20})
    """

    ret = df.pct_change()

    vol20 = ret.rolling(20).std()

    move20 = (
        df
        .pct_change(20)
        .abs()
    )

    noise_ratio = (
        vol20
        / move20
    ).median()

    return noise_ratio.sort_values()

def compute_yearly_noise_ratio(
    close: pd.Series
) -> pd.Series:
    """
    Compute yearly noise ratio
    for one commodity.
    """

    results = {}

    years = sorted(
        close.index.year.unique()
    )

    for year in years:

        year_data = (
            close[
                close.index.year == year
            ]
        )

        if len(year_data) < 30:
            continue

        ret = year_data.pct_change()

        vol20 = (
            ret
            .rolling(20)
            .std()
        )

        move20 = (
            year_data
            .pct_change(20)
            .abs()
        )

        noise_ratio = (
            vol20
            / move20
        ).median()

        results[year] = noise_ratio

    return pd.Series(results)

def compute_noise_ratio_period(
    df: pd.DataFrame,
    start_year: int,
    end_year: int
    ) -> pd.Series:
    """
    Compute noise ratio over a specific period.
    """

    period_df = df.loc[
        f"{start_year}-01-01":
        f"{end_year}-12-31"
    ]

    return compute_noise_ratio(
        period_df
    )


#===========================================================
# drawdown
#===========================================================
def compute_drawdown_structure_ratio(
    close: pd.Series
) -> float:
    """
    Compute drawdown structure ratio
    for a single commodity.

    drawdown-structure ratio = abs(max_drawdown) / abs(net_move)
    calculating how much pain was endured per unit of net travel

    lower = cleaner
    """

    nav = (
        close
        / close.iloc[0]
    )

    drawdown = (
        nav
        / nav.cummax() #cumulative maximum
        - 1
    )

    max_drawdown = drawdown.min()

    net_move = abs(
        close.iloc[-1]
        / close.iloc[0]
        - 1
    )

    if net_move == 0:
        return np.nan

    return (
        abs(max_drawdown)
        / net_move
    )

def compute_drawdown_structure_ratios(
    df: pd.DataFrame
    ) -> pd.Series:
    """
    calculating drawdown ratios of all commodities
    """

    ratios = {}

    for col in df.columns:
        close = df[col].dropna()
        if len(close) < 2:
            continue
        ratios[col] = (
            compute_drawdown_structure_ratio(
                close
            )
        )
    return (
        pd.Series(ratios)
        .sort_values()
    )

def compute_yearly_drawdown_ratio(
    close: pd.Series
    ) -> pd.Series:
    """
    Compute yearly drawdown structure ratio.
    """

    results = {}

    years = sorted(
        close.index.year.unique()
    )

    for year in years:

        year_data = (
            close[
                close.index.year == year
            ]
        )

        if len(year_data) < 30:
            continue

        ratio = (
            compute_drawdown_structure_ratio(
                year_data
            )
        )

        results[year] = ratio

    return pd.Series(results)

def compute_drawdown_ratio_period(
    df: pd.DataFrame,
    start_year: int,
    end_year: int
    ) -> pd.Series:
    """
    Compute drawdown ratio
    over a specific period.
    """

    period_df = df.loc[
        f"{start_year}-01-01":
        f"{end_year}-12-31"
    ]

    return (
        compute_drawdown_structure_ratios(
            period_df
        )
    )
#===========================================================
# gap ratio
#===========================================================

def compute_gap_ratio(
    close: pd.Series,
    threshold: float = 0.02
    ) -> float:
    """
    Compute gap ratio for a single commodity.

    Parameters
    ----------
    close : pd.Series
        Price series.

    threshold : float
        Daily move threshold regarded
        as a "gap".

    Returns
    -------
    float
        Fraction of days whose absolute
        return exceeds threshold.
    """

    gaps = (
        close
        .pct_change()
        .abs()
        > threshold
    )

    return gaps.mean()

def compute_gap_ratios(
    df: pd.DataFrame,
    threshold: float = 0.02
    ) -> pd.Series:
    """
    Compute gap ratio
    for every commodity.

    smaller = more continuous
    """

    ratios = {}

    for col in df.columns:

        close = df[col].dropna()

        ratios[col] = (
            compute_gap_ratio(
                close,
                threshold
            )
        )

    return (
        pd.Series(ratios)
        .sort_values()
    )

############
"""
make_figures.py
===============
Figure generation for the commodity trend study, organized to match your
existing layout:

    [ METRICS  ]  all computation (mirrors your metrics module)
    [ PLOTTING ]  cleaning & plotting helpers
    [ MAIN     ]  main() that builds the figures

In your project these three blocks live in separate files; just move each
block into the matching module and fix the imports. Everything here only
depends on pandas / numpy / matplotlib.

Figures produced:
    fig_5y_sharpe_blocks.png    5-year-period Sharpe, grouped bars
    fig_rolling_5y_sharpe.png   rolling 5-year Sharpe, lines
    fig_yearly_gap.png          yearly (1-year) gap ratio, selected commodities
    fig_yearly_drawdown.png     yearly drawdown structure ratio
    fig_four_dim_yearly.png     2x2 panel: Sharpe / noise / drawdown / gap
"""

def compute_yearly_gap_ratio(
    close: pd.Series,
    threshold: float = 0.02,
    min_days: int = 30,
) -> pd.Series:
    """
    Yearly gap ratio for ONE commodity — the missing twin of your
    compute_yearly_noise_ratio / compute_yearly_drawdown_ratio.

    gap ratio = fraction of days in the year whose |daily return| > threshold.
    Lower = more continuous.
    """
    results = {}

    years = sorted(close.index.year.unique())

    for year in years:
        year_data = close[close.index.year == year].dropna()
        if len(year_data) < min_days:
            continue
        gaps = year_data.pct_change().abs() > threshold
        results[year] = gaps.mean()

    return pd.Series(results)


def compute_block_sharpe(
    close: pd.Series,
    blocks: list[tuple[int, int]],
    ) -> pd.Series:
    """
    Sharpe over fixed multi-year blocks for ONE commodity, e.g.
    blocks = [(2010, 2014), (2015, 2019), (2020, 2024)] for 5-year periods.

    Returns a Series indexed by a "2010-2014" style label.
    """
    results = {}

    for start, end in blocks:
        seg = close.loc[f"{start}-01-01":f"{end}-12-31"].dropna()

        if len(seg) < 60:
            continue

        ret = seg.pct_change()
        results[f"{start}-{end}"] = (
            np.sqrt(252) * ret.mean() / ret.std()
        )

    return pd.Series(results)


def compute_rolling_sharpe(
    close: pd.Series,
    window_years: int = 5,
    trading_days: int = 252,
    ) -> pd.Series:
    """
    Rolling annualized Sharpe over a `window_years`-year window for ONE
    commodity (daily resolution). Use for the smooth "5-year Sharpe" curve.
    """
    window = window_years * trading_days

    ret = close.pct_change()
    rolling = (
        np.sqrt(252)
        * ret.rolling(window).mean()
        / ret.rolling(window).std()
    )

    return rolling.dropna()


def yearly_metric_table(
    df: pd.DataFrame,
    selected: list[str],
    metric_func,
    **kwargs,
    ) -> dict[str, pd.Series]:
    """
    Apply a per-commodity yearly metric (e.g. compute_yearly_gap_ratio,
    your compute_yearly_drawdown_ratio / compute_yearly_sharpe) to each
    name in `selected`. Returns {commodity: yearly Series}.
    """
    return {
        col: metric_func(df[col].dropna(), **kwargs)
        for col in selected
        if col in df.columns
    }

# =============================================================================
# TSMOM
# =============================================================================
def tsmom_sharpe(close, lookback=120, vol_lookback=60, vol_target=0.15, max_leverage=3.0):
    close = close.dropna()
    ret = close.pct_change()                                  # daily returns
    signal = np.sign(close.pct_change(lookback))              # +1 / -1 direction
    realized_vol = ret.rolling(vol_lookback).std() * np.sqrt(252)   # annualized vol
    scale = (vol_target / realized_vol).clip(upper=max_leverage)    # position size
    position = (signal * scale).shift(1)                      # lag 1 day, no look-ahead
    strat_ret = (position * ret).dropna()                     # strategy P&L
    return np.sqrt(252) * strat_ret.mean() / strat_ret.std()  # annualized Sharpe