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
        / nav.cummax()
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
