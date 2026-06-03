import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from cleaning_and_plotting import *
from metrics import *


def main():

    df = clean_wind_commodity_data(
        "Copy of wind品种指数数据(1).xlsx"
    )
    #################################################
    trend_sharpe = compute_trend_sharpe(df)

    plot_metric(
        trend_sharpe,
        "趋势流畅度对比",
        "trend_sharpe.png",
        "Trend Sharpe"
    )

    sharpe_5y = (
    compute_trend_sharpe_period(
        df,
        2020,
        2024
        )
    )

    plot_metric(
        sharpe_5y,
        "2020-2024 Trend Sharpe",
        "trend_sharpe_5y.png",
        "Trend Sharpe"
    )

    overall_sharpe = (
    compute_trend_sharpe(df)
    )

    top2 = (
        overall_sharpe
        .sort_values(
            ascending=False
        )
        .head(2)
        .index
    )

    bottom2 = (
        overall_sharpe
        .sort_values()
        .head(2)
        .index
    )

    selected = (
    list(top2)
    + list(bottom2)
)

    yearly_sharpes = {}

    for commodity in selected:
        yearly_sharpes[commodity] = (
            compute_yearly_sharpe(
                df[commodity]
            )
        )

    plot_multiple_time_series(
    yearly_sharpes,
    "Selected Commodities Yearly Sharpe",
    "selected_commodities.png"
)
    #################################################
    noise_ratio = compute_noise_ratio(df)

    plot_metric(
        noise_ratio,
        "噪音比例对比",
        "noise_ratio.png",
        "Noise Ratio"
    )

    yearly_noise = {}

    for commodity in selected:

        yearly_noise[commodity] = (
            compute_yearly_noise_ratio(
                df[commodity]
            )
        )

    plot_multiple_time_series(
        yearly_noise,
        "Selected Commodities Yearly Noise Ratio",
        "selected_noise_ratio.png"
    )

    noise_5y = (
    compute_noise_ratio_period(
        df,
        2020,
        2024
        )
    )

    plot_metric(
        noise_5y,
        "2020-2024 Noise Ratio",
        "noise_ratio_5y.png",
        "Noise Ratio"
    )
    #################################################
    drawdown_ratio = (
    compute_drawdown_structure_ratios(
        df
        )
    )

    plot_metric(
        drawdown_ratio,
        title="回撤结构比",
        filename="drawdown_ratio.png",
        xlabel="Drawdown Ratio"
    )

    drawdown_5y = (
    compute_drawdown_ratio_period(
        df,
        2020,
        2024
        )
    )

    plot_metric(
        drawdown_5y,
        "2020-2024 Drawdown Ratio",
        "drawdown_ratio_5y.png",
        "Drawdown Ratio"
    )
    #################################################
    gap_ratio = (
        compute_gap_ratios(df)
    )

    plot_metric(
        gap_ratio,
        title="跳空次数占比",
        filename="gap_ratio.png",
        xlabel="Gap Ratio"
    )

    yearly_drawdown = {}

    for commodity in selected:

        yearly_drawdown[commodity] = (
            compute_yearly_drawdown_ratio(
                df[commodity]
            )
        )

    plot_multiple_time_series(
        yearly_drawdown,
        "Selected Commodities Yearly Drawdown Ratio",
        "selected_drawdown_ratio.png"
    )

if __name__ == "__main__":
    main()