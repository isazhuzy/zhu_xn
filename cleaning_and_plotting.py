import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# df cleaning
# ==========================================================

def basic_clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform generic dataframe cleaning.

    Removes:
    - empty rows
    - empty columns
    - duplicate rows
    """

    df = df.copy()

    df.dropna(axis=0, how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    df.drop_duplicates(inplace=True)

    return df


def clean_wind_commodity_data(filepath: str) -> pd.DataFrame:
    """
    Load and clean Wind commodity index data.

    Parameters
    ----------
    filepath : str
        Path to Excel file.

    Returns
    -------
    pd.DataFrame
        Clean dataframe indexed by date.
    """

    df = pd.read_excel(
        filepath,
        sheet_name="Sheet1",
        header=3
    )

    df = basic_clean_df(df)
    # Remove Wind ticker row
    df = df.iloc[1:].copy()
    # Convert date column
    df["日期"] = pd.to_datetime(df["日期"])
    # Convert all commodity columns to numeric
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Set date as index
    df.set_index("日期", inplace=True)
    return df


# ==========================================================
# plotting
# ==========================================================

def plot_metric(
    metric: pd.Series,
    title: str,
    filename: str,
    xlabel: str
) -> None:
    """
    Plot a horizontal bar chart and save it.
    """

    plt.rcParams["font.sans-serif"] = [
        "PingFang HK"
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    metric.plot(
        kind="barh",
        ax=ax
    )

    for i, v in enumerate(metric):
        ax.text(
            v,
            i,
            f"{v:.2f}",
            va="center"
        )

    plt.title(title)
    plt.xlabel(xlabel)

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


def plot_time_series(
    metric: pd.Series,
    title: str,
    filename: str
):
    """
    line graph
    """

    plt.figure(figsize=(10,6))

    metric.plot(
        marker="o"
    )

    plt.title(title)

    plt.ylabel(
        "Trend Sharpe"
    )

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()
def plot_multiple_time_series(
    data_dict: dict,
    title: str,
    filename: str
    ) -> None:
    """
    Plot multiple time series
    on the same figure.
    """

    plt.rcParams["font.sans-serif"] = [
        "PingFang HK"
    ]

    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(12, 8))

    for label, series in data_dict.items():

        plt.plot(
            series.index,
            series.values,
            marker="o",
            label=label
        )

    plt.legend()

    plt.title(title)

    plt.ylabel("Trend Sharpe")

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

def plot_multiple_yearly_sharpes(
    df,
    commodities
    ):

    plt.figure(figsize=(12,8))

    for commodity in commodities:

        yearly_sharpe = (
            compute_yearly_sharpe(
                df[commodity]
            )
        )

        plt.plot(
            yearly_sharpe.index,
            yearly_sharpe.values,
            marker="o",
            label=commodity
        )

    plt.legend()

    plt.title(
        "Yearly Trend Sharpe"
    )

    plt.tight_layout()

    plt.savefig(
        "selected_commodities_sharpe.png",
        dpi=300
    )

    plt.show()