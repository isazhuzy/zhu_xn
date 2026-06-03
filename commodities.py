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
# metrics
# ==========================================================

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


def main() -> None:

    filepath = (
        "Copy of wind品种指数数据(1).xlsx"
    )

    print("Loading data...")
    df = clean_wind_commodity_data(
        filepath
    )

    print(
        f"Loaded {len(df)} rows and "
        f"{len(df.columns)} commodities."
    )

    # ------------------------------------
    # Trend Sharpe
    # ------------------------------------

    trend_sharpe = (
        compute_trend_sharpe(df)
    )

    print("\nTop 10 Trend Sharpe:")
    print(
        trend_sharpe
        .sort_values(ascending=False)
        .head(10)
    )

    plot_metric(
        trend_sharpe,
        title="趋势流畅度对比",
        filename="trend_sharpe.png",
        xlabel="Trend Sharpe"
    )

if __name__ == "__main__":
    main()

