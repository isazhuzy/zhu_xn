"""
generate the figures
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from cleaning_and_plotting import *
from metrics import *

TICKER_CN = {
    "AUFI.WI": "黄金", "AGFI.WI": "白银", "SNFI.WI": "锡", "CUFI.WI": "铜",
    "ALFI.WI": "铝", "ZNFI.WI": "锌", "NIFI.WI": "镍", "PBFI.WI": "铅",
    "SSFI.WI": "不锈钢", "RBFI.WI": "螺纹钢", "HCFI.WI": "热卷", "IFI.WI": "铁矿石",
    "JFI.WI": "焦炭", "JMFI.WI": "焦煤", "SFFI.WI": "硅铁", "SMFI.WI": "锰硅",
    "CFI.WI": "玉米", "CSFI.WI": "玉米淀粉", "AFI.WI": "豆一", "MFI.WI": "豆粕",
    "YFI.WI": "豆油", "OIFI.WI": "菜籽油", "RMFI.WI": "菜籽粕", "PFI.WI": "棕榈油",
    "CFFI.WI": "棉花", "CJFI.WI": "红枣", "SRFI.WI": "白糖", "APLFI.WI": "苹果",
    "JDFI.WI": "鸡蛋", "LHFI.WI": "生猪", "PKFI.WI": "花生", "SPFI.WI": "纸浆",
    "BUFI.WI": "沥青", "FUFI.WI": "燃料油", "LUFI.WI": "低硫燃料油", "SCFI.WI": "原油",
    "PGFI.WI": "液化石油气", "TAFI.WI": "PTA", "PXFI.WI": "对二甲苯", "PRFI.WI": "瓶片",
    "EGFI.WI": "乙二醇", "EBFI.WI": "苯乙烯", "MAFI.WI": "甲醇", "PPFI.WI": "聚丙烯",
    "LFI.WI": "聚乙烯", "VFI.WI": "PVC", "URFI.WI": "尿素", "SAFI.WI": "纯碱",
    "FGFI.WI": "玻璃", "RUFI.WI": "天然橡胶", "NRFI.WI": "20号胶", "LGFI.WI": "原木",
    "LCFI.WI": "碳酸锂", "SIFI.WI": "工业硅", "SHFI.WI": "烧碱", "AOFI.WI": "氧化铝",
    "ECFI.WI": "集运指数(欧线)", "PSFI.WI": "多晶硅", "PLFI.WI": "烯", "PTFI.WI": "铂", 
    "PDFI.WI": "钯",
}

def compute_yearly_sharpe(close: pd.Series, min_days: int = 20) -> pd.Series:
    out = {}
    for y in sorted(close.index.year.unique()):
        d = close[close.index.year == y].dropna()
        if len(d) < min_days:
            continue
        r = d.pct_change()
        out[y] = np.sqrt(252) * r.mean() / r.std()
    return pd.Series(out)


def compute_yearly_drawdown_ratio(close: pd.Series, min_days: int = 30,
                                  net_floor: float = 0.05) -> pd.Series:
    out = {}
    for y in sorted(close.index.year.unique()):
        d = close[close.index.year == y].dropna()
        if len(d) < min_days:
            continue
        nav = d / d.iloc[0]
        dd = abs((nav / nav.cummax() - 1).min())
        net = max(abs(d.iloc[-1] / d.iloc[0] - 1), net_floor)
        out[y] = dd / net
    return pd.Series(out)


def main():
    PATH = "Copy of wind品种指数数据(1).xlsx"
    df = load_clean_prices(PATH)
    df = df.rename(columns=TICKER_CN)          # <-- tickers become Chinese names

    # selected now uses Chinese names (must match the renamed columns)
    selected = ["黄金", "棉花", "生猪", "工业硅", "玉米", "集运指数(欧线)"]

    # 1) 5-year Sharpe — block comparison + rolling
    plot_block_sharpe(
        df, selected,
        blocks=[(2010, 2014), (2015, 2019), (2020, 2024)],
        savepath="fig_5y_sharpe_blocks.png",
    )
    plot_rolling_sharpe(df, selected, window_years=5,
                        savepath="fig_rolling_5y_sharpe.png")

    # 2) yearly (1-year) gap ratio of selected commodities
    gap_by_name = yearly_metric_table(df, selected, compute_yearly_gap_ratio)
    plot_yearly_metric(
        gap_by_name,
        title="Yearly gap ratio (|return| > 2%)",
        ylabel="Gap ratio",
        max_year=2025, hline=0.10,
        savepath="fig_yearly_gap.png",
    )

    # 3) yearly drawdown structure ratio
    dd_by_name = yearly_metric_table(df, selected, compute_yearly_drawdown_ratio)
    plot_yearly_metric(
        dd_by_name,
        title="Yearly drawdown structure ratio (low = clean)",
        ylabel="|max DD| / |net move|",
        max_year=2025,
        savepath="fig_yearly_drawdown.png",
    )

    # 4) four-dimension 2x2 panel
    plot_four_dim_yearly(
        df, selected,
        metric_funcs={
            "Yearly Sharpe":          (compute_yearly_sharpe,          "Sharpe",          {}),
            "Yearly noise (low=smooth)": (compute_yearly_gap_ratio,    "Gap ratio",       {}),
            "Yearly drawdown (low=clean)": (compute_yearly_drawdown_ratio, "DD ratio",   {}),
            "Yearly gap (>2%)":       (compute_yearly_gap_ratio,       "Gap ratio",       {}),
        },
        max_year=2025,
        savepath="fig_four_dim_yearly.png",
    )

    print("figures written.")

if __name__ == "__main__":
    main()