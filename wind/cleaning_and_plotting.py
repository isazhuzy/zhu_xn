"""
cleaning_and_plotting.py
========================
Data loading, CJK setup, house style, the ticker->Chinese map, and every
plotting helper used to render the report figures. Imports computation from
metrics (one-directional dependency; nothing here is imported back by metrics).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import (
    compute_block_sharpe,
    compute_rolling_sharpe,
    compute_yearly_sharpe,
    yearly_metric_table,
)

# =============================================================================
# TICKER -> CHINESE NAME   (PL=丙烯, PT=铂, PD=钯 per confirmed clarifications)
# =============================================================================
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
    "ECFI.WI": "集运指数(欧线)", "PSFI.WI": "多晶硅", "PLFI.WI": "丙烯", "PTFI.WI": "铂",
    "PDFI.WI": "钯",
}

# =============================================================================
# HOUSE STYLE
# =============================================================================
plt.rcParams.update({
    "font.size": 11,
    "font.sans-serif": ["PingFang HK", "PingFang SC", "Heiti SC",
                        "WenQuanYi Zen Hei", "Noto Sans CJK JP",
                        "Arial Unicode MS", "Arial"],
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PALETTE = [
    "#C8A02A", "#2c6fbb", "#c1452b", "#6a6a6a",
    "#2e8b57", "#8e44ad", "#d98c00", "#1f7a7a",
]


def setup_cjk_font(path: str | None = None) -> None:
    """
    Register a CJK font file so Chinese labels render, e.g. on macOS
    "/System/Library/Fonts/PingFang.ttc" or a Linux Noto CJK .ttf. Call once
    before plotting; skip if labels are English.
    """
    if path is None:
        return
    from matplotlib import font_manager as fm
    fm.fontManager.addfont(path)
    prop = fm.FontProperties(fname=path)
    plt.rcParams["font.family"] = prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False


# =============================================================================
# CLEANING / LOADING
# =============================================================================
def basic_clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Drop all-empty rows/columns and duplicate rows."""
    df = df.copy()
    df.dropna(axis=0, how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    df.drop_duplicates(inplace=True)
    return df


def clean_wind_commodity_data(filepath: str) -> pd.DataFrame:
    """Load + clean the Wind 品种指数 export (header on row 3, then a ticker row)."""
    df = pd.read_excel(filepath, sheet_name="Sheet1", header=3)
    df = basic_clean_df(df)
    df = df.iloc[1:].copy()                       # drop the Wind ticker row
    df["日期"] = pd.to_datetime(df["日期"])
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.set_index("日期", inplace=True)
    return df


def load_clean_prices(path: str, skiprows: int = 4) -> pd.DataFrame:
    """Minimal loader: 4 metadata rows, then Date + tickers."""
    raw = pd.read_excel(path, sheet_name=0, skiprows=skiprows)
    raw = raw.rename(columns={raw.columns[0]: "Date"})
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date"]).set_index("Date").sort_index()
    return raw.apply(pd.to_numeric, errors="coerce")


# =============================================================================
# 1-D RANKING BARH  (图1, 图4, 图7, 图10, 图11, 图14, 图17)
# =============================================================================
def plot_metric(metric: pd.Series, title: str, filename: str, xlabel: str) -> None:
    """Horizontal bar chart of a sorted cross-sectional metric."""
    fig, ax = plt.subplots(figsize=(12, max(8, len(metric) * 0.22)))
    metric.plot(kind="barh", ax=ax, color=PALETTE[1])
    for i, v in enumerate(metric):
        if np.isfinite(v):
            ax.text(v, i, f"{v:.2f}", va="center", fontsize=7)
    ax.set_title(title, fontsize=13, pad=8)
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.25, axis="x")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# TIME-SERIES LINES  (selected commodities)
# =============================================================================
def plot_time_series(metric: pd.Series, title: str, filename: str, ylabel: str = "value") -> None:
    plt.figure(figsize=(10, 6))
    metric.plot(marker="o")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_multiple_time_series(data_dict: dict, title: str, filename: str, ylabel: str = "value") -> None:
    plt.figure(figsize=(12, 8))
    for i, (label, series) in enumerate(data_dict.items()):
        plt.plot(series.index, series.values, marker="o", label=label, color=PALETTE[i % len(PALETTE)])
    plt.legend(frameon=False)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_yearly_metric(
    series_by_name: dict[str, pd.Series], title: str, ylabel: str,
    max_year: int | None = None, hline: float | None = None,
    savepath: str | None = None, ax: plt.Axes | None = None,
) -> plt.Axes:
    """Line chart of a yearly metric for several commodities (图6, 图8, 图12, 图18)."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(12, 4.8))
    for i, (name, s) in enumerate(series_by_name.items()):
        if max_year is not None:
            s = s[s.index <= max_year]
        ax.plot(s.index, s.values, marker="o", ms=4, lw=1.8,
                label=name, color=PALETTE[i % len(PALETTE)])
    if hline is not None:
        ax.axhline(hline, color="grey", lw=0.7)
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Year")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, frameon=False, ncol=min(3, len(series_by_name)))
    if own_fig and savepath:
        plt.tight_layout()
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
        plt.close()
    return ax


# =============================================================================
# ALL-PRODUCTS YEARLY: cross-sectional median + band  (readable 62-line cloud)
# =============================================================================
def plot_all_products_yearly(
    table: pd.DataFrame, title: str, ylabel: str,
    savepath: str | None = None, ymax: float | None = None, good_low: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in table.columns:
        ax.plot(table.index, table[col].values, color="0.7", lw=0.6, alpha=0.5)
    med = table.median(axis=1)
    q1, q3 = table.quantile(0.25, axis=1), table.quantile(0.75, axis=1)
    ax.fill_between(table.index, q1, q3, color="#2c6fbb", alpha=0.15, label="25–75% band")
    ax.plot(med.index, med.values, color="#c1452b", lw=2.4, marker="o", ms=4, label="Cross-sectional median")
    ax.set_title(title, fontsize=13, pad=8)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Year")
    if ymax is not None:
        ax.set_ylim(0, ymax)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=10)
    note = "lower = better" if good_low else "higher = better"
    ax.text(0.99, 0.97, note, transform=ax.transAxes, ha="right", va="top", fontsize=9, color="0.4")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# SMALL MULTIPLES: one mini-panel per product  (分品种小图: 图5, 图9, 图13, 图16)
# =============================================================================
def plot_yearly_small_multiples(
    table: pd.DataFrame, title: str, ylabel: str,
    savepath: str | None = None, ncols: int = 8, ymax: float | None = None,
    hline: float | None = None, max_year: int | None = None,
) -> None:
    """
    Grid of one small line chart per product (columns of `table`, the output of
    all_products_yearly_table). Keeps the 62-product yearly views readable.
    """
    cols = list(table.columns)
    n = len(cols)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.9, nrows * 1.5),
                             sharex=True, squeeze=False)
    idx = table.index
    if max_year is not None:
        idx = idx[idx <= max_year]
    for k, name in enumerate(cols):
        ax = axes[k // ncols][k % ncols]
        s = table[name]
        if max_year is not None:
            s = s[s.index <= max_year]
        ax.plot(s.index, s.values, lw=1.2, color=PALETTE[1])
        if hline is not None:
            ax.axhline(hline, color="grey", lw=0.5)
        ax.set_title(name, fontsize=7, pad=2)
        ax.tick_params(labelsize=5)
        if ymax is not None:
            ax.set_ylim(0, ymax)
    for k in range(n, nrows * ncols):              # blank unused cells
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle(title, fontsize=13)
    fig.supylabel(ylabel, fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# 5-YEAR SHARPE: blocks + rolling
# =============================================================================
def plot_block_sharpe(df: pd.DataFrame, selected: list[str], blocks: list[tuple[int, int]], savepath: str | None = None) -> None:
    table = pd.DataFrame({c: compute_block_sharpe(df[c].dropna(), blocks) for c in selected if c in df.columns})
    labels = table.index.tolist()
    x = np.arange(len(labels))
    n = len(table.columns)
    width = 0.8 / max(n, 1)
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, col in enumerate(table.columns):
        ax.bar(x + i * width - 0.4 + width / 2, table[col].values, width=width, label=col, color=PALETTE[i % len(PALETTE)])
    ax.axhline(0, color="grey", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Annualized Sharpe")
    ax.set_title(f"Sharpe by {blocks[0][1] - blocks[0][0] + 1}-year period")
    ax.legend(fontsize=9, frameon=False, ncol=min(4, n))
    ax.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close()


def plot_rolling_sharpe(df: pd.DataFrame, selected: list[str], window_years: int = 5, savepath: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for i, col in enumerate(selected):
        if col not in df.columns:
            continue
        roll = compute_rolling_sharpe(df[col].dropna(), window_years)
        ax.plot(roll.index, roll.values, lw=1.8, label=col, color=PALETTE[i % len(PALETTE)])
    ax.axhline(0, color="grey", lw=0.7)
    ax.set_ylabel("Annualized Sharpe")
    ax.set_title(f"Rolling {window_years}-year Sharpe")
    ax.legend(fontsize=9, frameon=False, ncol=min(4, len(selected)))
    ax.grid(alpha=0.25)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close()


def plot_four_dim_yearly(df: pd.DataFrame, selected: list[str], metric_funcs: dict, max_year: int | None = None, savepath: str | None = None) -> None:
    """2x2 panel comparing four yearly metrics for the same commodity set."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (title, (func, ylabel, kwargs)) in zip(axes.ravel(), metric_funcs.items()):
        series_by_name = yearly_metric_table(df, selected, func, **kwargs)
        plot_yearly_metric(series_by_name, title, ylabel, max_year=max_year, ax=ax)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# HEATMAPS  (图2 diverging Sharpe, 图3 sequential |Sharpe|)
# =============================================================================
def plot_yearly_sharpe_heatmap(
    table: pd.DataFrame, title: str = "各品种年度趋势夏普 (Annual Sharpe by product)",
    savepath: str | None = None, sort_by_mean: bool = True,
) -> None:
    """Diverging heatmap: red = losing year, green = winning year."""
    from matplotlib.colors import TwoSlopeNorm
    mat = table.copy()
    if sort_by_mean:
        order = mat.mean(axis=0).sort_values(ascending=False).index
        mat = mat[order]
    M = mat.T
    years = M.columns.astype(int).tolist()
    products = M.index.tolist()
    vmax = np.nanmax(np.abs(M.values))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    fig, ax = plt.subplots(figsize=(max(10, len(years) * 0.6), max(12, len(products) * 0.22)))
    im = ax.imshow(M.values, aspect="auto", cmap="RdYlGn", norm=norm)
    ax.set_xticks(np.arange(len(years)))
    ax.set_xticklabels(years)
    ax.set_yticks(np.arange(len(products)))
    ax.set_yticklabels(products, fontsize=8)
    ax.set_xlabel("Year")
    ax.set_title(title, fontsize=13, pad=10)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6,
                        color="black" if abs(v) < vmax * 0.6 else "white")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Annualized Sharpe")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close()


def plot_yearly_abs_sharpe_heatmap(
    table: pd.DataFrame, title: str = "各品种年度趋势强度 |夏普| (Absolute annual Sharpe)",
    savepath: str | None = None, sort_by_mean: bool = True, cmap: str = "YlGnBu",
) -> None:
    """Sequential heatmap from 0 (no trend) to high (strong trend either way)."""
    mat = table.copy()
    if sort_by_mean:
        order = mat.mean(axis=0).sort_values(ascending=False).index
        mat = mat[order]
    M = mat.T
    years = M.columns.astype(int).tolist()
    products = M.index.tolist()
    vmax = np.nanmax(M.values)
    fig, ax = plt.subplots(figsize=(max(10, len(years) * 0.6), max(12, len(products) * 0.22)))
    im = ax.imshow(M.values, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    ax.set_xticks(np.arange(len(years)))
    ax.set_xticklabels(years)
    ax.set_yticks(np.arange(len(products)))
    ax.set_yticklabels(products, fontsize=8)
    ax.set_xlabel("Year")
    ax.set_title(title, fontsize=13, pad=10)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6,
                        color="white" if v > vmax * 0.55 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("|Annualized Sharpe|")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# FIXED vs ADAPTIVE GAP SCATTER  (图15, Spearman ρ ≈ −0.07)
# =============================================================================
def plot_fixed_vs_adaptive_gap(joined: pd.DataFrame, rho: float, savepath: str | None = None) -> None:
    """Scatter of fixed-2% gap ratio vs adaptive-3σ gap ratio, one dot per product."""
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.scatter(joined["fixed"], joined["adaptive"], s=28, color=PALETTE[1], alpha=0.8)
    for name, row in joined.iterrows():
        ax.annotate(name, (row["fixed"], row["adaptive"]), fontsize=6, alpha=0.7)
    ax.set_xlabel("固定 2% 跳空比 (fixed-threshold gap ratio)")
    ax.set_ylabel("自适应 3σ 跳空比 (adaptive gap ratio)")
    ax.set_title(f"固定 vs 自适应跳空比  (Spearman ρ = {rho:.2f})", fontsize=12, pad=8)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# PREDICTABILITY SCATTERS  (图19: buy&hold vs TSMOM | IS-noise vs OOS-TSMOM)
# =============================================================================
def plot_predictive_scatter(
    bh_vs_tsmom: pd.DataFrame, is_oos: pd.DataFrame,
    rho_bh: float, rho_noise: float, savepath: str | None = None,
) -> None:
    """Two-panel scatter underpinning the predictability conclusion."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    ax.scatter(bh_vs_tsmom["buyhold"], bh_vs_tsmom["tsmom"], s=26, color=PALETTE[0], alpha=0.8)
    lo = float(np.nanmin(bh_vs_tsmom[["buyhold", "tsmom"]].values))
    hi = float(np.nanmax(bh_vs_tsmom[["buyhold", "tsmom"]].values))
    ax.plot([lo, hi], [lo, hi], color="grey", lw=0.7, ls="--")
    ax.set_xlabel("买入持有夏普 (buy-&-hold Sharpe)")
    ax.set_ylabel("TSMOM 策略夏普")
    ax.set_title(f"买入持有 vs 趋势策略  (ρ = {rho_bh:.2f})", fontsize=12)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.scatter(is_oos["is_noise"], is_oos["oos_tsmom"], s=26, color=PALETTE[2], alpha=0.8)
    ax.set_xlabel("样本内噪声比 (IS noise ratio, ≤2018)")
    ax.set_ylabel("样本外 TSMOM 夏普 (OOS, ≥2019)")
    ax.set_title(f"样本内指标无法预测样本外收益  (ρ = {rho_noise:.2f})", fontsize=12)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close()

# =============================================================================
# 跳空比阈值敏感性图 (correlation heatmap + fixed-vs-kσ scatters)
# =============================================================================
def plot_gap_threshold_corr(table, savepath=None, method="spearman"):
    """Annotated rank-correlation heatmap among fixed-% and kσ gap ratios."""
    corr = table.corr(method=method)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    n = len(corr)
    ax.set_xticks(range(n)); ax.set_xticklabels(corr.columns, fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(corr.index, fontsize=9)
    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(v) > 0.55 else "black")
    ax.set_title(f"跳空比阈值相关性 ({method} ρ)", fontsize=12, pad=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close()


def plot_gap_threshold_scatter(table, fixed_col=None, savepath=None, method="spearman"):
    """Fixed-% gap ratio vs each adaptive kσ gap ratio, one panel per k."""
    from metrics import _spearman
    fixed_col = fixed_col or table.columns[0]
    k_cols = [c for c in table.columns if c != fixed_col]
    fig, axes = plt.subplots(1, len(k_cols), figsize=(4.6 * len(k_cols), 4.4))
    if len(k_cols) == 1:
        axes = [axes]
    for ax, kc in zip(axes, k_cols):
        ax.scatter(table[fixed_col], table[kc], s=26, color=PALETTE[1], alpha=0.8)
        rho, _ = _spearman(table[fixed_col].values, table[kc].values)
        ax.set_xlabel(fixed_col); ax.set_ylabel(kc)
        ax.set_title(f"{fixed_col} vs {kc}  (ρ = {rho:.2f})", fontsize=11)
        ax.grid(alpha=0.25)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close()
