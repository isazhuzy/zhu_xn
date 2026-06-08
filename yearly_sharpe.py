"""
yearly_sharpe_all.py
====================
Annual (1-year) Sharpe for ALL products, collapsed into ONE figure.

Reuses your existing pieces:
    load_clean_prices()        -> cleaning_and_plotting module
    TICKER_CN                  -> your ticker->Chinese map
    compute_yearly_sharpe()    -> metrics module

A 62-line spaghetti chart is unreadable, so the single graph is a HEATMAP:
    rows    = products
    columns = years
    color   = that product's annualized Sharpe in that year
    blank   = product not trading yet / too few days that year
"""

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.colors import TwoSlopeNorm

# from cleaning_and_plotting import load_clean_prices, TICKER_CN
# from metrics import compute_yearly_sharpe


# # -----------------------------------------------------------------------------
# # 1) build the year x product Sharpe table
# # -----------------------------------------------------------------------------
# def yearly_sharpe_table(df: pd.DataFrame, min_days: int = 20) -> pd.DataFrame:
#     """
#     Annual Sharpe for every column in `df`.

#     Returns a DataFrame indexed by year, one column per product.
#     Years with < min_days observations are dropped per product (NaN).
#     """
#     cols = {}
#     for col in df.columns:
#         cols[col] = compute_yearly_sharpe(df[col].dropna(), min_days=min_days)
#     table = pd.DataFrame(cols)          # index = year, columns = product
#     return table.sort_index()


# # -----------------------------------------------------------------------------
# # 2) the single figure: heatmap of all products' yearly Sharpe
# # -----------------------------------------------------------------------------
# def plot_yearly_sharpe_heatmap(
#     table: pd.DataFrame,
#     title: str = "各品种年度趋势夏普 (Annual Sharpe by product)",
#     savepath: str | None = None,
#     sort_by_mean: bool = True,
# ) -> None:
#     """
#     One heatmap: products (rows) x years (columns), colored by annual Sharpe.
#     Diverging colormap centered at 0 so red = losing year, green = winning year.
#     """
#     mat = table.copy()

#     # order products by average Sharpe so the picture has structure
#     if sort_by_mean:
#         order = mat.mean(axis=0).sort_values(ascending=False).index
#         mat = mat[order]

#     M = mat.T                       # rows = products, cols = years
#     years = M.columns.astype(int).tolist()
#     products = M.index.tolist()

#     # diverging color scale centered on 0
#     vmax = np.nanmax(np.abs(M.values))
#     norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

#     fig, ax = plt.subplots(figsize=(max(10, len(years) * 0.6),
#                                     max(12, len(products) * 0.22)))

#     im = ax.imshow(M.values, aspect="auto", cmap="RdYlGn", norm=norm)

#     # axes
#     ax.set_xticks(np.arange(len(years)))
#     ax.set_xticklabels(years, rotation=0)
#     ax.set_yticks(np.arange(len(products)))
#     ax.set_yticklabels(products, fontsize=8)
#     ax.set_xlabel("Year")
#     ax.set_title(title, fontsize=13, pad=10)

#     # annotate each cell with the Sharpe value
#     for i in range(M.shape[0]):
#         for j in range(M.shape[1]):
#             v = M.values[i, j]
#             if np.isfinite(v):
#                 ax.text(j, i, f"{v:.1f}", ha="center", va="center",
#                         fontsize=6,
#                         color="black" if abs(v) < vmax * 0.6 else "white")

#     cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
#     cbar.set_label("Annualized Sharpe")

#     plt.tight_layout()
#     if savepath:
#         plt.savefig(savepath, dpi=150, bbox_inches="tight")
#     plt.close()


# # -----------------------------------------------------------------------------
# # main
# # -----------------------------------------------------------------------------
# def main():
#     PATH = "/mnt/user-data/uploads/Copy_of_wind品种指数数据_1_.xlsx"

#     df = load_clean_prices(PATH)
#     df = df.rename(columns=TICKER_CN)         # tickers -> Chinese names (falls back to ticker if unmapped)

#     table = yearly_sharpe_table(df, min_days=20)
#     print(f"{table.shape[1]} products, years {table.index.min()}–{table.index.max()}")

#     # save the table too, handy for later
#     table.round(3).to_csv("/mnt/user-data/outputs/yearly_sharpe_table.csv",
#                           encoding="utf-8-sig")

#     plot_yearly_sharpe_heatmap(
#         table,
#         savepath="/mnt/user-data/outputs/yearly_sharpe_heatmap.png",
#     )
#     print("done.")


# if __name__ == "__main__":
#     # CJK font for the Chinese product names
#     plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Noto Sans CJK JP"]
#     plt.rcParams["axes.unicode_minus"] = False
#     main()