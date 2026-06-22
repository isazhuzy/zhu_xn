"""
main.py
=======
Reproduces every figure and table in 《Wind 商品趋势研究 v2》, in report order.
Run:  python main.py [path_to_wind_xlsx]

Outputs (figures + CSVs) go to OUT below. Figure numbers match the report.
"""

from __future__ import annotations
import sys
import pandas as pd
import os

import metrics as mx
from cleaning_and_plotting import (
    TICKER_CN, load_clean_prices,
    plot_metric, plot_yearly_metric, plot_all_products_yearly,
    plot_yearly_small_multiples, plot_block_sharpe, plot_rolling_sharpe,
    plot_four_dim_yearly, plot_yearly_sharpe_heatmap, plot_yearly_abs_sharpe_heatmap,
    plot_fixed_vs_adaptive_gap, plot_predictive_scatter,
)
import cleaning_and_plotting as cp

OUT = "outputs"

# Focus baskets reused across the yearly views.
SELECTED = ["黄金", "棉花", "生猪", "工业硅", "玉米", "集运指数(欧线)"]

# (板块, 品种) rows for the 七 overview table — same order as the report.
SUMMARY_ROWS = [
    ("贵金属", "黄金"), ("有色金属", "铜"), ("黑色", "螺纹钢"), ("黑色", "铁矿石"),
    ("黑色", "焦煤"), ("农产品", "豆粕"), ("农产品", "棕榈油"), ("农产品", "玉米"),
    ("农产品", "生猪"), ("新能源", "碳酸锂"), ("航运", "集运指数(欧线)"), ("新能源", "工业硅"),
]


def main(path: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    df = load_clean_prices(path).rename(columns=TICKER_CN)
    print(f"Loaded {df.shape[1]} products, {df.shape[0]} days "
          f"({df.index.min().date()} -> {df.index.max().date()})")

    # ---- 二 趋势夏普 -------------------------------------------------------
    plot_metric(mx.compute_trend_sharpe(df),
                "全样本趋势夏普 (Full-sample trend Sharpe)",
                f"{OUT}/fig01_trend_sharpe.png", "Annualized Sharpe")            # 图1

    sharpe_tbl = mx.yearly_sharpe_table(df)
    plot_yearly_sharpe_heatmap(sharpe_tbl, savepath=f"{OUT}/fig02_yearly_sharpe_heatmap.png")  # 图2
    abs_tbl = mx.yearly_abs_sharpe_table(df)
    plot_yearly_abs_sharpe_heatmap(abs_tbl, savepath=f"{OUT}/fig03_yearly_abs_sharpe_heatmap.png")  # 图3

    # ---- 三 噪声比 ---------------------------------------------------------
    plot_metric(mx.compute_noise_ratios(df),
                "全产品噪声比 (Noise ratio, sorted)",
                f"{OUT}/fig04_noise_ratio.png", "Noise ratio")                   # 图4
    noise_all = mx.all_products_yearly_table(df, mx.compute_yearly_noise_ratio)
    plot_yearly_small_multiples(noise_all, "全品种年度噪声比 (分品种小图)",
                                "Noise ratio", f"{OUT}/fig05_yearly_noise_small_multiples.png",
                                ymax=1.2, max_year=2025)                          # 图5
    plot_yearly_metric(mx.yearly_metric_table(df, SELECTED, mx.compute_yearly_noise_ratio),
                       "全品种年度噪声比 (重点品种)", "Noise ratio",
                       max_year=2025, savepath=f"{OUT}/fig06_yearly_noise_selected.png")  # 图6

    # ---- 四 回撤结构比 -----------------------------------------------------
    plot_metric(mx.compute_drawdown_structure_ratios(df),
                "全产品回撤结构比 (|max DD| / |net move|, sorted)",
                f"{OUT}/fig07_drawdown_ratio.png", "Drawdown structure ratio")    # 图7
    plot_yearly_metric(mx.yearly_metric_table(df, SELECTED, mx.compute_yearly_drawdown_ratio),
                       "全品种年度回撤结构比 (重点品种)", "DD ratio",
                       max_year=2025, savepath=f"{OUT}/fig08_yearly_drawdown_selected.png")  # 图8
    dd_all = mx.all_products_yearly_table(df, mx.compute_yearly_drawdown_ratio)
    plot_yearly_small_multiples(dd_all, "全品种年度回撤结构比 (分品种小图)",
                                "DD ratio", f"{OUT}/fig09_yearly_drawdown_small_multiples.png",
                                ymax=8.0, max_year=2025)                          # 图9
    plot_metric(mx.compute_block_drawdown_ratios(df, 2020, 2024),
                "五年回撤结构比 (2020–2024, sorted)",
                f"{OUT}/fig10_block_drawdown_2020_2024.png", "Drawdown structure ratio")  # 图10

    # ---- 五 跳空比: 固定 2% --------------------------------------------------
    plot_metric(mx.compute_gap_ratios(df, threshold=0.02),
                "全商品跳空比 (固定 2%, sorted)",
                f"{OUT}/fig11_gap_fixed.png", "Gap ratio (|ret| > 2%)")           # 图11
    plot_yearly_metric(mx.yearly_metric_table(df, SELECTED, mx.compute_yearly_gap_ratio),
                       "全品种年度跳空比 (固定 2%, 重点品种)", "Gap ratio",
                       max_year=2025, hline=0.10, savepath=f"{OUT}/fig12_yearly_gap_fixed_selected.png")  # 图12
    gap_all = mx.all_products_yearly_table(df, mx.compute_yearly_gap_ratio)
    plot_yearly_small_multiples(gap_all, "全品种年度跳空比 (固定 2%, 分品种小图)",
                                "Gap ratio", f"{OUT}/fig13_yearly_gap_fixed_small_multiples.png",
                                ymax=0.6, max_year=2025)                          # 图13

    # ---- 五 跳空比: 自适应 3σ ------------------------------------------------
    adaptive_thresh = pd.Series(
        {c: mx.adaptive_gap_threshold(df[c]) * 100 for c in df.columns}
    ).dropna().sort_values()
    plot_metric(adaptive_thresh,
                "各品种自适应跳空阈值 (3σ, % 日涨跌幅)",
                f"{OUT}/fig14_adaptive_threshold.png", "Threshold (% daily move)")  # 图14
    joined, rho_gap, _ = mx.fixed_vs_adaptive_gap(df)
    plot_fixed_vs_adaptive_gap(joined, rho_gap, savepath=f"{OUT}/fig15_fixed_vs_adaptive_gap.png")  # 图15
    adapt_all = mx.all_products_yearly_table(df, mx.compute_yearly_adaptive_gap_ratio)
    plot_yearly_small_multiples(adapt_all, "全品种年度跳空比 (自适应 3σ, 分品种小图)",
                                "Adaptive gap ratio", f"{OUT}/fig16_yearly_gap_adaptive_small_multiples.png",
                                ymax=0.1, max_year=2025)                          # 图16

    # ---- 六 TSMOM ----------------------------------------------------------
    plot_metric(mx.tsmom_sharpes(df),
                "全样本 TSMOM 夏普 (sorted)",
                f"{OUT}/fig17_tsmom_sharpe.png", "TSMOM Sharpe")                  # 图17
    plot_yearly_metric(mx.yearly_metric_table(df, SELECTED, mx.compute_yearly_tsmom_sharpe),
                       "全品种年度 TSMOM 夏普 (重点品种)", "TSMOM Sharpe",
                       max_year=2025, savepath=f"{OUT}/fig18_yearly_tsmom_selected.png")  # 图18

    # ---- 七 六维度总览表 ----------------------------------------------------
    summary = mx.build_summary_table(df, SUMMARY_ROWS)
    summary.to_csv(f"{OUT}/table07_summary.csv", index=False, encoding="utf-8-sig")
    print("\n七 · 六维度总览\n", summary.to_string(index=False))

    # ---- 八 可预测性实验 ----------------------------------------------------
    pred_tbl, R = mx.predictive_validity(df)
    pred_tbl.to_csv(f"{OUT}/table08_predictive_validity.csv", index=False, encoding="utf-8-sig")
    print("\n八 · 可预测性\n", pred_tbl.round(3).to_string(index=False))

    bh = mx.buyhold_vs_tsmom(df)
    rho_bh, _ = mx._spearman(bh["buyhold"], bh["tsmom"])
    rho_noise = float(pred_tbl.loc[pred_tbl["metric"] == "is_noise", "rank_corr"].iloc[0])
    plot_predictive_scatter(bh, R, rho_bh, rho_noise, savepath=f"{OUT}/fig19_predictability.png")  # 图19

    # ---- supporting tables --------------------------------------------------
    abs_tbl.round(3).to_csv(f"{OUT}/yearly_abs_sharpe_table.csv", encoding="utf-8-sig")
    sharpe_tbl.round(3).to_csv(f"{OUT}/yearly_sharpe_table.csv", encoding="utf-8-sig")
    print("\nVariance ratio median:", round(mx.variance_ratios(df).median(), 3))
    print("Diversified basket vs single-name (2019+):")
    for k, v in mx.basket_vs_single(df).items():
        print(f"    {k:20s} {v:.3f}" if isinstance(v, float) else f"    {k:20s} {v}")
    print("\nAll figures + tables written to", OUT)

    # ---- 五 跳空比: 阈值敏感性 (1σ/2σ/3σ vs 固定 2%) ------------------------
    gap_thr = mx.gap_ratio_by_threshold(df, fixed=0.02, k_list=(1, 2, 3))

    # 表 A: 四口径 Spearman 秩相关矩阵
    corrA = mx.gap_threshold_corr(gap_thr, method="spearman")
    corrA.round(2).to_csv(f"{OUT}/tableA_gap_threshold_corr.csv", encoding="utf-8-sig")
    print("\n表 A · 四口径 Spearman 相关矩阵\n", corrA.round(2).to_string())

    # 表 B: 各口径 vs 真实波动率 (诊断它在量什么)
    vol = df.pct_change().std()                       # 每个品种的日波动率
    diagB = pd.DataFrame({
        col: [mx._spearman(gap_thr[col].values, vol.loc[gap_thr.index].values)[0]]
        for col in gap_thr.columns
    }, index=["与真实波动率 ρ"]).T.round(2)
    diagB.to_csv(f"{OUT}/tableB_gap_vs_vol.csv", encoding="utf-8-sig")
    print("\n表 B · 各口径与真实波动率的相关\n", diagB.to_string())

    # 图 15b / 15c
    cp.plot_gap_threshold_corr(gap_thr, savepath=f"{OUT}/fig15b_gap_threshold_corr.png")
    cp.plot_gap_threshold_scatter(gap_thr, savepath=f"{OUT}/fig15c_gap_threshold_scatter.png")

if __name__ == "__main__":
    PATH = sys.argv[1] if len(sys.argv) > 1 else "Copy of wind品种指数数据(1).xlsx"
    main(PATH)
