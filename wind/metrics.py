"""
metrics.py
==========
All metric computation for the Wind 商品趋势研究 study. Pure pandas/numpy,
no plotting and no imports from cleaning_and_plotting (keeps the dependency
one-directional: cleaning_and_plotting -> metrics -> nothing).

Coverage by report section / figure
------------------------------------
二  趋势夏普        compute_trend_sharpe (图1)
                   yearly_sharpe_table       -> heatmap (图2)
                   yearly_abs_sharpe_table   -> |Sharpe| heatmap (图3)
三  噪声比          compute_noise_ratios (图4)
                   all_products_yearly_table / yearly_metric_table (图5, 图6)
四  回撤结构比      compute_drawdown_structure_ratios (图7)
                   yearly tables (图8, 图9)
                   compute_block_drawdown_ratios (图10, 2020-2024)
五  跳空比          compute_gap_ratios fixed 2% (图11, 图12, 图13)
                   adaptive_gap_threshold (图14)
                   compute_adaptive_gap_ratios + fixed/adaptive pairing (图15)
                   yearly adaptive table (图16)
六  TSMOM          tsmom_sharpes (图17)
                   compute_yearly_tsmom_sharpe (图18)
七  六维度总览      build_summary_table
八  可预测性        variance_ratios, predictive_validity, buyhold_vs_tsmom (图19)
"""

from __future__ import annotations
import math
import numpy as np
import pandas as pd

TRADING_DAYS = 252


# =============================================================================
# 二 · TREND SHARPE  (buy-&-hold Sharpe of the underlying)
# =============================================================================
def compute_trend_sharpe(df: pd.DataFrame) -> pd.Series:
    """
    Annualized buy-&-hold Sharpe per column.
        Sharpe = sqrt(252) * mean(daily ret) / std(daily ret)
    Returns scale ~linearly with time, vol ~with sqrt(time) -> sqrt(252) annualizes.

    NOTE (报告结论): this is the Sharpe of HOLDING the contract, not of a trend
    strategy. A market can trend hard in both directions yet have ~0 buy-&-hold
    Sharpe (e.g. 生猪). For the strategy metric see tsmom_sharpe().
    """
    ret = df.pct_change()
    sharpe = np.sqrt(TRADING_DAYS) * ret.mean() / ret.std()
    return sharpe.sort_values()


def compute_trend_sharpe_period(df: pd.DataFrame, start_year: int, end_year: int) -> pd.Series:
    period = df.loc[f"{start_year}-01-01":f"{end_year}-12-31"]
    return compute_trend_sharpe(period)


def compute_yearly_sharpe(close: pd.Series, min_days: int = 20) -> pd.Series:
    """Annualized Sharpe within each calendar year for ONE commodity."""
    out = {}
    for y in sorted(close.index.year.unique()):
        d = close[close.index.year == y].dropna()
        if len(d) < min_days:
            continue
        r = d.pct_change()
        out[y] = np.sqrt(TRADING_DAYS) * r.mean() / r.std()
    return pd.Series(out)


def yearly_sharpe_table(df: pd.DataFrame, min_days: int = 20) -> pd.DataFrame:
    """(year x product) annual Sharpe — feeds the diverging heatmap (图2)."""
    return pd.DataFrame(
        {col: compute_yearly_sharpe(df[col].dropna(), min_days=min_days) for col in df.columns}
    ).sort_index()


def yearly_abs_sharpe_table(df: pd.DataFrame, min_days: int = 20) -> pd.DataFrame:
    """|annual Sharpe| (trend STRENGTH either direction) — sequential heatmap (图3)."""
    return yearly_sharpe_table(df, min_days=min_days).abs()


# =============================================================================
# 三 · NOISE RATIO   (median 20d vol / 20d directional move; low = smoother)
# =============================================================================
def compute_noise_ratio(close: pd.Series, winsor: float = 0.99) -> float:
    """
    Close-only noise ratio for ONE commodity, winsorized so move20~0 days
    cannot explode the median.
        noise = median( std_20(ret) / |ret over 20d| )
    """
    c = close.dropna()
    move20 = c.pct_change(20).abs()
    ratio = c.pct_change().rolling(20).std() / move20
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    if ratio.empty:
        return np.nan
    return ratio.clip(upper=ratio.quantile(winsor)).median()


def compute_noise_ratios(df: pd.DataFrame, min_obs: int = 30) -> pd.Series:
    """Cross-sectional noise ratio for every commodity (sorted) — 图4."""
    out = {c: compute_noise_ratio(df[c]) for c in df.columns if df[c].notna().sum() >= min_obs}
    return pd.Series(out).dropna().sort_values()


def compute_yearly_noise_ratio(close: pd.Series, min_days: int = 30) -> pd.Series:
    out = {}
    for y in sorted(close.index.year.unique()):
        d = close[close.index.year == y].dropna()
        if len(d) < min_days:
            continue
        move20 = d.pct_change(20).abs()
        ratio = (d.pct_change().rolling(20).std() / move20).replace([np.inf, -np.inf], np.nan).dropna()
        out[y] = ratio.median() if not ratio.empty else np.nan
    return pd.Series(out)


def compute_noise_ratio_period(df: pd.DataFrame, start_year: int, end_year: int) -> pd.Series:
    period = df.loc[f"{start_year}-01-01":f"{end_year}-12-31"]
    return compute_noise_ratios(period)


# =============================================================================
# 四 · DRAWDOWN STRUCTURE RATIO   (|max DD| / |net move|; low = cleaner path)
# =============================================================================
def compute_drawdown_structure_ratio(close: pd.Series, net_floor: float = 0.05) -> float:
    """
    |max drawdown| / |net move|, denominator floored so round-trip contracts
    (net~0, e.g. 尿素/焦煤/PTA over 5y) are not assigned an exploding score.
    """
    c = close.dropna()
    if len(c) < 2:
        return np.nan
    nav = c / c.iloc[0]
    max_dd = abs((nav / nav.cummax() - 1).min())
    net_move = max(abs(c.iloc[-1] / c.iloc[0] - 1), net_floor)
    return max_dd / net_move


def compute_drawdown_structure_ratios(df: pd.DataFrame, net_floor: float = 0.05) -> pd.Series:
    """Cross-sectional drawdown structure ratio (sorted) — 图7."""
    out = {}
    for col in df.columns:
        c = df[col].dropna()
        if len(c) < 2:
            continue
        out[col] = compute_drawdown_structure_ratio(c, net_floor=net_floor)
    return pd.Series(out).dropna().sort_values()


def compute_yearly_drawdown_ratio(close: pd.Series, min_days: int = 30, net_floor: float = 0.05) -> pd.Series:
    out = {}
    for y in sorted(close.index.year.unique()):
        d = close[close.index.year == y].dropna()
        if len(d) < min_days:
            continue
        out[y] = compute_drawdown_structure_ratio(d, net_floor=net_floor)
    return pd.Series(out)


def compute_block_drawdown_ratios(
    df: pd.DataFrame, start_year: int, end_year: int, net_floor: float = 0.05, min_days: int = 60
) -> pd.Series:
    """
    Drawdown structure ratio over ONE fixed multi-year block (e.g. 2020-2024),
    cross-sectional and sorted — feeds the 5-year ranking (图10).
    """
    seg = df.loc[f"{start_year}-01-01":f"{end_year}-12-31"]
    out = {}
    for col in seg.columns:
        c = seg[col].dropna()
        if len(c) < min_days:
            continue
        out[col] = compute_drawdown_structure_ratio(c, net_floor=net_floor)
    return pd.Series(out).dropna().sort_values()


def compute_drawdown_ratio_period(df: pd.DataFrame, start_year: int, end_year: int) -> pd.Series:
    return compute_block_drawdown_ratios(df, start_year, end_year)


# =============================================================================
# 五 · GAP RATIO  —  fixed 2% threshold
# =============================================================================
def compute_gap_ratio(close: pd.Series, threshold: float = 0.02) -> float:
    """Fraction of days with |daily return| > threshold. Lower = more continuous."""
    return (close.pct_change().abs() > threshold).mean()


def compute_gap_ratios(df: pd.DataFrame, threshold: float = 0.02, min_obs: int = 30) -> pd.Series:
    """Cross-sectional fixed-threshold gap ratio (sorted) — 图11."""
    out = {}
    for col in df.columns:
        c = df[col].dropna()
        if len(c) < min_obs:
            continue
        out[col] = compute_gap_ratio(c, threshold)
    return pd.Series(out).dropna().sort_values()


def compute_yearly_gap_ratio(close: pd.Series, threshold: float = 0.02, min_days: int = 30) -> pd.Series:
    out = {}
    for y in sorted(close.index.year.unique()):
        d = close[close.index.year == y].dropna()
        if len(d) < min_days:
            continue
        out[y] = (d.pct_change().abs() > threshold).mean()
    return pd.Series(out)


# =============================================================================
# 五 · GAP RATIO  —  adaptive per-product k·sigma threshold
# =============================================================================
def adaptive_gap_threshold(close: pd.Series, k: float = 3.0) -> float:
    """
    Per-product adaptive gap threshold tau(i) = k * sigma(i), expressed as a
    DAILY-RETURN FRACTION (multiply by 100 for the % bar chart 图14).
    sigma(i) = full-sample daily-return std.
    """
    return k * close.pct_change().std()


def compute_adaptive_gap_ratio(close: pd.Series, k: float = 3.0, vol_lookback: int | None = None) -> float:
    """
    Adaptive gap ratio for ONE commodity.

    vol_lookback is None (default, matches report headline 图15):
        constant per-product sigma over the whole sample, g = 1{|r| > k*sigma}.
    vol_lookback = w:
        rolling-window sigma_t, g = 1{|r_t| > k*sigma_t} (the σ(i,t) variant).
    """
    c = close.dropna()
    ret = c.pct_change()
    if vol_lookback is None:
        sd = ret.std()
        if not sd or np.isnan(sd):
            return np.nan
        return (ret.abs() > k * sd).mean()
    sd_t = ret.rolling(vol_lookback).std()
    return (ret.abs() > k * sd_t).mean()


def compute_adaptive_gap_ratios(df: pd.DataFrame, k: float = 3.0, min_obs: int = 30, **kw) -> pd.Series:
    """Cross-sectional adaptive gap ratio (sorted)."""
    out = {c: compute_adaptive_gap_ratio(df[c], k=k, **kw) for c in df.columns if df[c].notna().sum() >= min_obs}
    return pd.Series(out).dropna().sort_values()


def compute_yearly_adaptive_gap_ratio(close: pd.Series, k: float = 3.0, min_days: int = 30) -> pd.Series:
    """
    Yearly adaptive gap ratio (图16): threshold uses the WHOLE-sample sigma(i)
    (stable per-product), only the counting is split by calendar year — matches
    GapRatio(i,y) = mean_{t in y} 1{|r_t| > k*sigma(i)}.
    """
    c = close.dropna()
    ret = c.pct_change()
    sd = ret.std()
    if not sd or np.isnan(sd):
        return pd.Series(dtype=float)
    gap = (ret.abs() > k * sd).astype(float)
    out = {}
    for y in sorted(c.index.year.unique()):
        g = gap[gap.index.year == y]
        if g.count() < min_days:
            continue
        out[y] = g.mean()
    return pd.Series(out)


def fixed_vs_adaptive_gap(df: pd.DataFrame, threshold: float = 0.02, k: float = 3.0, min_obs: int = 500):
    """
    Aligned (fixed, adaptive) gap-ratio pair per product + their Spearman rho,
    for the comparison scatter (图15, report ρ ≈ −0.07).
    """
    fixed = compute_gap_ratios(df, threshold=threshold, min_obs=min_obs)
    adapt = compute_adaptive_gap_ratios(df, k=k, min_obs=min_obs)
    joined = pd.concat([fixed.rename("fixed"), adapt.rename("adaptive")], axis=1).dropna()
    rho, p = _spearman(joined["fixed"].values, joined["adaptive"].values)
    return joined, rho, p


# =============================================================================
# 六 · TSMOM  (vol-targeted time-series momentum, no look-ahead)
# =============================================================================
def tsmom_daily(
    close: pd.Series, lookback: int = 120, vol_lookback: int = 60,
    vol_target: float = 0.15, max_leverage: float = 3.0,
) -> pd.Series:
    """Daily P&L of the vol-targeted TSMOM strategy on one contract (lagged 1d)."""
    c = close.dropna()
    ret = c.pct_change()
    signal = np.sign(c.pct_change(lookback))
    rv = ret.rolling(vol_lookback).std() * np.sqrt(TRADING_DAYS)
    scale = (vol_target / rv).clip(upper=max_leverage)
    return ((signal * scale).shift(1) * ret).dropna()


def tsmom_sharpe(close: pd.Series, lookback: int = 120, vol_lookback: int = 60,
                 vol_target: float = 0.15, max_leverage: float = 3.0) -> float:
    """Annualized Sharpe of the TSMOM strategy on one contract."""
    c = close.dropna()
    if len(c) < lookback + vol_lookback + 40:
        return np.nan
    strat = tsmom_daily(c, lookback, vol_lookback, vol_target, max_leverage)
    if len(strat) < 60 or strat.std() == 0:
        return np.nan
    return np.sqrt(TRADING_DAYS) * strat.mean() / strat.std()


def tsmom_sharpes(df: pd.DataFrame, min_obs: int = 500, **kw) -> pd.Series:
    """Cross-sectional TSMOM Sharpe (sorted) — 图17."""
    out = {c: tsmom_sharpe(df[c], **kw) for c in df.columns if df[c].notna().sum() >= min_obs}
    return pd.Series(out).dropna().sort_values()


def compute_yearly_tsmom_sharpe(close: pd.Series, min_days: int = 60, **kw) -> pd.Series:
    """
    Yearly TSMOM Sharpe for ONE commodity (图18). The daily strategy series is
    built ONCE over the full sample (so the lookback warm-up is not lost), then
    annualized within each calendar year.
    """
    strat = tsmom_daily(close, **kw)
    out = {}
    for y in sorted(strat.index.year.unique()):
        s = strat[strat.index.year == y]
        if s.count() < min_days or s.std() == 0:
            continue
        out[y] = np.sqrt(TRADING_DAYS) * s.mean() / s.std()
    return pd.Series(out)


# =============================================================================
# 八 · VARIANCE RATIO  (Lo–MacKinlay; VR>1 trending, VR<1 mean-reverting)
# =============================================================================
def variance_ratio(close: pd.Series, q: int = 10) -> float:
    logp = np.log(close.dropna())
    r1 = logp.diff().dropna()
    if len(r1) < q * 5:
        return np.nan
    var_1 = r1.var(ddof=1)
    var_q = logp.diff(q).dropna().var(ddof=1)
    return var_q / (q * var_1) if var_1 > 0 else np.nan


def variance_ratios(df: pd.DataFrame, q: int = 10, min_obs: int = 500) -> pd.Series:
    out = {c: variance_ratio(df[c], q) for c in df.columns if df[c].notna().sum() >= min_obs}
    return pd.Series(out).dropna().sort_values()


# =============================================================================
# YEARLY TABLE HELPERS  (selected set vs all products)
# =============================================================================
def yearly_metric_table(df: pd.DataFrame, selected: list[str], metric_func, **kwargs) -> dict[str, pd.Series]:
    """Apply a per-commodity yearly metric to each name in `selected` -> {name: yearly Series}."""
    return {col: metric_func(df[col].dropna(), **kwargs) for col in selected if col in df.columns}


def all_products_yearly_table(df: pd.DataFrame, metric_func, **kwargs) -> pd.DataFrame:
    """Apply a per-commodity yearly metric to EVERY column -> (year x product) DataFrame."""
    return pd.DataFrame(
        {col: metric_func(df[col].dropna(), **kwargs) for col in df.columns}
    ).sort_index()


# =============================================================================
# BLOCK / ROLLING SHARPE  (5-year comparisons)
# =============================================================================
def compute_block_sharpe(close: pd.Series, blocks: list[tuple[int, int]], min_days: int = 60) -> pd.Series:
    out = {}
    for start, end in blocks:
        seg = close.loc[f"{start}-01-01":f"{end}-12-31"].dropna()
        if len(seg) < min_days:
            continue
        ret = seg.pct_change()
        out[f"{start}-{end}"] = np.sqrt(TRADING_DAYS) * ret.mean() / ret.std()
    return pd.Series(out)


def compute_rolling_sharpe(close: pd.Series, window_years: int = 5) -> pd.Series:
    window = window_years * TRADING_DAYS
    ret = close.pct_change()
    rolling = np.sqrt(TRADING_DAYS) * ret.rolling(window).mean() / ret.rolling(window).std()
    return rolling.dropna()


# =============================================================================
# 七 · SIX-DIMENSION SUMMARY TABLE
# =============================================================================
def history_years(close: pd.Series) -> float:
    """Calendar history length in years (first to last valid observation)."""
    c = close.dropna()
    if len(c) < 2:
        return np.nan
    return (c.index[-1] - c.index[0]).days / 365.25


def build_summary_table(
    df: pd.DataFrame,
    rows: list[tuple[str, str]],
    threshold: float = 0.02,
    k: float = 3.0,
) -> pd.DataFrame:
    """
    Build the 七 overview table. `rows` = [(板块, 品种), ...] using the SAME
    (Chinese) names as df's columns. Columns mirror the report:
        历史 / 趋势夏普 / 噪声比 / 回撤结构比 / 跳空比(2%) / 自适应跳空比(3σ) / TSMOM夏普
    (variance ratio added as an extra column; drop it if the boss table omits it).
    """
    records = []
    for sector, name in rows:
        if name not in df.columns:
            continue
        c = df[name].dropna()
        ret = c.pct_change()
        records.append({
            "板块": sector,
            "品种": name,
            "历史(年)": round(history_years(c), 1),
            "趋势夏普": round(np.sqrt(TRADING_DAYS) * ret.mean() / ret.std(), 2),
            "噪声比": round(compute_noise_ratio(c), 3),
            "回撤结构比": round(compute_drawdown_structure_ratio(c), 2),
            "跳空比(2%)": round(compute_gap_ratio(c, threshold), 3),
            "自适应跳空比(3σ)": round(compute_adaptive_gap_ratio(c, k=k), 3),
            "TSMOM夏普": round(tsmom_sharpe(c), 2),
            "方差比VR": round(variance_ratio(c), 2),
        })
    return pd.DataFrame.from_records(records)


# =============================================================================
# COMPOSITE SCORE  (cross-sectional z, sign-aligned so + = trend-friendly)
# =============================================================================
def composite_score(df: pd.DataFrame, min_obs: int = 500) -> pd.DataFrame:
    cols = [c for c in df.columns if df[c].notna().sum() >= min_obs]
    m = pd.DataFrame({
        "tsmom": pd.Series({c: tsmom_sharpe(df[c]) for c in cols}),
        "vr":    pd.Series({c: variance_ratio(df[c]) for c in cols}),
        "noise": pd.Series({c: compute_noise_ratio(df[c]) for c in cols}),
        "dd":    pd.Series({c: compute_drawdown_structure_ratio(df[c]) for c in cols}),
        "gap":   pd.Series({c: compute_adaptive_gap_ratio(df[c]) for c in cols}),
    }).dropna()
    signs = {"tsmom": 1, "vr": 1, "noise": -1, "dd": -1, "gap": -1}
    z = (m - m.mean()) / m.std()
    for col, s in signs.items():
        z[col] *= s
    m["score"] = z.mean(axis=1)
    return m.sort_values("score", ascending=False)


# =============================================================================
# 八 · PREDICTIVE VALIDITY  (IS metric -> OOS TSMOM Sharpe)
# =============================================================================
def _spearman(x, y):
    """Spearman rank correlation + approximate two-sided p-value (no scipy)."""
    rx = pd.Series(np.asarray(x, float)).rank().to_numpy()
    ry = pd.Series(np.asarray(y, float)).rank().to_numpy()
    rho = np.corrcoef(rx, ry)[0, 1]
    n = len(rx)
    if n > 2 and abs(rho) < 1:
        t = rho * math.sqrt((n - 2) / (1 - rho ** 2))
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    else:
        p = float("nan")
    return rho, p


def predictive_validity(df: pd.DataFrame, split: str = "2018-12-31", min_obs: int = 400) -> pd.DataFrame:
    """Rank-correlation of each IS (<=split) metric vs OOS (>=split) TSMOM Sharpe."""
    IS, OOS = df.loc[:split], df.loc[split:]
    rows = []
    for c in df.columns:
        if IS[c].dropna().shape[0] < min_obs or OOS[c].dropna().shape[0] < min_obs:
            continue
        rows.append(dict(
            t=c,
            is_vr=variance_ratio(IS[c]),
            is_noise=compute_noise_ratio(IS[c]),
            is_dd=compute_drawdown_structure_ratio(IS[c]),
            is_tsmom=tsmom_sharpe(IS[c]),
            oos_tsmom=tsmom_sharpe(OOS[c]),
        ))
    R = pd.DataFrame(rows).dropna()
    specs = [("is_vr", 1), ("is_noise", -1), ("is_dd", -1), ("is_tsmom", 1)]
    res = []
    for col, sign in specs:
        rho, p = _spearman(sign * R[col], R["oos_tsmom"])
        res.append(dict(metric=col, rank_corr=rho, p_value=p, n=len(R)))
    return pd.DataFrame(res), R


def buyhold_vs_tsmom(df: pd.DataFrame, min_obs: int = 500) -> pd.DataFrame:
    """
    Per-product buy-&-hold Sharpe vs TSMOM Sharpe, for the left panel of 图19
    (report rank-corr ≈ 0.42 — the two are related but NOT equivalent).
    """
    rows = []
    for c in df.columns:
        s = df[c].dropna()
        if s.shape[0] < min_obs:
            continue
        ret = s.pct_change()
        rows.append(dict(t=c,
                         buyhold=np.sqrt(TRADING_DAYS) * ret.mean() / ret.std(),
                         tsmom=tsmom_sharpe(s)))
    return pd.DataFrame(rows).dropna()


# =============================================================================
# DIVERSIFIED BASKET vs SINGLE-NAME  (2019+)
# =============================================================================
def basket_vs_single(df: pd.DataFrame, start: str = "2019-01-01", min_obs: int = 500) -> dict:
    cols = [c for c in df.columns if df[c].loc[start:].dropna().shape[0] >= min_obs]
    sr = pd.DataFrame({c: tsmom_daily(df[c]) for c in cols}).loc[start:]
    basket = sr.mean(axis=1).dropna()
    sharpe = lambda x: np.sqrt(TRADING_DAYS) * x.mean() / x.std()
    singles = sr.apply(lambda x: sharpe(x.dropna()))
    return {
        "basket_sharpe": sharpe(basket),
        "avg_single_sharpe": singles.mean(),
        "basket_annvol": basket.std() * np.sqrt(TRADING_DAYS),
        "avg_single_annvol": (sr.std() * np.sqrt(TRADING_DAYS)).mean(),
        "n_contracts": len(cols),
    }


# =============================================================================
# 跳空比阈值敏感性 (fixed % vs 1σ / 2σ / 3σ adaptive)
# =============================================================================
def gap_ratio_by_threshold(df, fixed=0.02, k_list=(1.0, 2.0, 3.0), min_obs=500):
    """
    One row per product; columns = fixed-% gap ratio plus adaptive kσ gap ratios
    for each k in k_list. Feeds the threshold-sensitivity correlation study.
    """
    cols = [c for c in df.columns if df[c].notna().sum() >= min_obs]
    data = {f"固定{fixed*100:.0f}%": pd.Series(
        {c: compute_gap_ratio(df[c].dropna(), fixed) for c in cols})}
    for k in k_list:
        data[f"{k:g}σ"] = pd.Series({c: compute_adaptive_gap_ratio(df[c], k=k) for c in cols})
    return pd.DataFrame(data).dropna()


def gap_threshold_corr(table, method="spearman"):
    """Spearman (default) rank-correlation matrix among the gap-ratio definitions."""
    return table.corr(method=method)