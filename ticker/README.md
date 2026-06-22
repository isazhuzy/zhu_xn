# Index-Futures Open-Minute Momentum / Reversal Study

Tick-level study of CFFEX stock-index futures (`IC`, `IF`, `IH`, `IM`), asking a
simple question: **at the 1-minute scale, does a price move predict the next
move (momentum) or fade (reversal)?** — with a deep focus on the market open
(the first two minutes, 09:30→09:31 and 09:31→09:32).

> **Headline result.** In a 38-day window (Jan–Feb 2023) the open's first minute
> looked like a *strong* reversal. Re-run on **11 years (2015–2026, ~2,660 days)**
> pulled from DolphinDB, the edge **shrinks ~5× (to ~1 bp/trade), the win rate
> drops to ~0.55, and the sign is unstable year-to-year.** The striking small-sample
> result was overfitting to a favorable regime. See [§5 Conclusions](#5-conclusions).

---

## 1. Data

| | |
|---|---|
| Source | DolphinDB `dfs://hft_future_ts`, table `TickPartitioned`, partitioned by `(month, code_init)` |
| Server | internal DolphinDB host (credentials in local `ddb_config.py`, not committed) |
| Contracts | continuous series `IC0000`, `IF0000`, `IH0000`, `IM0000` (`code_init` = `IC`/`IF`/`IH`/`IM`) |
| Underlyings | IC = CSI 500, IF = CSI 300, IH = SSE 50, IM = CSI 1000 |
| Fields | `m_nDatetime, m_nPrice, m_iVolume, m_nBidPrice, m_nBidVolume, m_nAskPrice, m_nAskVolume` |
| Tick cadence | ~2 snapshots/sec (`HH:MM:SS.000` and `.100`) |
| Tick size | **0.2 index points** for all four (CFFEX); multipliers ¥300 (IF/IH), ¥200 (IC/IM) |
| Sessions | morning **09:30–11:30**, afternoon **13:00–15:00** |
| Coverage | IC/IF/IH from **2015-04-16**, IM from **2022-07** (CSI 1000 futures launch), through 2026-06 |

Two CSV vintages are used:
- `IC_IF_IH_IM_20230104_20230304.csv` — the original 38-day raw tick dump (~1.8 GB).
- `open_breakdown/open_bars_all_2015_2026.csv` — **server-aggregated open-minute
  bars** (one row per `code, date, minute ∈ {9:30,9:31,9:32,9:33}`), the full-history
  sample used for the out-of-sample test.

---

## 2. Methodology & equations

Everything is built from **1-minute mid-price bars**. Notation: minute index `τ`,
trading day `d`, tick size `q = 0.2`.

**Mid price** (avoids using the last-trade tape; cleaner microstructure):
```
mid_t = (bid_t + ask_t) / 2
```

**1-minute bar close** (resample, last tick in the bin — this is the price you
could mark at the top of the next minute):
```
close(τ) = mid of the LAST tick in [τ, τ+1min)
```

**Forward return** — the payoff of a position opened at τ and closed at τ+1.
Shifts are done *within each (day, session) group*, so no return ever spans the
lunch break or the overnight gap:
```
fwd(τ) = close(τ+1) / close(τ) − 1            (simple)
       = ln close(τ+1) − ln close(τ)          (if log_returns=True)
```

**Signal** — two modes:
```
momentum (lookback ℓ):   signal_move(τ) = close(τ) − close(τ−ℓ)     # past move
perfect_foresight:       signal_move(τ) = close(τ+1) − close(τ)     # FUTURE move (benchmark only, not tradeable)
pos(τ) = sign(signal_move(τ)) ∈ {−1, 0, +1}
```
`momentum` = "ride the last minute's move into the next". `perfect_foresight`
is an unbeatable oracle used to measure the *ceiling*, never a strategy.

**Dead-band (threshold)** — ignore moves too small to be a real signal. Three units:
```
tick :   trade only if |signal_move(τ)| > k · q          (q = 0.2; k = 5,10,15,20,…)
sigma:   trade only if |signal_ret(τ)| > k · σ_day       (σ_day = std of that day's signal_ret)
return:  trade only if |signal_ret(τ)| > k               (k a raw return)
         where signal_ret(τ) = signal_move(τ) / close(τ)
```

**Per-minute strategy return** (the cell of the matrix):
```
r(τ) = pos(τ) · fwd(τ) · 1{ |signal_move(τ)| passes the dead-band }
     = 0   when inside the dead-band (flat / no trade)
```

**Day × minute matrix** (the object every script slices):
```
R[d, τ] = r on day d at minute τ          # pivot, rows = days, cols = minutes
```

**Contrarian (fade) variant** — because the open turned out to *reverse*, the
tradeable form is the opposite sign:
```
r_contra(τ) = − pos(τ) · fwd(τ) · 1{…} = − r(τ)
```

### Aggregation & statistics

For a return series `s` (e.g. the values of one matrix column across days,
dropping flats):
```
n          = |s|
total      = Σ s
mean       = s̄ ,   median = med(s)
hit rate   = mean(1{s > 0})                  # momentum win rate
win rate   = mean(1{s < 0})  (contrarian)    # = 1 − hit, ignoring ties
t-stat     = s̄ / ( σ_s / √n ) ,  σ_s = sample std (ddof=1)   # H0: true mean = 0
top-3 share= (Σ of the 3 most extreme days in the direction of `total`) / total
             # ≈1 means "this PnL is basically 3 days"; ≪1 means broad
```

**Bucket profile** (`analyze.bucket_profile`) groups active minutes into
`freq`-minute blocks (default 30):
```
bucket(τ) = floor( minute_of_day(τ) / freq ) · freq
```
and reports `active, hit_rate, mean_ret, total` per bucket.

**Reversal regression (OLS, threshold-free)** — the cleanest view of the open.
Let `m1, m2, m3` be the 9:30→9:31, 9:31→9:32, 9:32→9:33 returns (in bp):
```
REV-A:  m2 = α + β·m1 + ε      β < 0  ⇒ reversal
REV-B:  m3 = α + β·m2 + ε      (does the reversal persist?)
LAG-2:  m3 = α + β·m1 + ε      (does the original move still matter 2 min out?)

β  = Σ(x−x̄)(y−ȳ) / Σ(x−x̄)²            slope
SE(β) = √( Σε²/(n−2) ) / √( Σ(x−x̄)² )   standard error
t  = β / SE(β) ,   R² = corr(x,y)²        (simple regression)
```
Interpretation: **β is the fraction of the open move given back next minute.**
β = −0.5 ⇒ half is reversed; β ≈ 0 ⇒ a random walk (no edge).

**Units.** Returns are dimensionless decimals; `×1e4` → **basis points** (1 bp =
0.01%). A `k`-tick move in bp is `k·q/price·1e4` (≈ 0.3 bp/tick for IC/IM at ~6000,
≈ 0.5 bp/tick for IF, ≈ 0.7 bp/tick for IH at ~2700). Charts use the CN convention
**red = up/positive, green = down/negative**.

---

## 3. Code files

### Core engine
| File | What it does | Key formula |
|---|---|---|
| **`matrix.py`** | The whole methodology in one place. `_minute_frame()` (expensive, once per contract): ticks → mid → 1-min bars → `fwd`, `signal_move`, `pos`. `apply_threshold()` (cheap, per threshold): applies the dead-band and pivots to the day×minute matrix `R`. `split_by_month()` slices `R` by calendar month. | `r = pos·fwd·1{keep}`; pivot → `R[d,τ]` |

### Data acquisition
| File | What it does |
|---|---|
| **`fetch_all.py`** | Pulls the raw 38-day tick dump per product from DolphinDB (`code_init=`X`, date range) → `{P}_20230104_20230304.csv`. |
| **`combine_all.py`** | Concatenates the four per-product CSVs → `IC_IF_IH_IM_20230104_20230304.csv`. |

### Analysis (38-day sample)
| File | What it does | Outputs |
|---|---|---|
| **`analyze.py`** | Sweeps momentum over *every* contract × tick threshold {1,2,3,5,8,10,15,20}. `overall_stats` (profitable-minute share, mean active return, mean daily PnL) and `bucket_profile` (30-min intraday profile), each split by month. | `momentum_stats/summary.csv`, `buckets_all.csv` |
| **`open_breakdown.py`** | Per-**minute** breakdown inside two named windows (`pos_open` 09:30–09:59, `neg_pm` 13:30–13:59), swept over thresholds and split by month. | `open_breakdown/minute_{name}_*.csv` |
| **`minute_deepdive.py`** | Day-by-day decomposition of ~10 hand-picked "special minutes" (e.g. 09:31, 09:34, 13:51). Answers *"is this minute's PnL a broad pattern or a few extreme days?"* via `n, mean, t, top-3 share, Jan/Feb split`, with one figure per (product, minute). | `open_breakdown/deepdive_{daily,summary}.csv`, `figs_deepdive/*.png` |

### Plotting
| File | What it does |
|---|---|
| **`plots.py`** | Fig 1 profitability heatmap (from `summary.csv`), Fig 2 intraday PnL/hit profile, Fig 3 cross-month stability, Fig 4 2×2 grid of the four contracts, Figs 5–6 per-month / open-bucket stability. Carefully labels "0.50 = coin flip". |
| **`plot_breakdown.py`** | Plots the `open_breakdown` per-minute CSVs: top = per-trade average (bp), bottom = cumulative SUM (bp), plus a Jan-vs-Feb overlay with a sample-size panel. Footnotes spell out the units (per-trade vs summed). |

### First-two-minute deep dive (this study)
| File | What it does | Key formula |
|---|---|---|
| **`first_two_minutes.py`** | Reproduces `matrix.py` on the open, extracting the momentum PnL at columns **09:31** and **09:32** for all four contracts × thresholds {5,10,15,20}. Validates a lightweight open-only extract against the full pipeline (matches exactly). | `09:31`: signal `c31−c30`, payoff `c32/c31−1` |
| **`open_trade_stats.py`** | Reframes the open as a **contrarian** trade and reports *trade count, win rate, per-trade bp, total bp, t* vs tick threshold {5…30}. (Answers "多少笔交易 / 胜率 / 提高 ticks".) | `r_contra = −r`; win = `mean(r<0)` |
| **`reversal_regression.py`** | Threshold-free OLS reversal structure (REV-A/B, LAG-2) + open-move magnitudes. | `m2 = α+β·m1+ε` |

### DolphinDB full-history pipeline (this study)
| File | What it does |
|---|---|
| **`fetch_open_bars.py`** | Pulls **server-aggregated** open bars: last mid in each of 9:30/9:31/9:32/9:33 per `(code, date)`. Partition-pruning lessons baked in: filter `code_init` (a partition col) + a **direct range on `m_nDatetime`** (a `.date()` function does *not* prune). Tiny download (~bars, not ticks). |
| **`fetch_open_bars_all.py`** | Loops the fetch **quarter-by-quarter** (≤12 partitions/query) to stay under the server's open-files limit, concatenates → `open_bars_all_2015_2026.csv`. |
| **`analyze_open_bars.py`** | Runs the contrarian table + reversal regression on any bars CSV; `--cap BP` drops glitch days, `--by-year` prints yearly stability, `--keep-march` keeps March (a normal month at full history). |
| **`diag_open_bars.py`** | Data-quality forensics: per-product move distributions, glitch counts per year, IH by-year regression, and the worst-offending days. Found the single bad tick. |

### Separate sub-project
- **`wind/`** — an unrelated commodity-trend study on Wind index data
  (`metrics.py` computes trend-Sharpe, noise ratio, drawdown-structure ratio, gap
  ratio; see `wind/conclusions.txt`). Not part of the futures study.

---

## 4. Reproduce

```bash
# env: project .venv is Python 3.12 with dolphindb 3.0.4.2, pandas, numpy, matplotlib
PY=/Users/zhuisabella/xn/.venv/bin/python

# 38-day sample (needs the raw CSV)
$PY first_two_minutes.py          # momentum PnL @ 09:31 / 09:32
$PY open_trade_stats.py           # contrarian: trades / win rate / tick sweep
$PY reversal_regression.py        # OLS reversal structure

# full history from DolphinDB (network; run outside the sandbox)
$PY fetch_open_bars_all.py        # -> /tmp/open_bars_all.csv  (~1 min, quarter-chunked)
$PY analyze_open_bars.py open_breakdown/open_bars_all_2015_2026.csv --keep-march --cap 150 --by-year
$PY diag_open_bars.py             # data-quality check
```

---

## 5. Conclusions

### A. Pipeline is validated
The DolphinDB server-side bars reproduce the original `matrix.py` numbers **exactly**
(e.g. IC `09:31` @5t: n=31, total −96.6 bp, win 0.355, t=−1.94). Whatever differs
between the 38-day and full-history results is *signal*, not a code change.

### B. What the 38-day window (Jan–Feb 2023) said
1. **The open's first minute (09:31) is a strong reversal in all four contracts.**
   Trading the open move as momentum *loses*: IC −97, IF −169, IH −222, IM −137 bp
   total; win 16–36%; t −1.9 to −3.7.
2. As a **contrarian** trade (fade the open): win **65–77%** at 5 ticks, +3 to +7 bp/trade.
3. **Raising the tick threshold helped** — bigger open moves reversed harder:
   IF/IH win climbed to **90–100%** at 20–30 ticks, edge to +10–24 bp.
4. **The second minute (09:32) had no edge** (win ≈ 50%, |t| < 1.5).
5. Regression: `m2~m1` β = −0.44…−0.61, **R² up to 0.57** (IF/IH ≈ −0.75 corr);
   `m3~m2 ≈ 0`, `m3~m1 ≈ 0` → a **one-minute snap-back**, strongest in large-caps.

### C. What 11 years (2015–2026) said — the corrections
1. **The edge largely evaporates.** Per-trade +3–7 bp → **~+1 bp**; win 0.65–0.77 →
   **0.51–0.56**; reversal **R² 0.56 → 0.01** (β −0.05…−0.10). The open move gives
   back only **~5–10% of itself**, not ~50%.
2. **The "more ticks → higher win rate" effect is gone.** Full-sample win rate is
   **flat ~0.55 across all thresholds**, and t-*falls* as you raise ticks (you just
   shrink n). That climb to 100% was tiny-sample selection.
3. **The sign is regime-dependent.** Positive 2016–18 and 2022–23; ~zero/negative
   in **2015, 2024, 2025** (e.g. IC 2025: win 0.45, t = −2.1). **2023 was a peak
   year** — the original window unknowingly sat on a local maximum.
4. **Statistically real, economically marginal.** IF/IH `09:31` still clears t ≈ 3–4
   over 2,000+ days at 5 ticks — but ~1 bp gross ≈ the bid-ask round-trip (1 tick ≈
   0.3–0.7 bp here). **Net ≈ 0.** Not a standalone strategy.
5. **IM** (smallest, newest) shows essentially **no** open reversal over its full sample.

### D. Data quality
- A **single bad tick** — IH **2024-10-08**, 9:31 mid = 1569.4 ≈ half its neighbors
  (a missing quote side) — produced a spurious **β = −1.94, R² = 0.98** over 2,660
  days. Removing 5 glitch rows (`|1-min move| > 150 bp`) fixed it. The large **2015**
  moves are the *real* China-crash open and were kept.
- The `…0000` series is a **stitched continuous contract**; roll dates can inject
  artificial jumps. The open window mostly avoids them, but always check.

### E. Mechanics
- "First two minutes" = matrix columns **09:31** and **09:32**. The 09:30 bar itself
  has **no signal** (no prior bar inside the session).

**Bottom line:** increasing the sample *overturned the conclusion*. The open reversal
is genuine but tiny and unstable — a textbook small-sample / favorable-regime artifact.

---

## 6. Quant 101 — lessons this study teaches

1. **t-stats scale like √n; a big mean on tiny n is not evidence.**
   `t = mean / (std/√n)`. The 38-day open looked huge per-trade (+7 bp) but rested on
   ~30 observations; the *same* effect on 2,000 days is ~1 bp. Demand significance at
   large n before believing a number.

2. **In-sample vs out-of-sample / data-snooping.** We "discovered" an edge in one
   2-month window and it shrank 5× out of sample because that window was a favorable
   regime. We also scanned many cells (≈10 minutes × 4 products × 6 thresholds); with
   enough cells some look significant by chance (multiple testing → Bonferroni /
   deflated-Sharpe thinking). Always validate on independent data.

3. **Conditioning is selection.** Raising the tick threshold *both* shrinks the sample
   *and* cherry-picks extremes. "100% win" on 5–6 trades is noise, not skill. Watch n
   as you slice; a rising win rate with a collapsing sample is a red flag.

4. **Statistical ≠ economic significance.** Full-sample IF/IH cleared t ≈ 3–4 yet the
   ~1 bp edge sits below the spread. A strategy must beat **costs** (spread + fees +
   slippage + impact), not just zero. Net-of-cost is the only number that trades.

5. **Microstructure can manufacture fake reversal.** Mid-price/last-trade series show
   negative autocorrelation purely from **bid-ask bounce** — quotes ping between bid
   and ask with no information. Our single largest "reversal" was literally a quote
   glitch. Distinguish real mean-reversion from microstructure noise.

6. **One outlier can own an OLS fit.** A single bad print drove **R² = 0.98** over
   2,660 days (leverage point). Before trusting a regression: plot the data, check
   min/max, winsorize/cap, inspect the extreme rows. R² and β are *not* robust.

7. **Avoid look-ahead; align bars carefully.** The engine shifts within `(day,
   session)` so the signal at τ uses only data ≤ τ and is paid τ→τ+1 — never peeking,
   never spanning lunch/overnight. The `perfect_foresight` mode deliberately *does*
   peek, as a ceiling benchmark you can't trade. Leakage is the cardinal backtest sin.

8. **Continuous contracts roll.** `…0000` stitches successive contracts; rolls create
   artificial price jumps. Use back-adjustment and sanity-check around roll dates.

9. **Non-stationarity / regime dependence.** The open reversal was positive some years,
   negative others. Markets aren't stationary; test an edge's **stability over time**,
   not just its pooled average. A pooled t-stat hides sign flips.

10. **Hit rate ≠ PnL; check concentration.** Win rate ignores payoff size; a 70%-win
    rule can still lose on a few fat tails. Pair it with mean/total PnL and a
    concentration check (the **top-3-day share**: is the PnL broad, or three lucky days?).

11. **Units discipline.** Decimals vs bp (×1e4), ticks→bp depends on price level, and
    "per-trade average" vs "summed across days" are different axes — `plot_breakdown.py`'s
    footnotes exist precisely because mixing them misleads.

12. **Build cheap/expensive correctly.** `matrix.py` computes the costly bar frame
    once, then applies many cheap thresholds — and the DolphinDB fetch aggregates
    *server-side* so you download ~35k bars instead of ~50M ticks. Know where the cost
    is and push work to the data.

---

## 7. 研究进程与结论提纲 / Study progression (4 stages)

**数据:** 2 个月(2023-01 ~ 2023-02,38 个交易日),4 个股指期货(IF 沪深300 / IM
中证1000 / IC 中证500 / IH 上证50)。*Data: 2 months (Jan–Feb 2023, 38 trading days),
4 CFFEX index futures.* 下面四步是研究的推进逻辑——每一步的发现引出下一步。
原始提纲措辞保留,仅补充【结论】与【配图】。Figures live in `figs_conclusion/`.

### 1. 根据前一秒的信号判断下一秒做多/做空
1. 以 ticks 阈值为控制变量,研究 ticks 的提高和回报/胜率的关系
2. 研究是否有时间段有明显趋势

**结论:** 整体上信号 ≈ 抛硬币——各 ticks 阈值下胜率都在 ~49–50%,提高阈值不带来系统性
优势(全天合计无边际)。但分时段看日内并不均匀:**早盘偏弱动量、午后(13:30–14:30)偏反转**,
存在两个明显时段 → 引出第 2 步。
**配图:** [fig8 全日动量地图](figs_conclusion/fig8_全日动量地图.png)、[fig9 日内累计曲线](figs_conclusion/fig9_日内累计曲线.png)　**脚本:** `analyze_day.py`, `fetch_day_bars.py`, `analyze.py`

### 2. 发现 9:30-9:59,13:30-13:59 两个时间段有明显的上升/下滑趋势
1. 以分钟为单位,以 ticks 阈值为控制变量,研究 ticks 的提高和回报/胜率的关系
2. 以分钟为单位,研究特定分钟是否有强趋势

**结论:** 下钻到分钟级后:窗口收益**高度集中在少数几个分钟**,而非整段 30 分钟均匀分布;
开盘端最极端的是 **09:31** 这一分钟,午后端 13:30–13:59 也有自己的强势分钟。分钟级上提高
ticks 阈值会**放大单笔幅度**(只留大波动),但样本数同时下降。→ 聚焦 09:31/09:32(第 3 步)。
**配图:** [fig8 全日动量地图](figs_conclusion/fig8_全日动量地图.png)、逐日分解 [deepdive_IC0000_0931.png](open_breakdown/figs_deepdive/deepdive_IC0000_0931.png)　**脚本:** `open_breakdown.py`, `minute_deepdive.py`

### 3. 发现 9:31,9:32 有非常较强烈的价格趋势
1. 以 ticks 阈值为控制变量,研究 ticks 的提高和回报/胜率的关系
2. 把数据从原始的两个月拉长到 11 年(2015-2026),研究这个趋势是否仍然存在

**结论:** 2 个月样本上 09:31 呈现**很强的反转**(逆势胜率 65–77%、每笔 +3~+7bp,且阈值越高
越强、胜率可达 90–100%);09:32 已明显减弱。但样本拉长到 **11 年**后这个 edge **基本消失**:
每笔缩到 ~1bp、胜率 ~55%、反转回归 **R² 从 0.56 塌到 0.01**,且**逐年正负交替**(2016–18、
2022–23 为正,2015、2024–25 为负),2023 恰好是最好的年份之一。→ 原始结论是**小样本/有利
行情的过拟合**;~1bp ≈ 买卖价差,扣成本后 ≈ 0,不可独立交易。
**配图:** [fig1 样本量对比](figs_conclusion/fig1_样本量对比.png)、[fig2 反转回归散点](figs_conclusion/fig2_反转回归散点.png)、[fig3 逐年稳定性](figs_conclusion/fig3_逐年稳定性.png);阈值用 [fig4](figs_conclusion/fig4_阈值胜率假象.png)/[fig6](figs_conclusion/fig6_两分钟阈值胜率.png);累计用 [fig5](figs_conclusion/fig5_累计收益曲线.png)　**脚本:** `first_two_minutes.py`, `open_trade_stats.py`, `reversal_regression.py`, `fetch_open_bars*.py`, `analyze_open_bars.py`, `make_figures.py`

### 4. 把每分钟的起始点从原始的第 0 个 tick 变更为第 n 个,n = 8,10,12,14 …,研究 n 和回报的关系
1. 做逆势
2. 做顺势

**结论:** 最终口径——**信号 = 上一分钟涨跌** `C(M-1)−C(M-2)`,**反向开仓(做反转)**;进场 = 该分钟
**第 N 个 tick**,出场 = **下一分钟第 N 个 tick**(即**持有整一分钟、窗口随 N 后移**,所以 N 只改变"进场早晚",
不改变持有长度);**各合约单独看(不用等权组合)**。

- **2 个月窗口:** 固定整分钟持有后,收益**对 N 基本平**(第 1 与第 30 个 tick 进场几乎无差) → "进场早晚"
  不是有效杠杆。只有 **IH 上证50 有微弱反转**(~+0.12bp,**t≈2.0–2.6**);**IC、IM ≈ 0**,**IF 微正不显著**。
  做顺势 = 镜像、符号相反。
- **选择性跨年复现(2019/2022/2023/2024/2025 各 1–2 月,fig13):** **IH 反转不稳定、符号随年份翻转**——
  2023/2025 显著为正(t≈2.2–2.4),**2019 显著为负**(t≈−2~−3.6),2022≈0。而且**四个合约同步**:
  **2019/2022 偏延续(做反转亏),2023–25 偏反转** → 像全市场 **regime(体制)切换**,不是某合约的结构性边际。
  (2016 受限期数据异常:IC +13 / IF −14 bp,已剔除。)

→ 与第 3 步同一教训:**2 个月看到的"IH 反转"只是 2023–25 这段体制的特征,2019 是反的**;且幅度全程 **<买卖价差**。
故**不是可外推、可交易的信号**,而是随体制漂移的微观结构现象。"选择性抽样"用 5 个分散窗口即证伪了"IH 稳定反转"。
（注:更早两版口径已作废——(i) 误把信号取成当前分钟前 N 个 tick→会得到"亚分钟动量";(ii) 出场固定在该分钟收盘
→"进场越晚收益越衰减"其实是"持有变短"的假象,改成整分钟持有后消失。）
**配图:** [fig11 进场延迟扫描(整分钟持有)](figs_conclusion/fig11_全日逐分钟tick扫描.png)、[fig13 跨年复现](figs_conclusion/fig13_跨年复现.png)、[fig12 时段分布](figs_conclusion/fig12_动量日内分布.png)　**脚本:** `intraminute_wholeday_ddb.py`(扫描)、`replication_sample.py`(跨年)、`intraminute_byhour_ddb.py`(时段)、`intraminute_*_plot.py`、`replication_plot.py`;开盘早期探索 `subminute_demo.py`/`subminute_compare.py`(fig7/fig10)

### 一句话总览 / One-line summary
全天整体 ≈ 抛硬币;开盘 09:31 的强反转是 2 个月小样本的**过拟合**,11 年上消失;**隔分钟反转**只在
**IH 上证50** 上微弱出现(对进场 tick 不敏感),但**跨年符号翻转**(2019 为负、2023–25 为正)、是**体制依赖**;
**所有效应都 ≈ 价差量级**,属 microstructure 事实而非可交易策略。**没有发现可外推、可交易的边际。**

### 图表索引 / Figure index (`figs_conclusion/`)
| 图 | 内容 | 对应步骤 |
|---|---|---|
| fig1 样本量对比 | 09:31 每笔收益+胜率,38天 vs 全历史 | 3 |
| fig2 反转回归散点 | m2~m1 回归,R² 0.56→0.01 | 3 |
| fig3 逐年稳定性 | 逐年逆势收益热力图 | 3 |
| fig4 阈值胜率假象 | 胜率 vs 阈值,两样本 | 3 |
| fig5 累计收益曲线 | 09:31 逆势累计毛收益 | 3 |
| fig6 两分钟阈值胜率 | 09:31/09:32 × 阈值胜率 | 3 |
| fig7 进场tick演示(开盘) | 开盘宽 N 扫描 | 4 |
| fig8 全日动量地图 | 合约×30分钟时段热力图 | 1/2 |
| fig9 日内累计曲线 | 平均日内累计动量收益 | 1/2 |
| fig10 进场tick对比(开盘) | 6/8/10/12 对比 | 4 |
| fig11 全日逐分钟tick扫描 | 上一分钟反转、第N tick进场、整分钟持有(进场延迟扫描,各合约) | 4 |
| fig12 动量日内分布 | 反转收益:合约 × 时段 热力图(整分钟持有,N=10) | 4 |
| fig13 跨年复现 | IH 等反转跨年(2019–2025)复现:符号翻转、不稳定 | 4 |

> 口径提醒:所有收益为**毛收益(mid-to-mid,未扣价差/手续费)**,bp = 0.01%。第 1–3 步为
> 1 分钟 bar、动量/反转(`r=sign(close−prev)·fwd`);第 4 步为 tick 级:**信号取上一分钟、在该分钟
> 第 N 个 tick 进场做反转、持有整一分钟(出场=下一分钟第 N tick)**,N 只改变进场早晚、不改变持有长度。
> "tick" 有两义:① 时间(快照,≤120/分钟,实测均~90);② 价格(最小变动 = 0.2 指数点)。第4步的 N 是①,
> 第1–3步的阈值"k 跳"是②。
