# Study Guide — VOI / OIR / MPB: a Real Multi-Factor Order-Book Forecast
### Based on D. Shen (2015), *Order Imbalance Based Strategy in High Frequency Trading*, MSc thesis, Mathematical Institute, University of Oxford.

Paper #3 in our series, and the closest to home possible: Shen built it on **CSI 300
index futures (IF) on CFFEX, 500 ms snapshots, 244 trading days of 2014** — literally
the same market and data cadence as our `hft_future_ts`, six years before our sample
starts. Code: `voi_ddb.py` / `voi_plot.py`.

---

## 0. The one-sentence idea
> **Three cheap numbers from the last few snapshots — how the best queues *changed* (VOI), how lopsided they *are* (OIR), and where trades printed relative to the mid (MPB) — linearly forecast the average mid-price over the next few seconds, out of sample.**

Paper #1 (OFI) showed flow *explains* the same interval's move. This paper crosses the line into genuine *forecasting* — with the honest drop in R² that comes with it.

---

## 1. The three factors (all computable from one L1 feed)

### 1.1 VOI — Volume Order Imbalance (盘口增量失衡)
Between consecutive snapshots, per side:
```
            ⎧ 0            if Pb went DOWN     (queue died / retreated — ignore)
ΔV_bid  =   ⎨ qb − qb_prev if Pb unchanged     (net adds minus cancels/fills)
            ⎩ qb           if Pb went UP       (fresh queue at a better price)
ΔV_ask  =   mirror image (down ↔ up)
VOI     =   ΔV_bid − ΔV_ask
```
VOI is the *snapshot-differenced twin of OFI* from paper #1 — net buying "effort" at the top of the book. Positive VOI = bid side strengthening.

### 1.2 OIR — Order Imbalance Ratio (盘口静态失衡)
```
OIR = (qb − qa)/(qb + qa)
```
Exactly the queue imbalance of paper #2 (Gould–Bonart). VOI is the *derivative*; OIR is the *level*.

### 1.3 MPB — Mid-Price Basis (成交价相对中间价的偏离)
```
TP_t  = ΔTurnover / (ΔVolume × multiplier)        ← average trade price this interval
        (carried forward from the last trading interval when no trades)
MPB_t = TP_t − (M_t + M_{t−1})/2
```
Where inside the spread did trades actually print? Buyers lifting the ask push TP toward the ask → MPB > 0 = aggressive buying. MPB is the *trade-side* information (like TI in paper #1), packaged as a price, not a volume. Our data: `m_iTurnover`, `m_iVolume`, multiplier 300 (IF/IH) or 200 (IC/IM).

## 2. The target: average future mid change
```
y_k(t) = mean( M_{t+1}, …, M_{t+k} ) − M_t        k ∈ {1, 4, 20, 120} snapshots
                                                    = 0.5s, 2s, 10s, 60s
```
Averaging the next k mids (instead of taking the point value M_{t+k}) filters bid-ask bounce out of the target — you predict the *level the price hangs around*, which is also what a passive fill would earn you. This is Shen's choice and standard in HFT research.

## 3. The two models
```
Model A:  y_k ~ β0 + Σ_{l=0..5} β_l · VOI_{t−l}
Model B:  y_k ~ β0 + Σ β_l·VOI_{t−l}/s_t + Σ γ_l·OIR_{t−l}/s_t + δ·MPB_t/s_t
```
- **Lags 0..5** (≈ the last 3 seconds): order flow is autocorrelated; yesterday's-snapshot flow still carries information.
- **Divide by the spread `s_t`** (in ticks, Shen's key refinement): the same imbalance means *less* when the spread is wide — a wide spread is a shock absorber. Normalizing makes one coefficient fit all spread regimes. (Same physics as paper #2's "signal is stronger when spread = 1 tick".)
- Everything is plain **OLS** — 13 coefficients, no machine learning, fully interpretable.

## 4. The discipline: train / test split (this is the part to copy)
- **Train 2020-01 … 2024-12** → estimate β once, freeze it.
- **Test 2025-01 … 2026-05** → apply frozen β; report **out-of-sample R²**:
  `R²_oos = 1 − Σ(y−Xβ_train)² / Σ(y−ȳ_test)²`.
- Implementation trick: we accumulate the moment blocks (XᵀX, Xᵀy, yᵀy, n) per month, so both the pooled train fit and the exact OOS R² come from sufficient statistics — no need to hold 100M rows in memory. (`voi_ddb.py`)

Remember why we're paranoid: our own `ticker/` open-reversal had in-sample R² 0.56 on 38 days and **evaporated** out of sample. A signal is only real if it survives data it has never seen.

## 5. Our results (full sample; train 2020–24, test 2025–26)

> Definitive numbers in `voi_results.csv`, `voi_hitrate.csv`, `voi_permonth.csv`; figures 81–83.

Headlines (train n = 13–29M rows per contract, test n = 7–9M):
- **It predicts, out of sample.** Model B at k=1 (next 0.5s), frozen 2020–24 coefficients on 2025–26 data: OOS R² = **8.7% (IF), 8.2% (IH), 7.0% (IM), 5.2% (IC)**. Compare: our lagged-OFI predictive R² in paper #1 was ≈ 0.1–1%. Proper factors + lags + spread-normalization buy an order of magnitude or two.
- **R² decays fast with horizon** (fig81): ~5–9% (0.5s) → 3.5–6% (2s) → 1–1.7% (10s) → 0.15–0.3% (60s). The book knows the next seconds, not the next minute.
- **Model B ≫ Model A** at every horizon: adding OIR + MPB and spread-normalizing multiplies R² by ~3–5× over raw VOI lags at k=1 (e.g. IF OOS 2.8% → 8.7%). Diverse information (flow + level + trades) beats more of the same.
- **Sign hit rates rise with signal strength** (fig83, OOS k=20): unconditional ≈ 54.5–55.6%, conditioning on |ŷ| > 0.5 tick gives **62.5% (IH), 60.5% (IF), ~58% (IC/IM)** on the 14–33% of snapshots that trigger. A usable confidence dial.
- **Stable through time — and not decaying out-of-sample** (fig82): monthly R² wiggles with the volatility regime but never dies. Strikingly, OOS R² in 2025–26 is *higher* than the pooled train R² (IF 8.7% vs 7.3%) — the frozen coefficients aged fine; 2025–26 was simply a more order-flow-driven market than the 2020–24 average. Refit-on-test only adds ~1pp over frozen β.
- Cross-contract: IF/IH (tight spread) strongest, IC weakest — same reason as paper #2: tighter spread ⇒ cleaner signal. (IM's high OOS number is partly a period effect: its test window is its most liquid era.)
- **Against Shen's own 2014 numbers**: he reports average R² ≈ 0.03 for the VOI-only model and ≈ 0.07 for the extended model at k=20 (10s) on IF. Our IF at k=20 gets 0.010 (A) / 0.016 (B) in-sample — the *structure* replicates (B ≈ 2–5× A, same shapes, same horizon decay), at roughly **a third to half the strength**. Plausible reasons: 2014 was the pre-restriction, hyper-liquid era (the 2015.09 CFFEX crackdown gutted volumes and changed the microstructure), and more competition has since arbitraged the easy part away. A perfect example of "the phenomenon persists, the magnitude decays".

## 6. From forecast to P&L — the backtest that quantifies "inside the spread" ⚠️

We ran the full Shen-style threshold strategy out of sample (`voi_backtest.py`, figures 84–85): frozen 2020–24 model-B k=20 coefficients, 2025–26 test period, trade 1 lot when |ŷ| > q, hold 10 s, non-overlapping. Five execution/cost tiers per trade:

| per-trade avg (ticks), q=2.0 | IC | IF | IH | IM |
|---|---|---|---|---|
| **mid→mid (signal value)** | **+1.79** | **+1.47** | **+1.17** | **+1.90** |
| cross spread, same snapshot | −1.07 | −0.14 | −0.25 | −0.47 |
| cross spread, 500 ms later | −4.49 | −1.86 | −1.62 | −3.25 |
| + fees 开0.23/平昨0.23 bp | −6.09 | −2.85 | −2.28 | −4.85 |
| + fees 开0.23/平今3.45 bp | −17.3 | −9.77 | −6.86 | −16.0 |

Read it top to bottom — each line is one layer of reality:
1. **The signal value is genuinely positive and grows with the threshold** (+0.3 → +1.9 ticks, t-stats 17–80). The forecast works. This line is the R² made tangible.
2. **Crossing the spread eats all of it.** Even with zero latency and zero fees, the best case (IF, q=2) is −0.14 ticks — the edge falls ~0.15–1 tick short of the round-trip spread. This is the precise, quantified meaning of "the signal lives inside the spread".
3. **Latency is brutal: one 500 ms snapshot costs ~0.7–3 ticks** on strong signals (IF −0.14 → −1.86). The forecast realizes *immediately* — most of the predicted move happens within the first snapshot. Whoever is faster collects it.
4. Commissions only bury the corpse deeper: 平昨 adds ~1–1.6 ticks, 平今 (3.45 bp) adds 5–12 ticks — the post-2015 intraday-close fee alone makes any 10-second round trip unthinkable.
Bottom line (fig85): at q=0.5 with realistic latency + 平昨 fees, the "strategy" loses 12–25万 CNY *per lot per day* with Sharpe ≈ −80 to −120. Nothing flips positive anywhere in the sweep.

So the honest monetization is unchanged, but now with numbers attached:
- As a **taker** you are 0.15–1 tick short *before* fees — dead on arrival, forever, unless your latency is near zero AND you only take the very strongest signals (and even then you're the marginal player).
- As a **maker** the economics invert: you *receive* the spread instead of paying it — the same +1.5-tick signal value that couldn't cover a crossing becomes protection for your quotes (lean/pull the side the forecast says will be run over).
- For us: a **short-horizon fair-drift overlay on executions we already need** — sit on the bid vs lift the ask when entering a slower signal; the 0.3–1.9 ticks/trade is exactly the budget such timing can save.
(Shen reports Sharpe ~6–7 on 2014 IF — but under a 0.25 bp cost assumption, at-touch fills, and a pre-crackdown market. Our table is what's left of that world.)

## 7. How the four papers fit together
| paper | x → y | horizon | verdict on our data |
|---|---|---|---|
| #1 CKS OFI | flow → *same interval* ΔP | contemporaneous | R² ≈ 0.5 — explains, doesn't predict |
| #2 Gould-Bonart QI | book level → *next move direction* | one tick ahead | 59–68% hit — predicts direction |
| #3 Shen VOI+ | flow + level + trades → next k-sec drift | 0.5–60 s | OOS R² up to ~0.11 — predicts magnitude |
| #4 Stoikov micro-price | book level → the *fair price now* | limit of tiny horizons | better ruler than mid (see guide #4) |

One object — the L1 book state and its changes — read four ways. The predictable component lives at **seconds** horizon and **sub-spread** amplitude; every method agrees.

## 8. Practice
1. Derive that VOI = (bid contribution) − (ask contribution) of paper #1's OFI when computed snapshot-to-snapshot. Where exactly do the definitions differ (hint: price-equal case uses ties)?
2. Recompute Model B for IF with MPB *not* divided by spread. Does δ stay stable across spread states?
3. Add lags 6..20. Watch in-sample R² creep up and OOS R² not follow (mild overfitting, live demo).
4. Compute the trigger-rule P&L gross and net of 1 tick: confirm §6's arithmetic yourself.
5. Shen backtests a threshold strategy (trade when |ŷ| clears a quantile) on 2014 IF — replicate it on IM (widest spread; hardest case) and compare net-of-fees results.

## 9. Mini-glossary
**VOI** volume order imbalance (snapshot flow) · **OIR** order imbalance ratio (queue level) · **MPB** mid-price basis (avg trade price − avg mid) · **spread-normalization** divide regressors by spread in ticks · **y_k** future k-snapshot average mid change · **OOS R²** variance explained on frozen coefficients, unseen data · **moment accumulation** pooled OLS from XᵀX/Xᵀy sums · **taker/maker** cross the spread vs rest in the queue.
