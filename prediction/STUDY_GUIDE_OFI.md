# Study Guide — Order Flow Imbalance (OFI) & Price Impact
### Based on Cont, Kukanov & Stoikov (2014), *The Price Impact of Order Book Events*, J. Financial Econometrics 12(1):47

A from-scratch guide for a quant beginner, built around the code in `xn/prediction/`.

---

## 0. The one-sentence idea
> **Prices move because of the net imbalance of order-book activity (orders added, cancelled, and executed) at the best bid and ask — and that relationship is linear and nearly parameter-free.**

Everything below unpacks that sentence.

---

## 1. Prerequisites — the order book (read this first if terms are fuzzy)

A **limit order book (LOB)** is the exchange's list of resting buy and sell orders:

```
        price      size
ASK →  6001.0       12     ← best ask (lowest price someone will SELL at)  = Pᵃ, qᵃ
       6000.8       30
       6000.6       45
       -------------------  ← the "spread" gap
       6000.4       28     ← best bid (highest price someone will BUY at)  = Pᵇ, qᵇ
BID →  6000.2       50
       6000.0       33
```

- **Best bid** `Pᵇ` (price) / `qᵇ` (size): the top buy order. **Best ask** `Pᵃ`/`qᵃ`: the top sell order.
- **Mid-price** `P = (Pᵇ + Pᵃ)/2`. This is the "price" we track.
- **Tick size** `δ`: the smallest price increment (CFFEX index futures: δ = 0.2 index points). We measure price changes in *ticks*: `ΔP` in units of δ.
- **Three event types** that change the book at the top:
  1. **Limit order** — someone posts a new resting order (adds size).
  2. **Cancellation** — someone withdraws a resting order (removes size).
  3. **Market order** (a.k.a. marketable / aggressive order) — someone crosses the spread to trade immediately (removes size from the opposite side).

Notation (per side, over a time interval `k`): `Lᵇ`=limit buys added, `Cᵇ`=buy cancels, `Mᵇ`=marketable buys; `Lˢ, Cˢ, Mˢ` for the sell side.

---

## 2. The model, step by step

**Step 1 — a single-side impact rule.** In a simple book, the bid price rises by one tick roughly when buy-side additions outnumber buy cancels + aggressive sells; it falls when the reverse holds. The paper writes this as a linear relation (their eq. 1–2). The key move is to combine both sides.

**Step 2 — Order Flow Imbalance (OFI).** Sum all six event types with signs by whether they push price up or down (their eq. 4):
```
OFIₖ = Lᵇ − Cᵇ − Mˢ − Lˢ + Cˢ + Mᵇ
```
Intuition: things that create buy pressure (buy limits added `+Lᵇ`, sell orders removed `+Cˢ`, aggressive buys `+Mᵇ`) are positive; things that create sell pressure are negative.

**Step 3 — the impact equation (their eq. 3):**
```
ΔPₖ = OFIₖ / (2D) + εₖ
```
- `ΔPₖ` = mid-price change over interval k, in ticks.
- `D` = **market depth** (average size resting at the best quotes) — the *only* parameter.
- `εₖ` = noise (things OFI doesn't capture).
- **Slope = 1/(2D).** Thin book (small D) → big price moves per unit of net flow. Deep book (large D) → small moves.

**Step 4 — building OFI from Level-1 snapshots (what our data has).** We don't see individual events, only 500 ms snapshots of `(Pᵇ, qᵇ, Pᵃ, qᵃ)`. Cont et al. show the net flow between two consecutive snapshots is recoverable from price/size changes:
```
bid contribution:  Pᵇ↑ → +qᵇ(new)      Pᵇ same → Δqᵇ      Pᵇ↓ → −qᵇ(old)
ask contribution:  Pᵃ↑ → +qᵃ(old)      Pᵃ same → −Δqᵃ     Pᵃ↓ → −qᵃ(new)
OFIₙ = bid contribution + ask contribution
```
Sum `OFIₙ` over the snapshots in interval k → `OFIₖ`. (Code: `ofi_ddb.py` / `ofi_full_ddb.py`.)

**Step 5 — fit it.** Ordinary least squares of `ΔP` on `OFI`. Read off:
- **R²** = fraction of price variance explained (0 = nothing, 1 = perfect). High R² ⇒ order flow really is what moves price.
- **slope → D = 1/(2·slope)** = the depth / inverse-impact.

---

## 3. Trade Imbalance (TI) — and why OFI beats it (their eq. 5–6)
```
TIₖ = Mᵇ − Mˢ       (aggressive buys − aggressive sells; only TRADES)
ΔPₖ = TIₖ/(2D) + ηₖ
```
TI is the "obvious" flow measure, but it ignores limit orders and cancellations — which also move price. So OFI explains **~2–3× more** of the variance than TI. That gap is the paper's headline. (Our data: `m_nActBidVolume/m_nActAskVolume` are the aggressor volumes → `Mᵇ, Mˢ`.)

---

## 4. The crucial distinction: impact vs prediction ⚠️
Eq. (3) is **contemporaneous** — OFI and ΔP over the *same* interval. It explains what *already* moved the price; it does **not** forecast the next interval. When you lag it (OFI now → return next interval), R² collapses to ≈0. That's market efficiency: once the flow is observed, the move has happened. Remember this — it's the #1 thing beginners misread about OFI.

---

## 5. Extensions (in the paper and the follow-on literature)
- **Impact ∝ 1/depth (the paper's deeper result).** The impact coefficient isn't a universal constant — it *scales inversely with depth D*, cross-sectionally and intraday. Normalizing by depth gives a stable, near-universal linear law. (Connects to **Kyle's λ**, the classic price-impact coefficient from Kyle 1985.)
- **Linear, not concave.** The older "trade size → impact" literature found a *concave* (square-root) law; Cont et al. argue that's partly an artifact of ignoring limit-order activity — with OFI the relation is *linear*.
- **Intraday depth variation.** D is U-shaped over the day (deep midday-ish, thinner at open/close), so raw impact varies intraday even though the OFI/depth relation is stable.
- **Multi-level / "integrated" OFI** (Cont, Cucuringu, Zhang and others, ~2021+): use several book levels, combine via PCA into one factor → higher explanatory power and some short-horizon predictive power.
- **Cross-impact** (Cont et al. 2023): one asset's OFI moves *other* correlated assets' prices → an OFI cross-impact matrix. Relevant for index-futures ↔ constituents.

---

## 6. Our implementation (files in `xn/prediction/`)
| file | what it does |
|---|---|
| `ofi_ddb.py` | one-month pilot (2024-06): fetch L1, build OFI/TI, regress, save |
| `ofi_full_ddb.py` | full 2020–2026: chunk by month, accumulate OLS *moments* for exact pooled R²/D |
| `ofi_plot.py` / `ofi_full_plot.py` | figures: dP-vs-OFI scatter, R² vs interval, R² stability over time |
| `ofi_results*.csv`, `ofi_scatter*.csv`, `ofi_permonth.csv` | verifiable outputs |

Pilot result (2024-06): contemporaneous **R²(OFI) ≈ 0.61–0.72** at 10–60 s; **R²(TI) ≈ 0.22–0.30**; **predictive R² ≈ 0**; depth **D ≈ 1.6 (IC/IM), 4–5 (IF/IH)**.

---

## 7. Skills you needed here — and how to practice each

1. **Market microstructure** (order book, limit vs market orders, spread, depth, tick).
   *Practice:* read Bouchaud–Bonart–Donier–Gould, *Trades, Quotes and Prices* (the bible) or Larry Harris, *Trading and Exchanges*; then reconstruct a book from raw messages by hand.
2. **Probability & statistics** (OLS, R², correlation, t-stats, pooling via sufficient statistics).
   *Practice:* derive the OLS slope/R² formula yourself; recompute our pooled R² from `[n, Σx, Σy, Σxy, Σx², Σy²]`; run regressions on synthetic `y = a·x + noise` and watch R² change with noise.
3. **Time-series discipline** (stationarity, autocorrelation, lead–lag, look-ahead bias, in-sample vs out-of-sample).
   *Practice:* redo the contemporaneous-vs-lagged test; deliberately introduce look-ahead and see the fake R² appear.
4. **Programming / data engineering** (Python pandas+numpy, querying tick DBs, chunking, dedup, session/timestamp handling).
   *Practice:* reproduce `ofi_ddb.py` from scratch; handle the 2024-02 duplicate rows and lunch-break gaps yourself.
5. **Financial-data literacy** (field meanings, tick size, contract specs, data-quality quirks, roll/continuous-contract construction).
   *Practice:* explore a table schema; validate a stat against a known fact (e.g., "500 ms cadence ⇒ ~120 ticks/min"); spot the dense-month artifact.
6. **Linear algebra** (for multi-level OFI / PCA / cross-impact matrices).
   *Practice:* run PCA on a small multi-level OFI matrix; interpret the first component.
7. **Research skepticism** (overfitting, transaction costs, robustness across time & instruments, coverage/sample-size awareness).
   *Practice:* for every "edge," ask: in-sample? net of the spread? holds out-of-sample? enough sample at the tail?
8. **Reading papers** (extract model → assumptions → results → limitations; reproduce one figure).
   *Practice:* read the Cont paper section by section and reproduce its OFI-vs-TI R² comparison (you just did the equivalent).

---

## 8. Suggested order to study
1. Order book basics (§1) → build one from data.
2. OLS + R² by hand (§2, §7.2).
3. Reproduce `ofi_ddb.py`; get the contemporaneous R² (§2, §6).
4. Add TI and compare (§3).
5. Do the lagged/predictive test; internalize impact≠prediction (§4).
6. Then extensions: depth-scaling & Kyle's λ, then multi-level/cross-impact (§5).

---

## 9. Mini-glossary
- **LOB** limit order book · **Pᵇ/Pᵃ** best bid/ask price · **qᵇ/qᵃ** best bid/ask size · **mid** (Pᵇ+Pᵃ)/2 · **δ** tick size · **ΔP** mid change in ticks · **OFI** net signed order-book flow · **TI** trade (aggressor) imbalance · **D** depth (slope=1/2D) · **R²** variance explained · **contemporaneous** same interval · **predictive** next interval · **Kyle's λ** classic price-impact coefficient.
