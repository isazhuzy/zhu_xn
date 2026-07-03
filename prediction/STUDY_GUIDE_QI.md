# Study Guide — Queue Imbalance (QI) & the Direction of the Next Price Move
### Based on Gould & Bonart (2016), *Queue Imbalance as a One-Tick-Ahead Price Predictor in a Limit Order Book*, Market Microstructure and Liquidity 2(2). [arXiv:1512.03492]

Paper #2 in our series. A from-scratch guide built around `qi_ddb.py` / `qi_plot.py`.
(Read `STUDY_GUIDE_OFI.md` §1 first if the order book itself is still fuzzy.)

---

## 0. The one-sentence idea
> **The relative size of the two queues at the best quotes — one number you can read off the screen right now — predicts the direction of the *next* mid-price move with well over 50% accuracy.**

This is the simplest genuinely *predictive* order-book signal known. Where the OFI paper (paper #1) explained the price move that *already happened*, this paper forecasts the one that *hasn't happened yet*.

---

## 1. The variable: queue imbalance

At any moment the best bid has `qb` contracts resting, the best ask has `qa`. Define

```
I = (qb − qa) / (qb + qa)      ∈ [−1, +1]
```

- `I → +1`: bid queue huge, ask queue tiny → wall of buyers below, thin sellers above.
- `I → −1`: the mirror image.
- `I = 0`: balanced book.

That's the whole signal. No history, no trades, no parameters — a single snapshot of two numbers. (In Shen's thesis, paper #3, the same quantity is called **OIR**.)

## 2. The prediction target: direction of the *next* mid move

The mid `M = (Pb+Pa)/2` stays constant for a while, then jumps (a best-quote price changed). The paper asks:

> standing at time t, what is **P(the next mid change is UP | I at time t)**?

Note what is *not* asked: how big the move is, or when it comes. Direction only. This makes the problem a clean binary classification, evaluated by a probability curve rather than an R².

## 3. The paper's method and findings

1. For each snapshot, record `I` and the sign `y ∈ {up, down}` of the next mid move.
2. Estimate the empirical curve `P(up | I)` by binning `I`.
3. Fit a **logistic regression**: `P(up|I) = 1/(1 + e^−(a + b·I))`. Logistic, because a probability must live in (0,1) — a straight line would escape it.

Findings on 10 Nasdaq stocks (2014):
- `P(up|I)` is **monotone increasing** in I — everywhere, for every stock.
- The relationship is much **stronger for large-tick stocks** (price ≈ few dollars, spread almost always = 1 tick) than for small-tick stocks (spread many ticks wide). For large-tick stocks the curve spans ~0.2 → ~0.8; for small-tick ones it is much flatter.

## 4. WHY does it work? (the queue race)

Think of the two best queues as racing to zero. A quote level disappears when its queue is fully eaten/cancelled — and the mid then moves *toward the side that died*.

- If `qa` is much smaller than `qb`, the ask queue will (probabilistically) die first → next move UP. Purely **mechanical**: even with random order flow, the thin side loses the race.
- On top of the mechanics sits **information**: traders who know something consume the side that's in their way; a thin ask often *is* the footprint of buyers having eaten it already.

Both stories predict the same sign, which is why the effect is so robust. But note the flip side: since it's half-mechanical and visible to everyone, it is **priced into where you can actually trade** — see §7.

## 5. Our adaptation to CFFEX index futures (`qi_ddb.py`)

| paper (Nasdaq 2014) | ours (CFFEX 2020–2026) |
|---|---|
| event-by-event book updates | 500 ms L1 snapshots (`hft_future_ts`) |
| 10 stocks, 1 year | IC/IF/IH/IM continuous contracts, 6.4 years |
| large-tick vs small-tick *stocks* | spread-state split *within* each contract: 1 / 2 / ≥3 ticks |

Implementation notes:
- "next mid move" = first future snapshot-to-snapshot mid change within the same session (AM/PM separated; overnight and lunch never crossed). Code: sign of `mid.shift(−1) − mid`, zeros → NaN, then **backfill** within session — a neat vectorized way to say "direction of the next change".
- I==0 rows go to their own bin (they sit exactly on the decision boundary).
- Grouped logistic fit by Newton's method on 40 bins — mathematically identical to row-level logistic regression when the regressor is binned, and ~10⁶× cheaper on 100M rows.
- Data quirks handled as usual: duplicate timestamps (2024-02), one-sided quotes dropped, |Δmid| > 50 ticks treated as bad ticks (cf. the IH 2024-10-08 halved-quote incident).

## 6. Our results (full sample 2020-01 … 2026-05)

> Numbers below are from `qi_results.csv` / `qi_permonth.csv`; figures 71–73.

- **P(up|I) is monotone and near-logistic for all four contracts** (fig71) — the paper's main claim reproduces cleanly on a completely different market, decade, and asset class. Sample: 22–37 **million** classified snapshots per contract.
- **Sign rule hit rate** (predict "up" iff I>0): **IH 61.5%, IF 61.0%, IC 58.4%, IM 58.4%** — every contract, every month above 50% (fig73). Pooled logistic slope b ≈ 0.71–1.08, intercept a ≈ 0 (no directional bias), pseudo-R² 0.015–0.035.
- **The tick-size effect reproduces as a spread-state effect** (fig72): when the spread is 1 tick, hit rates jump to **66.2–66.7% (IF/IH), 62.5–63.1% (IC/IM)** with pseudo-R² up to 0.075; at ≥3 ticks they sag to ~57%. IF/IH spend far more time in the tight-spread regime (large-tick-like), IC/IM in the wide regime — so *across* contracts we see exactly the paper's *cross-stock* pattern.
- Stability: no sign of decay over 2020→2026; this is a structural property of the matching mechanism, not a regime artifact (contrast with our `ticker/` open-reversal, which flipped sign by year).

## 7. Can you trade it? (the sobering arithmetic) ⚠️

A 62% coin sounds like money. It isn't, by itself:

- The move you predict is the next **mid** change — typically ±0.5 to ±1 tick around the current mid.
- To bet on it you must **cross the spread** (≥1 tick) or join a queue (and the queue you'd join is exactly the long one, which fills last — adverse selection eats you).
- Expected gross edge ≈ (2·0.62−1) × ~0.6 tick ≈ 0.14 tick, vs ≥1 tick round-trip cost. Same verdict as all our previous studies: **real signal, inside the spread.**

Where it *is* used in practice: market makers use QI to decide **when to lean/pull quotes** (avoid being the last one in the dying queue), execution algos use it to time child orders — i.e., it improves prices for trades you were going to do anyway, rather than creating trades.

## 8. Extensions & follow-on literature
- **Lipton, Pesavento & Sotiropoulos (2013)** — analytic P(up|I) under diffusing queues: `P(up) ≈ ½(1 + arctan-shaped curve)`; our empirical curves match its S-shape.
- Add **deeper book levels**, recent trade flow, and time-of-day → you converge to Shen's model B (paper #3) and eventually DeepLOB.
- Queue *position* models (where in the queue is your order) — the market-making view.

## 9. Practice
1. Re-derive why `bfill` on the signed-diff series gives "direction of next change". Break it by removing the session grouping — what leaks?
2. Refit the logistic on one month by raw MLE (row-level) and confirm it matches the grouped fit.
3. Split the curve by hour of day. Is the open different?
4. Compute the *time until* the next mid move as a function of |I|. (Strong imbalance should also mean *sooner*.)
5. Simulate the queue race: two independent Poisson-depleted queues of sizes qb, qa — derive P(ask dies first) and compare its shape with fig71.

## 10. Mini-glossary
**QI / I** queue imbalance at best quotes · **large-tick asset** spread almost always = 1 tick (tick "feels big") · **grouped logistic** logistic MLE on binned counts · **pseudo-R²** 1 − LL/LL₀ (McFadden), the logistic analogue of R² · **adverse selection** you get filled exactly when you're wrong · **spread state** spread in ticks at signal time (our 1/2/3+ split).
