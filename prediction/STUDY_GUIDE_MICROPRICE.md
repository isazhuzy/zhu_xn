# Study Guide — The Micro-Price: a Better "Fair Price" than the Mid
### Based on S. Stoikov (2018), *The micro-price: a high-frequency estimator of future prices*, Quantitative Finance 18(12), 1959–1966.

Paper #4 in our series. Code: `microprice_ddb.py` / `microprice_plot.py`.

---

## 0. The one-sentence idea
> **The "true" price of an asset, given the order book you see, is not the mid — it is the mid plus a small correction g\*(imbalance, spread), chosen so that the corrected price has no predictable drift. That correction can be estimated from data as a Markov chain, and the result predicts future mids better than the mid or the weighted mid.**

Papers #2–#3 said "the book predicts the price". This paper flips the logic: if the book predicts the mid, then the mid was the wrong price to begin with. Fix the price instead of trading the forecast.

---

## 1. Three candidate "prices" and what's wrong with them

1. **Mid** `M = (Pb+Pa)/2`. Ignores imbalance entirely. Paper #2 proved the book drifts away from it predictably → **biased, under-reacts**.
2. **Weighted mid** `Pw = I·Pa + (1−I)·Pb` with `I = qb/(qb+qa)` (big bid queue pushes fair price toward the ask). Better on average, but it **over-reacts**: queue sizes are noisy and mean-revert; a 10:1 imbalance does not move fair value 90% of the way to the ask. Empirically its future drift has the *opposite* sign of the mid's — you can see this in our fig92.
3. **Micro-price** `P_micro = M + g*(I, s)`: let the *data* say how far to shift, as a function of imbalance `I` **and** spread `s` — the two state variables papers #2–#3 identified.

The one mental model: the three prices differ only in **how hard they react to queue imbalance** — mid reacts *zero*, wmid reacts *too much* (copies the raw queue ratio), micro reacts *just right* (a data-calibrated amount). Worked on a concrete book — bid 3900.8 (21 lots), ask 3901.0 (4 lots), so `I = qb/(qb+qa) = 21/25 = 0.84`, buyers dominant:
```
mid   = (3900.8 + 3901.0)/2                 = 3900.90   ← ignores 21-vs-4, stays centred (too low)
wmid  = 0.84·3901.0 + 0.16·3900.8           = 3900.97   ← copies the ratio, lurches toward ask (too far)
micro = 3900.90 + g*(I=0.84, s=1 tick)                  ← +a calibrated bump, sized to kill drift
```
Ideal ordering (buyers dominant): **mid < micro < wmid**. (Our frozen 5-yr g\* over-shoots, so empirically micro can exceed wmid — a magnitude issue, §5, not a flaw in the ordering logic.)

## 2. The defining property: a martingale
A good fair price should be **drift-free**: knowing today's book, you cannot predict where the fair price goes next (otherwise it isn't fair yet — fold the prediction in!). Formally Stoikov defines
```
P_micro(t) = lim_{n→∞} E[ M_{τ_n} | book state at t ]
```
the expected mid at the time of the n-th future *price change*, letting n grow. Each layer of expectation absorbs one step of predictability; the limit has none left.

## 3. The construction, step by step (this is the technical heart)

**State.** `x = (spread s, imbalance bin i)`. Ours: s ∈ {1, 2, ≥3 ticks} × 10 imbalance bins = 30 states. Imbalance here is `I = qb/(qb+qa) ∈ [0,1]` (note: the [0,1] cousin of paper #2's OIR ∈ [−1,1]; `I = (OIR+1)/2`, so `I=0.5` is balanced). State index = `(spread_tier − 1)·10 + ibin`.

The full 30-state map, with IF's estimated `g*` in ticks (this is fig91 as numbers; sign follows imbalance, magnitude grows toward the extremes, buy/sell antisymmetric by construction):

| spread ＼ ibin | 0 (ask-dom) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 (bid-dom) |
|---|---|---|---|---|---|---|---|---|---|---|
| **1 tick** (states 0–9) | −1.22 | −0.85 | −0.58 | −0.35 | −0.09 | +0.09 | +0.35 | +0.58 | +0.85 | +1.22 |
| **2 tick** (states 10–19) | −1.06 | −0.75 | −0.50 | −0.29 | −0.08 | +0.08 | +0.29 | +0.50 | +0.75 | +1.06 |
| **≥3 tick** (states 20–29) | −0.86 | −0.62 | −0.43 | −0.25 | −0.07 | +0.07 | +0.25 | +0.43 | +0.62 | +0.86 |

(`ibin = floor(I·10)`, so ibin 0 = I∈[0,0.1) sellers dominate, ibin 9 = I∈[0.9,1] buyers dominate; ibin 4/5 straddle the balanced point.)

**Events.** Watch consecutive snapshots. Either
- the mid does **not** move → the state just wanders: count into matrix `T[x → x']`, or
- the mid **moves** by `ΔM` → record the jump for the state we came *from*: `Rsum[x] += ΔM`, `Rcnt[x] += 1`, and where the state lands: `B[x → x']`.

**Step 1 — expected move at the FIRST price change.** Row-normalize by total exits `N[x]`; `T̂` = no-move transitions, `r̂[x]` = mean immediate jump contribution. Then
```
G1 = (Id − T̂)⁻¹ · r̂
```
The matrix inverse implements "wander around no-move states as long as needed, collect the jump when it finally comes" — an infinite sum `Σ T̂^k r̂` in closed form. `G1(x)` = expected mid change between now and the *next* price move, given state x. (Same trick as absorbing-state Markov chains / first-step analysis.)

**Step 2 — then keep going.** After that first move you land in a new state (distribution `B̂`, row-normalized from `B`), from which the *next* move has expected drift `G1` again:
```
g* = G1 + B̂·G1 + B̂²·G1 + … + B̂⁶·G1        (converges fast; 6 terms like the paper)
```
Each term is the drift harvested at the 1st, 2nd, … price change. The sum stabilizes because after several price moves the state has mixed — no information left.

**Step 3 — symmetry.** Buying and selling are mirror images, so impose `g*(I,s) = −g*(1−I,s)` by antisymmetrizing (the paper does the same via data symmetrization).

**Toy example to internalize G1.** Two states: A = "balanced" (mid never moves, goes to B with prob 1), B = "tilted" (mid +1 tick with prob ½, back to A with prob ½). Then G1(B) solves G1(B) = ½·1 + ½·G1(A), G1(A) = G1(B) → G1(A) = G1(B) = 1. The inverse `(Id−T̂)⁻¹` is doing exactly this algebra for 30 states at once.

## 4. Our implementation (`microprice_ddb.py`)
- **Train 2020-01…2024-12** → count `T̂, B̂, r̂` per contract, compute g* once, freeze.
- **Test 2025-01…2026-05** → horse race on unseen data: predict `M_{t+h}` for h ∈ {1,4,20,120} snapshots with (a) mid, (b) weighted mid, (c) micro-price; compare RMSE, and plot the **conditional drift** `E[M_{t+20} − price | imbalance decile]` — the paper's signature figure.
- Adaptations: 500 ms snapshots instead of event time (so "price move" = between-snapshot mid change ≥ 0.5 tick); 30-state grid; bad-tick guards as usual.

## 5. Our results (full sample)

> Definitive numbers in `mp_gstar.csv`, `mp_rmse.csv`, `mp_bias.csv`; figures 91–93.
> Two experiments: **pilot** (train 2024-06 → test 2024-07, adjacent months) and
> **full** (train 2020–24 → test 2025–26, frozen for up to 6 years). They disagree in
> an instructive way.

**How we test "fairness" (turning the definition into an experiment).** Fair = drift-free. So for each candidate price P, measure its future drift conditional on imbalance — `E[M_{t+20s} − P | I]` — and plot vs I (fig92). The shape *is* the verdict:
- a **truly fair** price → **flat line at 0** (no predictable drift, any I);
- **mid** → **up-sloping** line (positive gradient): buyers dominant ⇒ future mid rises above the frozen mid ⇒ positive drift = **under-reaction**;
- **wmid** → **down-sloping** line: it overshot toward the ask, so the future mid comes back *below* it = **over-reaction** (mirror of the mid);
- **micro** → the **flattest** line if g\* is well-calibrated. (Frozen 5-yr g\* over-shoots → micro also slopes down, but less than wmid — see below.)

- **g\* is a clean S-curve in imbalance, for every spread state** (fig91) — paper #2's logistic curve reborn as a price adjustment. Magnitudes reach ~±1.2 ticks at extreme imbalance, *larger* than event-time studies report: our 500 ms snapshots aggregate many events, so a between-snapshot "price move" is often several ticks. (Adaptation artifact, not a bug.)
- **Adjacent-month test (pilot) reproduces the paper exactly**: micro beats mid and wmid at 0.5s and 2s for all four contracts (fig92/93 `_pilot`: micro's conditional-drift line is the flattest).
- **Frozen 5-year test tells a subtler story** (fig92/93): the mid still under-reacts (up-sloping drift) and wmid still over-reacts badly (RMSE +2–4% at 0.5s). But the 2020–24-trained g\* now **over-corrects** in 2025–26 — micro's drift line slopes down, sitting between mid and wmid. RMSE at 0.5s: micro still wins for IC (−0.6%), IF (−1.2%), IM (−0.8%); IH ties (+0.2%); beyond 2s all three converge within 0.4%.
- **The lesson — shape is structural, magnitude drifts.** The S-curve never changes sign or shape, but its scale was ~2× larger in the wild 2020–22 years than in 2025–26. A micro-price must be **re-estimated on a rolling recent window** (weeks–months, as Stoikov effectively does), unlike paper #2/#3's coefficients which survived a 5-year freeze much better. The better a parameter fits the microstructure of one era, the faster it stales.
- Implementation quirk worth knowing: a perfectly balanced book (qb=qa, I=0.5 exactly) falls into bin 5, inflating its occupancy; antisymmetrization keeps the signs honest, but a dedicated I=0.5 bin would be cleaner.
- Read the magnitudes honestly: a ~1% RMSE gain will never be an alpha. That's not the point — see §6.

## 6. What a micro-price is FOR
It's a **better ruler, not a crystal ball**:
- **Feature engineering**: compute returns/signals off `P_micro` instead of mid — every study in `xn/` that used mid-price bars inherits a tiny bias that the micro-price removes (e.g. the intraminute momentum work).
- **Market making**: quote around `P_micro`, not the mid — you systematically avoid quoting the wrong side when the book is tilted.
- **Execution / marking**: a fill at the bid when `P_micro` ≈ bid was not a good fill; benchmark executions and mark inventories at micro, not mid.
- The three uses share one theme: anywhere "the price" enters a formula, the micro-price is a strictly better plug-in.

## 7. How it connects to papers #1–#3
- Paper #2 (QI): `P(next move up | I)` monotone in I ⇒ mid is biased ⇒ *there exists* a correction g*. Stoikov computes it.
- Paper #3 (Shen): model B's fitted `E[y_k | book]` is a regression cousin of g*; Stoikov's Markov chain is the non-parametric, horizon-consistent version.
- Paper #1 (OFI): once prices are marked at micro instead of mid, part of the "impact" of imbalance-correlated flow is absorbed into the ruler — the residual impact coefficient is cleaner.
- Deep-learning LOB models (Sirignano–Cont) effectively learn g* with more state — same object, bigger basis.

## 8. Practice
1. Work the 2-state toy example of §3 by hand, then with `numpy.linalg.solve`, and verify `(Id−T̂)⁻¹r̂` gives the same answer.
2. Plot `G1` vs the 6-term `g*` from `mp_gstar.csv`. How much do the later bounces (terms 2–6) add? (Paper: most of the value is in the first two terms.)
3. Estimate g* on 2020–2022 only and compare to 2023–2024's. Stationary?
4. Recompute fig92 at h=120. Everything should flatten — why?
5. Take the intraminute momentum study and recompute its signal off `P_micro`. Does the within-minute reversal change?

## 9. Mini-glossary
**micro-price** mid + g*(imbalance, spread) · **martingale** no predictable drift · **weighted mid** imbalance-weighted average of best quotes · **first-step analysis** conditioning on the first transition; source of the (Id−T̂)⁻¹ · **T̂ / B̂ / r̂** no-move transitions / at-move transitions / mean immediate jump · **G1** expected mid change at the next price move · **antisymmetrization** enforcing buy/sell mirror symmetry · **horse race** out-of-sample RMSE comparison of estimators.
