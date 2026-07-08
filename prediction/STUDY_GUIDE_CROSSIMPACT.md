# Study Guide — Cross-Impact: does one index future's order flow move the others?
### Based on Cont, Cucuringu & Zhang (2023), *Cross-Impact of Order Flow Imbalance in Equity Markets*, Quantitative Finance 23(10). [arXiv:2112.13213]

Direction A of round 2 (the first study that *needs* more than one instrument). Code:
`crossimpact_ddb.py` / `crossimpact_plot.py`. Read `STUDY_GUIDE_OFI.md` (OFI) first.

---

## 0. The one-sentence idea
> **Order-flow imbalance in one instrument helps predict the *next* move of *other* correlated instruments — but on our four index futures that "cross-impact" turns out to be a shared-market-factor effect (read the whole basket to nowcast the common move better), not a clean "X leads Y."**

Papers #1–#4 lived inside one contract. This is the first question that only four contracts (IC/IF/IH/IM) can answer.

---

## 1. The one new problem: asynchronous clocks
Within one contract, consecutive snapshots are trivially ordered. Across contracts they are **not**: IF updates at its own times, IC at its own — their 500 ms snapshots don't line up. To regress IC's return on IF's OFI you first need them on a **common clock**.

Fix (Cont's aggregation): floor every tick to a wall-clock bin of width `W` seconds; per (contract, session, bin) **sum** the tick-level OFI and take the **last** mid. Empty/stale bins forward-fill the mid (⇒ return 0) and get OFI 0. Now all four are synchronous by construction.

⚠️ **W must be ≥ the native cadence.** Our snapshots arrive ~every 500 ms but not phase-locked to wall-clock 500 ms bins, so a `W=0.5s` grid *aliases* — half the bins are stale-fills and the predictive R² collapses to ~0. This is a grid artifact, not "no signal": at 0.5 s the right tool is the native-tick VOI regression (paper #3), not this grid. **Use W ≥ 1 s here.**

## 2. Two questions, two methods

### 2.1 Who leads whom? (model-free) — return lead-lag
For each ordered pair, the cross-correlation `corr(r_a(t), r_b(t+ℓ))` across lags ℓ.
- Peak at **ℓ = 0** → they move together, no lead.
- Peak at **ℓ > 0** → `a` leads `b` (a's move today predicts b's tomorrow).

### 2.2 Does cross-OFI predict? (Cont 2023) — the cross-impact regression
For each target contract `i`, predict its next-bin return from *everyone's* current OFI:
```
r_i(t+1) ~ β0 + Σ_j β_ij · OFI_j(t)            j ∈ {IC, IF, IH, IM}
```
Compare **own-only** (`r_i ~ OFI_i`) vs **full** (all four OFIs). If the cross terms `β_ij (j≠i)` add out-of-sample R², cross-OFI carries information. Same discipline as papers #3–#4: **freeze β on train (2022-07…2024-12), score on test (2025-26)**; exact pooled OLS via accumulated `XᵀX / Xᵀy` moments.

## 3. Our results (2022-07 … 2026-05; train ≤2024-12, test 2025-26)

> Numbers in `xc_leadlag.csv` / `xc_predict.csv` / `xc_betas.csv`; figs 101–103.

**Q1 — no lead-lag above 500 ms** (fig101). All six pairs are symmetric tents **peaked at ℓ=0** at every grid (0.5/1/2 s). Lag-0 correlations 0.23–0.41; the ±1-bin asymmetries are tiny (IF↔IC 0.205 vs 0.175 — a whisper that liquid IF leads, within noise). Any real lead-lag is finer than our snapshot cadence.

**Q2 — cross-OFI ~triples the OOS R², and it holds** (fig102), W=1 s:

| target | own-OFI OOS R² | +cross OOS R² | gain |
|---|---|---|---|
| IC 中证500 | 0.0154 | **0.0576** | 3.7× |
| IF 沪深300 | 0.0187 | **0.0506** | 2.7× |
| IH 上证50 | 0.0173 | **0.0439** | 2.5× |
| IM 中证1000 | 0.0175 | **0.0410** | 2.3× |

Train-full for IC was 0.066 vs OOS 0.058 — no collapse, ~4.6 M test rows each. At W=2 s the same pattern at ~⅓ the magnitude (R² decays with horizon, as always).

**The β-matrix has structure** (fig103): a **small-cap block** — IC and IM drive each other *more than themselves* (IC←IM 0.063, IM←IC 0.071 > their own diagonals). IH (large-cap 上证50) is the least predictable target.

## 4. The honest interpretation: common factor, not causality ⚠️
The cross-OFI gain is real and OOS-robust — but it is **most likely a shared-factor effect, not "IF causes IC."** The logic:
- returns co-move at **lag 0** (Q1), and OFI is persistent, so `OFI_j(t)` correlates with `r_j(t+1)`, which ≈ `r_i(t+1)` contemporaneously;
- therefore knowing *all four* books gives a better **nowcast of the common market move** than one book alone — and that transitively "predicts" `r_i(t+1)` with no directional cross-impact needed.

The IC↔IM block says there's a **small-cap sub-factor** on top of the broad-market factor (中证500 & 中证1000 share small-cap exposure). This matches Cont-Cucuringu-Zhang's own finding: once OFI is integrated, cross-impact largely reflects common structure. To isolate *genuine* directional cross-impact you'd have to strip the common factor first (regress out the market/basket return, then test residual cross-OFI) — a clean next experiment.

## 5. Is it tradeable?
Same spread arithmetic as every prior paper. R²≈0.06 at 1 s ≈ the single-contract VOI edge — a real *nowcast* improvement, but the predicted move is sub-spread, so for a **taker** it's still inside the round-trip cost. Where a basket read could pay: an **index/basket market maker** (quote all four off a common-factor fair value estimated from every book), or **hedge-ratio / execution timing** across the four contracts. Not a standalone taker alpha.

## 5.5 Closing chapter — the IC↔IM zoom and the maker capstone

The §4 caveat said: the cross-signal *reads like* a common factor; to find genuine structure, isolate a pair. We did — the small-cap pair IC↔IM — and then followed it all the way to a money question. Three experiments (`xc_icim_ddb.py`, `xc_icim_backtest_ddb.py`, `xc_maker_ddb.py`; figs 104–106).

**(a) IС↔IM is a real, ~80%-genuine small-cap sub-factor** (fig104). Pairwise `r_IC(t+1) ~ OFI_IC + OFI_IM`: adding the partner's flow ~2.4× the R², and it **holds OOS** (IC 0.018→0.044 IS, 0.047 OOS). The key test — *is it just market beta?* — controlling for the large-caps IF/IH, the cross-β shrinks only from 0.087→0.071 (IM→IC) and 0.105→0.083 (IC→IM): **~80% survives ⇒ genuine small-cap factor, ~20% broad-market leakage.** Lead-lag peaks at lag 0 in both IS and OOS, with a faint, consistent **IM-slightly-leads-IC** (lag −1 > lag +1).

**(b) As a taker it's dead — and "more accurate" made it worse** (fig105). Trade IC on the frozen cross-signal, 1 s hold, fig85 cost ladder. Frictionless signal value: own 0.45 → **pair 0.72 tick** (IM genuinely sharpens the forecast). But IC is small-cap with a *wide* spread, so crossing it costs ~4 ticks → taker0 −3.85, +fees −6 to −17. Net: **−1.09 M 元/day per lot for the pair vs −0.97 M for own** — the *better* signal loses *more*, because it fires more often on a still-sub-spread edge. The sharpest "signal ≠ alpha": a better forecast of a sub-spread move just lets you lose more confidently.

**(c) As a maker it flips — the capstone** (fig106, `xc_maker_ddb.py`). Rest at pb/pa, fills inferred from real aggressor volume (`m_nActAsk/BidVolume`), P&L = markout (`mid(t+H)−pb` / `pa−mid(t+H)`), and a fair-value F decides which side to quote (F>mid ⇒ keep bid, pull ask). Three centers, frozen train, 2025-26 OOS:

| quote center | markout/fill | fills |
|---|---|---|
| mid (naive, no skew) | +2.07 tick | 4.88 M |
| P_micro (paper #4) | +2.21 tick | 3.95 M |
| **P_micro + cross-drift (paper #4+5)** | **+2.95 tick** | 2.52 M |

Each signal layer **pulls more toxic fills** (count ↓) and **lifts per-fill edge** (2.07→2.95, **+0.88 tick** cross vs mid), out-of-sample. The wide IC spread that *cost* the taker −4.5 ticks *pays* the maker ~+2 — same spread, opposite sign, decided by rest-vs-cross.
⚠️ **Gross & optimistic**: assumes always-filled-at-touch (no queue priority — real fills come preferentially when flow is heavy = toxic), flatten-at-mid, no fees. The robust claim is the **relative** mid<micro<cross ordering, **not** the absolute +2 tick. A queue-aware sim is the honest next step.

**The arc's payoff:** five papers found real, OOS-stable, sub-spread signals; four taker backtests said *no alpha*. The maker experiment shows the same signals have **real, measurable value on the quoting side** — reducing adverse selection by ~0.9 tick/fill. The project's thesis, now nailed with data: *the edge isn't in predicting price as a taker; it's in quoting smarter as a maker.*

## 6. How it connects to papers #1–#4
- Paper #1 (OFI): this is OFI applied across instruments instead of within one; the aggregation into `W`-bins is the same "sum tick OFI over an interval" move.
- The common-factor reading is the multi-asset cousin of paper #4's insight that "the book is biased" — here the whole *basket's* books jointly pin down the common fair-value move.
- Deep-LOB / multi-asset ML (Kolm-Turiel-Westray 2023; Sirignano-Cont 2019) learn this cross-structure with a bigger basis — same object, richer model.

## 7. Practice
1. Strip the common factor: build the equal-weight (or PCA-first-component) basket return, regress each contract's return on it, and re-run 2.2 on the **residuals**. Does any cross-OFI β survive? (This is the real test of §4.)
2. Re-do Q1 at tick level (not binned) with the Hayashi–Yoshida estimator (handles async ticks natively) — is there a sub-500 ms lead you can't see on the grid?
3. Add lag-2/lag-3 cross-OFI to 2.2. Does deeper history help, or is it all in lag-1? (Cont: decays fast.)
4. Split the β-matrix train vs test — is the IC↔IM small-cap block stable across the 2024→2025 boundary?

## 8. Mini-glossary
**cross-impact** one asset's order flow moving another's price · **lead-lag** whose move comes first (lag of peak cross-correlation) · **common factor** the shared market move all four track · **W-bin** wall-clock aggregation window that synchronizes async contracts · **aliasing** grid finer than the data cadence → degenerate (our 0.5 s case) · **own vs full model** predict with own OFI only vs all contracts' OFI · **Hayashi–Yoshida** async-aware covariance/lead-lag estimator.
