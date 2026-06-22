# Intra-Minute Momentum / Reversal in China Index Futures — Full Write-Up

**Universe:** continuous contracts IC (中证500), IF (沪深300), IH (上证50), IM (中证1000)
**Data:** DolphinDB `hft_future_ts`, 500 ms snapshot mid-prices, **2020-01 → 2026-05 (76 months)**
**Sessions:** AM 09:30–11:30, PM 13:00–15:00 (≈240 minutes/day)

**Headline:** A genuine, cross-month-stable, out-of-sample-robust intra-minute structure exists (mean-reversion against the prior minute, plus weak end-of-session momentum). **But its magnitude (0.05–0.3 index points) is universally 2–5× smaller than the measured bid-ask spread (0.5–1.0 points), so nothing is tradeable net of cost.** This is shown by exhaustive scan, not by sampling. *Real physics, no arbitrage.*

---

## 1. Objective
Determine whether a fixed, simple intra-minute timing rule produces a tradeable return in liquid China index futures, and if not, characterize *why* with rigor (cross-month stability, out-of-sample discipline, and real transaction costs).

## 2. The Strategy (fixed throughout)
Every minute *M*, take a position in the direction of the **previous** minute's move and hold it for the whole minute:
- Signal: `d_M = sign( close(M−1) − close(M−2) ) ∈ {−1, +1}`
- P&L: `pnl_M = d_M × ( close(M) − open(M) )`

`pnl > 0` = **continuation** (momentum wins); `pnl < 0` = **reversal** (the move fades the prior direction). A minute is used only if M−1, M−2 are consecutive and it has ≥2 ticks.

## 3. Core Definitions (equations)
For minute M with tick mids `P_1 … P_L` (L ≈ 90–120):
- Open = `P_1`, Close = `P_L`, within-minute displacement `X_k = P_k − P_1` (so `X_1 = 0`).
- **Per-minute P&L** (raw index points): `pnl_M = d_M (P_L − P_1)`.
- **Curve per minute-of-day**: align each day's `X_k` by *real tick index* (no time-normalization), average across days → one curve.
- **Cross-month correlation**: Pearson of two months' curves along the tick axis.
- **Variance ratio** (1st-vs-2nd-moment cross-check): `VR(k) = Var(X_k) / [(k−1)·Var(X_2)]`; `<1` mean-reversion, `=1` random walk, `>1` momentum.
- **Tradeability**: `net per trade = E[pnl_M] − cost`, where `cost ≈ E[ask − bid]` (full spread per round trip, market orders).

## 4. Worked Example — IC, 2023-03-01, minute 09:37
| Quantity | Computation | Value |
|---|---|---|
| Signal | `sign(close 09:36 − close 09:35) = sign(6342.40 − 6336.70)` | `d = +1` (long) |
| Open / Close | `P_1 = 6342.80`, `P_L = 6337.20` | — |
| **P&L** | `+1 × (6337.20 − 6342.80)` | **−5.60 pts (reversal)** |

We went long (prior minute was up), but price fell 5.6 pts below the open → momentum was punished. Averaging this quantity over many days at 09:37 produces the "dip-then-recover" curve.

---

## 5. Finding 1 — The directional structure (and why framing matters)
- **Buy-and-hold** `(P_L − P_1)` (always long, ignore the signal): average ≈ 0, cross-month correlation ≈ 0 → looks like pure noise.
- **Momentum** `d·(P_L − P_1)`: the averaged curve **dips to −0.15…−0.25 pts over the first ~1/3 of the minute (trough ~tick 40) then partially recovers** = a short-horizon **mean-reversion against the prior minute's direction**.
- **Why the signal is essential:** the reversal is *conditional* on prior direction (after an up-minute price tends to fall; after a down-minute, rise). Buy-and-hold averages these opposite cases to ≈0; multiplying by `d` aligns them so the effect survives averaging. Correlation jumps from ≈0 to **+0.44–0.65 (raw tick)**.
- **Corroboration:** variance ratio `VR ≈ 0.7 < 1` — an independent (second-moment) lens agrees the market is mildly mean-reverting intra-minute.

## 6. Finding 2 — Cross-month stability is real, not a statistical artifact
High Pearson correlation between cumulative curves can be spurious (two smooth integrated series correlate even when independent). We tested it:
- **Increment (differenced) correlation** across months stays **+0.15–0.19** (null ≈ 0, p95 ≤ 0.005) → the *tick-by-tick* shape is genuinely shared, immune to the cumulative-inflation trap.
- ~**95% of months** show the same "先降再升" (dip-then-recover) shape.

## 7. Finding 3 — Time-of-day structure (survives 76-month + OOS validation)
Aggregating per-minute P&L into 15-min windows reveals a stable map (IS = 6 calm months, OOS = 6 fresh calm months, then all 76):
- **Reversal** (momentum loses): mid-morning **10:00–11:00**, **13:15–13:30**, **14:15**.
- **Continuation** (momentum wins): pre-lunch **11:15–11:30**, post-lunch **13:00**, into-close **14:45–15:00**.
- Direction holds across all 76 months, but only as a **~60–75% monthly tilt** (it was ~100% in the calm-month subsample) — a real but moderate tendency.

---

## 8. The Tradeability Investigation (the heart of the project)

| Test | Method | Result |
|---|---|---|
| Every minute, all day | trade all ~230 session minutes | **Negative even gross** (IC −0.062, IF −0.057, IH −0.037, IM −0.168 pts/trade) → the session is net mean-reverting; momentum bleeds before costs (fig40) |
| Best-minute selection | pick IS-significant minutes | IS↔OOS correlation ≈ 0; t = −6 minutes **flip sign** OOS → overfitting (fig32) |
| Volatility conditioning | bucket by prior-move size | shape **fails OOS** (t < 1, fig33) |
| 15-min screening | aggregate to windows | reveals the real time-of-day structure ✓ (fig34–36) |
| Hold-as-one-block | enter 11:21, exit 11:30 | prior direction **doesn't predict** the 9-min block (t < 1) → no |
| Best lead, deep-dive | 11:21–11:29 per-minute momentum | gross **+0.22 (IC) / +0.28 (IM)** pts/trade, 74–76% of months positive, equity rises at an *assumed* 0.1pt cost ✓ |
| **Make it real** | **measure the actual spread** | IC/IM ≈ **1.0 pt**, IF/IH ≈ 0.5 pt = **2–5× the edge** → net −0.4…−0.8/trade; 76-mo equity collapses to ≈ **−2,000,000 元/lot** (fig38) |
| Exhaustive scan | all 16 windows × 4 contracts vs each window's real spread | **65/65 below breakeven; zero tradeable** (fig39) |
| Session edges | AM open / lunch close / PM open / day close, refined to precise minutes | edge and spread **scale together** — AM open has the biggest reversal (~0.5) *and* the widest spread (~1.07); all 16 cells net-negative (fig41, fig42) |
| Open momentum, isolated | surgically keep only the momentum minutes at 09:38–09:39/09:44 | best net −0.44 (IF) to −0.73 (IM); the open is the **widest-spread** time, so its big reversal still can't pay → worst window of all (fig43) |
| Per-contract potential ranking | rank every window by `\|edge\| − spread` per contract | potential concentrates in the **afternoon (14:00–14:45)**; **IM 14:15 fade is the single closest to breakeven (net −0.03, 83% of months same-sign)** but still negative; IH has no potential (fig44) |
| Pure directional bet | "just short 14:15→14:30" every day, no signal | **hit rate ≈ 48–50% (coin flip)**, block drift ≈ 0 (slightly *up*), net −0.55…−2.23 → confirms **volatility ≠ edge**; without a direction forecast a directional bet = coin flip minus spread |
| **Tick-level fade** | enter at open, exit at a fixed within-minute tick (sweep 20/42/60/90/close) | exiting at the reversal trough (**tick ~42, ≈20 s in**) captures **1.5–2.4× more gross** than exit-at-close (IC +0.147 vs +0.062) — but best gross (0.07–0.25) is still **4–7× below the spread** → every exit tick net-negative (fig45) |

### The decisive arithmetic
`IC 11:21–11:29: gross +0.225 − spread 1.023 = net −0.798 pts/trade.`

### The structural reason (the key insight)
A one-minute price move is small (edge 0.05–0.3 pts), but every trade pays the spread once (0.5–1.0 pts) → **the edge is structurally an order of magnitude below the cost.** Two reinforcing facts make this inescapable:
1. **Edge and spread scale together.** The *high-edge* times (the open, ~0.5 reversal) are exactly the *wide-spread* times (~1.07) — they are bound together, so there is **no "high-edge + low-cost" sweet spot** anywhere in the session (fig41, fig44).
2. **Refining the trade doesn't break the wall.** Every granularity was tried — minute close-open, 15-min blocks, per-minute, and finally **tick-level optimal exit (fig45)**. Each finer cut extracts a bit more gross (your timing instinct is correct), but the best of all (tick-42 exit, 0.07–0.25 pt) is still 4–7× below the spread. The reversal is real at every scale; it is simply smaller than the cost of touching the market once.

The only way "buy at tick X, sell at tick Y" turns positive is to **post limit orders (earn the spread, market-making)** instead of crossing it — an execution/liquidity-provision question outside this dataset.

---

## 9. Conclusions
1. **Genuine microstructure exists:** conditional mean-reversion against the prior minute (corroborated by VR < 1), plus weak end-of-session momentum. It is cross-month stable, OOS-robust, and confirmed by the differenced-correlation test.
2. **None of it is tradeable:** every directional edge is buried inside the bid-ask spread. This is an *exhaustive* result — across all windows (65/65 fail at the measured spread), both directions (momentum and fade), all four contracts, and **every granularity down to the tick-level optimal exit**.
3. **A pure directional bet has no edge:** "just short" a window is a coin flip (≈50% hit, ≈0 drift) minus the spread → guaranteed loss. Volatility is variance, not expected return; you cannot convert movement into profit without a direction forecast (which the data shows does not exist net of cost).
4. **The whole "find a tradeable intra-minute edge" line is closed in the data layer.** 有物理，无套利.
5. **The only unclosed door is not in the data:** earning the spread via passive / market-making (limit-order) fills instead of paying it — an execution / liquidity-provision capability, outside this dataset.

## 10. Methodology lessons (transferable)
- You must multiply by the signal `d` to reveal a *conditional* effect; unconditional averaging hides it.
- Cross-series correlation of cumulative curves is inflated — always confirm with **increments** + a shuffle null.
- Tradeability must use the **measured spread**, never an assumed cost: 0.1 vs 1.0 is the difference between "profitable" and "−2M".
- Every "looks profitable in-sample" candidate must clear **strict out-of-sample** selection; the strongest IS minutes (t = −6) flipped sign OOS.
- Beware regime mixing: never pool variance across calm and crisis months (it once inflated a variance ratio to 4); normalize per-month, then take the median.

## 11. Figures & files
**Figures** (`figs/`):
- fig21–23 — 76-month momentum curves + net-drift over time
- fig24, fig26 — cross-month correlation; all-history average curve
- fig34–36 — 15-min time-of-day screening (12-mo & 76-mo) + per-minute precision
- fig37–39 — 11:21 deep-dive; spread reality (equity → −2M); exhaustive 65-window scan
- fig40 — whole-session per-minute momentum return scan (momentum ↓ / fade ↑ mirror)
- fig41 — four session-edge windows: |edge| vs real spread
- fig42 — per-minute momentum, 4 contracts shown separately
- fig43 — open momentum vs real per-minute spread
- fig44 — per-contract ranking of the most-promising windows (report version)
- fig45 — tick-level fade backtest: gross vs exit-tick vs spread

**Key scripts:** `minute_curves_raw_multi_ddb.py` (76-mo raw-tick curves), `minuteofday_allmonths_ddb.py` (per-minute P&L), `focus_1121_ddb.py` (lunch-close deep-dive), `focus_spread_ddb.py` / `spread_allwin_ddb.py` / `open_spread_perminute_ddb.py` (measured spreads), `focus_1415_short_ddb.py` (pure-short test), `tick_fade_ddb.py` (tick-level fade), plot scripts `*_plot.py`.

**Companion:** `WRITEUP_中文.md` (full Chinese version).

*Deleted along the way (per user direction): the time-normalized (interp-to-120) branch and the pure buy-and-hold branch.*
