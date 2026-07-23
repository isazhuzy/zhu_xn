# Experiment C — the look-back × forward window plane, for VOI / MPB / OIR

Separate report requested 2026-07-22. Extends the extreme-tail tables of
`NOTE_cumsum_教学.md` §五 (which fixed **both** windows: instantaneous factor,
10 s hold) to the full 2-D window plane, all 4 contracts, all 3 factors.

Code `window_exp_c.py` · report/figure `window_exp_c_report.py` ·
data `window_exp_c_all.csv` (2160 rows) · figure `fig_window_exp_c_all.png`.

---

## 0. THE CALCULATION (read this every time)

Per tick *t* we build a **factor** (look-back) and an **outcome** (forward),
then report the mean outcome of the factor's extreme tails, for every
(look-back J, forward k) pair.

**Factors** (per-tick, then aggregated over the look-back window):
- `voi` — Shen Volume-Order-Imbalance = ΔV_bid − ΔV_ask, signed lots — a **flow**.
- `oir` — (q_bid − q_ask)/(q_bid + q_ask) ∈ [−1,+1] — a book **level**.
- `mpb` — avg trade price this 500 ms − avg of last two mids, in ticks — trade-side **pressure**.

**Look-back J** (J ticks = J·0.5 s): `factor_J(t) = Σ_{i=t−J+1..t} factor(i)`, summed
**inside gid** (day × AM/PM), requires J full prior ticks.
> **Key calc fact:** we bucket by **quantile** (a pure *ranking*), and every kept
> tick sums exactly J terms, so rolling-**SUM** and rolling-**MEAN** give the
> **identical ranking → identical buckets → identical numbers**. "Sum vs mean" is
> irrelevant here. **J = 1 recovers the instantaneous factor** of the note.

**Outcome — forward k** (k ticks = k·0.5 s), **mid-to-mid, uniform for all three**:
`dy_k(t) = mid(t+k) − mid(t)` in ticks, `shift(−k)` inside gid, |dy|>100 dropped.
Mid basis is **mandatory for MPB** (the bid-ask bounce would otherwise cancel its
signal — the PRICE=mid lesson) and using it for all three makes them comparable.

**Extreme buckets** (nested tails): `neg0.1/1/5` = ticks with factor ≤ q(0.001/0.01/0.05)
= the most net-**selling**; `pos5/1/0.1` = factor ≥ q(0.95/0.99/0.999) = most net-**buying**.
Reported cell = `mean(dy_k)` over the bucket = **ticks/trade, gross, sub-spread**.
Per-trade means are **not** inflated by window overlap (one dy per tick), unlike the
cumsum y-axis.

**Decomposition used below** (drift control): for each (factor,code,J,k),
- `S = (pos0.1 − neg0.1)/2` = **drift-robust predictive signal**. `S>0` = momentum
  (buy→up, sell→down); `S<0` = reversal (buy→down, sell→up).
- `A = (pos0.1 + neg0.1)/2` = common component = sample drift + genuine buy/sell asymmetry.

**Drift is small.** Measured `A` at J=1 is <0.1 tick to 60 s and <0.4 tick even at
300 s → the net 2020-26 down-drift does **not** drive the patterns; the buy/sell
asymmetries are real microstructure.

---

## 1. Results — three factors, three different personalities

Grids read: **rows = seconds looked BACK, cols = seconds held FORWARD**, cell =
ticks/trade. `S` = drift-robust signal (+momentum / −reversal). Full grids per
contract in `window_exp_c_all.csv`; heatmaps in `fig_window_exp_c_all.png`.

### VOI (flow) — LOOK-BACK flips momentum → reversal (diagonal)
IC0000 `S`:
```
        fwd:0.5s   10s   30s   60s  120s  300s
back0.5s  1.83  3.05  2.75  2.30  1.59  0.88     momentum, peaks ~10s
   2.5s   0.92  1.72  1.37  0.62  0.10 -0.36
    10s   0.40  1.01  0.40 -0.47 -1.15 -1.23
    30s   0.22  0.51 -0.86 -2.10 -2.33 -1.96
    60s   0.12 -0.22 -1.80 -3.21 -2.82 -3.45     reversal
```
- Fresh flow (top-left) = **momentum**, peaks at ~10 s hold. Accumulated flow over
  30–60 s held for 1–5 min = **reversal** (S ≈ −3.5). The **look-back** is the knob.
- **Asymmetric & real:** on the raw SELL tail the reversal reaches **+7.2 ticks/trade**
  (IC, 30 s back / 300 s hold) — the largest number in the whole study; the BUY tail
  just fades. Heavy accumulated *selling* snaps back hard; heavy *buying* doesn't.
- Strong in IC/IM (small-cap), weak in **IH** (large-cap barely flips).

### MPB (trade side) — pure fast momentum, no reversal wing
IC0000 `S`:
```
        fwd:0.5s   10s   30s   60s  120s  300s
back0.5s  1.57  1.04  0.44  0.27  0.46  0.53     only the top row has signal
   2.5s   0.25 -0.09 -0.52 -0.51 -0.37 -0.11
    10s   0.11 -0.01 -0.29 -0.38 -0.28  0.18
   ...    ~0 everywhere else
```
- Signal lives **only** in the instantaneous / short-hold corner (~1–1.7 ticks),
  **decays to zero by ~10 s**, and develops **no** reversal wing. Look-back does not
  help (it destroys it). MPB is priced almost immediately (matches fig111 R²-at-0.5s).
- The scary blue long-horizon cells in the raw grid are mostly the `A`/asymmetry
  term, not predictive `S`. MPB is a one-tick momentum signal, full stop.

### OIR (book level) — FORWARD horizon flips it; momentum <1 s, then reversal
IM0000 `S`:
```
        fwd:0.5s   10s   30s   60s  120s  300s
back0.5s  0.83 -0.20 -0.66 -0.67 -1.12 -0.74
   2.5s   0.10 -1.09 -1.44 -1.35 -2.14 -1.75
    10s  -0.02 -0.99 -1.49 -1.87 -2.89 -2.90     reversal, strengthens with hold
```
- Momentum **only** at sub-second hold (top-left, S≈+0.8); flips to **reversal by
  ~10 s** and stays, deepening with the hold (IM S≈−2.9, IC≈−2.5). Here the
  **FORWARD horizon** does the flipping — it happens even at J=1; look-back barely
  matters (mild dampening).
- The book's static tilt predicts continuation for <1 s, then reliably mean-reverts:
  a heavily ask-tilted book (resting sell liquidity) precedes a *rise*, not a fall.
- Strongest reversal in **IM** then IC; the generalized, robust version of the note's
  one-line "IM extreme-ask-tail curl" curiosity.

---

## 2. Synthesis

Three factors that all "measure order-flow pressure" have **three different
window signatures**, and the window you pick determines the sign of your bet:

| factor | what it is | short-back / short-hold | how the sign flips | who flips it |
|---|---|---|---|---|
| **VOI** | flow (Δqueue) | momentum, peak ~10 s | → reversal at long back **and** hold | the **look-back** |
| **MPB** | trade side | momentum, ~1 tick, gone by 10 s | no flip — just decays to 0 | (nothing) |
| **OIR** | book level | momentum only <1 s | → reversal from ~10 s on | the **forward hold** |

**One-line takeaway:** the note's fixed "instantaneous factor, 10 s hold" window
sits on the **momentum peak for VOI and MPB**, but for **OIR the 10 s hold is already
in the reversal zone** — so reading all three at the same window silently mixes a
momentum signal with a reversal signal. VOI is the only one whose *look-back* turns
it contrarian; MPB has no contrarian regime at all; OIR is contrarian at essentially
every horizon beyond a second.

**Caveats:** all gross & sub-spread; 0.1 % tail ≈ ~25 events/side/day; long-horizon
cells use heavily overlapping windows (fewer independent samples than the tick count)
and 1–5 min holds carry real cost; drift is small (<0.4 tick @300 s) so it is not the
driver, but the raw (undecomposed) grids do contain a genuine buy/sell asymmetry `A`
on top of the predictive `S`.
