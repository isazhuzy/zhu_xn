# future/ — IM tick "follow-through" study

> **Living document.** We update this as the study grows. Status: **pass 1 complete**
> (price-pulse + volume-peak follow-through on IM, full history).

This folder asks one question on **IM0000** (中证1000 stock-index futures, the most
volatile of the four CFFEX contracts):

> When something *sudden* happens in the tape — a sharp price move, or a burst of
> volume — **does it lead to more of the same (a trend), or does it snap back
> (reversal)?**

We test this at the **tick** level (one row per 500 ms market snapshot, ~120/minute).

---

## Quant 101 — the ideas behind this

- **Tick.** The exchange publishes a snapshot ~every 500 ms: best bid/ask, last
  price, volume traded since the previous snapshot. ~120 ticks per trading minute.
- **Mid price.** `mid = (bid + ask) / 2`. We measure moves in mid, not last-trade,
  so a single print bouncing between bid and ask doesn't masquerade as a "move."
- **Momentum vs mean-reversion.** Two opposite hypotheses about short-horizon prices:
  - *Momentum / follow-through / trend:* a move up tends to be followed by more up.
  - *Mean-reversion:* a move up tends to be followed by a pull-back down.
  Which one wins depends on the **horizon** and the **size** of the trigger — finding
  *where* the line sits is the whole game.
- **Signing the forward return.** A raw forward return averages to ~0 (up and down
  cancel). The trick: multiply the forward return by the **direction of the trigger**.
  `signed = dir × (price_later − price_now)`.
  - `signed > 0` → price kept going the trigger's way → **trend / follow-through**.
  - `signed < 0` → price reversed the trigger → **mean-reversion**.
  This aligns up-triggers and down-triggers so they reinforce instead of cancel.
- **Why a t-stat, not just a mean.** With millions of observations even a microscopic
  mean can be "real." The **t-stat** = mean / standard-error tells you how many
  standard errors the mean sits from zero. `|t| > ~2` ≈ statistically significant;
  here we see `t` of 10–20, so the *direction* is not luck. But significant ≠ tradeable
  (next point).
- **Gross vs net — the spread caveat (the thing that kills most "edges").** Every
  round trip pays the **bid-ask spread**. For IM that is **~1.0 index point**. Our
  edges below are **0.02–0.06 pts** — i.e. *inside* the spread. So they are **real
  structure** (the market genuinely behaves this way) but **not money** once you pay
  to trade. We always quote results **gross** and flag this.
- **Don't let windows cross gaps.** A lookback/forward window must not span the lunch
  break (11:30–13:00) or an overnight gap, or you'd "predict" across a halt. We compute
  everything inside contiguous **(day, session)** blocks (AM 09:30–11:30, PM 13:00–15:00).

---

## EXP1 — Price pulse: does a move lead to a trend?

**Construction.** At each tick `i`:
1. **Lookback** `n` ticks: `back = mid[i] − mid[i−n]` (the "pulse").
2. **Trigger** only if `|back|` exceeds a threshold `k` (in index points). `dir = sign(back)`.
3. **Forward** `x` ticks: `signed = dir × (mid[i+x] − mid[i])`.
4. Average `signed` over all of history, per `(n, k, x)`.

**The three knobs (each varied while the other two are fixed — see `fig_price_pulse.png`):**
- `n` = how far back we measure the pulse (5, 10, 20, 40 ticks)
- `k` = how big the pulse must be to count (0, 0.2, 0.4, 0.8, 1.6 pts; `k=0` = every tick)
- `x` = how far forward we look (5, 10, 20, 40, 80, 120 ticks)

**What we found (full history, ~22M tick-events):**
- **Fast, small moves TREND.** Short lookback (`n=5`) over the next ~10 ticks →
  `+0.043 pts, t≈18`. Raising `k` up to ~0.4 keeps/strengthens it: a clean quick
  push continues.
- **Slow or huge moves REVERSE.** Long lookback (`n=40`) at long horizon (`x=80–120`)
  → `−0.03 to −0.06 pts, t≈−8`. An outsized pulse (`k=1.6`) flips negative too
  (`−0.017, t=−2.4`).
- **One picture:** `fig_price_heatmap.png` — solid **red** (trend) in the top row
  (small `n`), grading to **blue** (reversal) bottom-right (large `n`, long `x`).
- **Interpretation:** IM has a **fast-momentum / slow-reversion** structure. Quick
  order-flow imbalance carries for a few seconds; anything large or drawn-out is
  liquidity being *taken* and then *given back*.

---

## EXP2 — Volume peak: what happens after a volume burst?

**Construction.** Compare a **short** trailing volume window to a **long** one:
`spike = (V_t1 / t1) / (V_t2 / t2)` = recent per-tick volume ÷ its longer baseline
(`t2 > t1`). A **peak** is `spike > r`. After a peak we measure, over the next `x` ticks:
- **|forward move|** — does a peak precede *volatility* (a bigger move, either way)?
- **signed forward move** (signed by the short-window price direction) — continuation
  or reversal *after* the burst?
- **forward volume/tick** — does the burst *persist* (volume clusters) or die out?

`r="ALL"` is the every-tick baseline so peak rows read *relative* to normal.

**The knobs (change `t1`/`t2` one at a time — see `fig_volume.png`):**
- `(t1, t2)` pairs: (5,30) (5,60) (5,120) (10,60) (10,120) (20,120)
- `r` = spike threshold (1.5, 2.0, 3.0); `x` = forward horizon (10, 20, 60 ticks)

**What we found:**
- **Volume → volatility (strong, monotone).** The bigger the peak, the bigger the
  forward move. Extreme spikes (`t1=20,t2=120,r=3.0,x=60`): `|move| = 7.1 pts vs 3.0
  baseline` (~2.4×). A volume burst reliably *front-runs* a big move.
- **Big sustained peaks = EXHAUSTION.** Signed forward move is mildly positive at
  baseline (+0.04) but flips **negative** as the peak grows (`−2.14, t=−5.3` at the
  extreme): a large surge tends to *spend* the move and reverse it.
- **Volume clusters** when the spike is measured over a *longer* short window
  (`t1=10–20`): forward volume stays high (4–5 vs 3.3). A 5-tick blip is transient.

---

## Bottom line so far

1. IM short-horizon prices are **fast-momentum, slow-reversion** — and the crossover
   in `(n, k, x)` is clean and highly significant.
2. **Volume peaks predict volatility**, and **large peaks mark exhaustion** (reversal).
3. **All gross.** Every edge here is **≤ 0.06 pts**, inside IM's ~1.0 pt spread →
   confirmed *structure*, **not a net-tradeable signal** on its own. Next passes
   should look for places the structure is large enough, or combinable, to clear costs.

---

## Files

| file | what |
|---|---|
| `im_followthrough_ddb.py` | fetch IM ticks from DolphinDB + compute both experiments → CSVs. Run on `../.venv` with the sandbox OFF. |
| `im_followthrough_plot.py` | build the figures from the CSVs. Run on system `python3` (has matplotlib). |
| `price_trend.csv` | EXP1 accumulators: `code,n,k,h, s_sum,s_ss,s_n,hits` (mean = `s_sum/s_n`). |
| `volume_peak.csv` | EXP2 accumulators: `code,t1,t2,r,h,n_peak, a_*`(abs move) `g_*`(signed) `v_*`(fwd vol). |
| `figs/fig_price_pulse.png` | EXP1: vary `n` / `k` / `x`, each fixing the other two. |
| `figs/fig_price_heatmap.png` | EXP1: `n × x` grid — the trend↔reversal regime map. |
| `figs/fig_volume.png` | EXP2: change `t1` / `t2` one at a time, lines = peak strength. |
| `im_followthrough.log` | run log of the full-history sweep. |
| `ddb_config.py` | local DolphinDB credentials (gitignored). |

**CSVs store sum / sum-of-squares / count** (not means) so any subset can be
re-aggregated and given a t-stat downstream:
`mean = sum/n`, `se = sqrt(ss/n − mean²) / sqrt(n)`, `t = mean/se`.

## Data notes
- Table `dfs://hft_future_ts` / `TickPartitioned`, IM exists from **2022-07**.
- Per-tick volume column is `m_iVolume` (incremental, not accumulated).
- 2024-02 has duplicate timestamps (we `drop_duplicates("ts")`); 2023-07 is a data gap.
- EXP2 drops each session's first tick (opening-auction volume burst) via `i ≥ t2`.
