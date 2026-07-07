# end-of-day reversal after a >=1% intraday rally

**Question:** on a day where CFFEX index futures have risen >=1% from the open at
some point, is there a specific time before the 15:00 close after which the
market tends to reverse (give back the gain)?

## Data & method

Full-history 1-min mid bars (`(bid+ask)/2`, last tick per minute), whole session
09:30-15:00, IC/IF/IH/IM continuous contracts, 2015-2026, from DolphinDB
`dfs://hft_future_ts` (see `fetch_day_minutes_ddb.py`; ~2.16M rows).

For each contract, minute `t`, and day: `cumret(t) = close(t)/open - 1`,
`fwd(t) = close(15:00)/close(t) - 1`. Event-conditional test: take the sample
of days where `cumret(t) >= 1%` **at that specific minute** (day-clustered, one
observation per day — avoids pseudo-replication from autocorrelated within-day
minutes), and summarize `fwd(t)` (bp) across minutes `t` in 13:00-14:59.

`analyze_eod_reversal.py` produces the per-minute scan + a year-by-year check
on each contract's single strongest ("best t-stat") minute. `plot_eod_reversal.py`
(system python3, matplotlib) makes fig01/fig02.

## Result: no robust reversal signal

- **t-stats stay mostly inside [-2, +2]** across the whole afternoon for all
  four contracts (fig01, top panel) — i.e. never reaches conventional
  significance except brief, non-persistent dips.
- The single **best (most negative t-stat) minute per contract**, full history
  (n=96-390 days per contract): IC 14:42 (-34.4bp, t=-1.54), IF 14:39 (-4.8bp,
  t=-1.61), IH 13:00 (-43.1bp, t=-1.16), IM 13:37 (+2.5bp, t=+0.25 — i.e. no
  reversal at all, mild continuation). Win rates (fraction of days that
  actually reverse) sit at **~40-55%, i.e. close to a coin flip** — there is no
  asymmetric edge, only elevated variance.
- **Cross-year robustness kills it**: breaking the "best minute" down by year
  (fig02) shows the negative mean is **entirely carried by 2015** (IC -179bp,
  IH -196bp that year alone) — the 2015 crash/circuit-breaker period. Excluding
  2015, IC's "reversal" **flips to +2.1bp (continuation)** and IF shrinks to
  -1.0bp; both are noise-sized and inside the bid-ask spread.
- IH's t-stat trends *positive* (continuation, not reversal) through the last
  hour in most years (fig01 top panel, green line rising above 0 after 13:30).

**Conclusion:** there is no dependable time-of-day at which a >=1% intraday
rally reliably reverses into the close. The apparent "best minute" pattern
found on the pooled full-history sample is a **single-crisis-year artifact**
(2015), not a structural end-of-day phenomenon — the same overfitting/regime
trap documented in `ticker/README.md` and `intraminute/`
([[xn-ticker-open-reversal]], [[intraminute-momentum-reversal]]). Consistent
with those studies: no tradeable signal, and this one doesn't even clear the
"real but sub-spread" bar those did — it's statistically indistinguishable
from noise once the crisis year is removed.

## Filter variants tried (`filter_variants.py`)

The plain ">=1% from open" filter found nothing. Tried seven other ways to
slice the "up day" universe, each scanned per-minute 13:00-14:59 for the
strongest reversal, then immediately checked ex-2015 + year-by-year (the
lesson from the plain filter: never trust a pooled "best minute" until you've
looked year by year):

1. **Threshold sweep** (0.5%/1%/1.5%/2%/3% instead of 1%) — bigger rallies
   don't revert more cleanly; |t| stays ~1-1.6 throughout, still 2015-driven.
2. **At the fresh high** — up>=1% AND currently within 3bp of the day's
   running max (momentum still extending). IF0000 hit t=-2.55, but n=53 drops
   to n=28 ex-2015 and the per-year mean collapses to -3.3bp (noise).
3. **Faded off the high** — up>=1% AND already pulled back >=20bp from the
   running max (momentum stalling). Directionally the most negative raw bp of
   any filter (IC -127bp, IF -125bp) but ex-2015 IF/IH shrink to ~-7 to -9bp,
   IC alone stays large (-89bp) — single-contract, not corroborated.
4. **Fast spike** — up>=1%, and >=0.7% of that gain came in just the last 30
   minutes. Tiny samples (n=10-46), dominated by a handful of extreme days.
5. **Slow grind** — up>=1%, but <0.2% of it came in the last 30 min (stale
   gains). IF0000 t=-2.63 full-history, but ex-2015 weighted mean is only
   -3.4bp on n=98 — inside the spread, not tradeable.
6. **Market-wide** — all four contracts simultaneously up>=1% (broad rally,
   not idiosyncratic). Samples too small (n=20-32) to say anything; IH hit
   t=-2.63 but only n=8 ex-2015.
7. **Calm vs. choppy day** — day range < / >= 2.5%. This *looked* like the
   strongest lead by far (IF0000 t=-4.39, IH0000 t=-3.29, best minute right at
   the 13:00 lunch reopen, 8-9/9 years negative for IF/IH) — **but the "day
   range" as first computed uses the full 09:30-15:00 high/low, i.e. it uses
   afternoon data to decide whether the afternoon was calm: look-ahead bias.**
   Recomputed with a range that only uses information available *as of* the
   signal minute (running high/low up to `t`, not the whole day) and the
   effect mostly evaporates: t drops to ~-1.2 to -1.8, samples shrink
   (n=33-46), and recent years (IF0000 2020/2024/2025) turn *positive*.

**Bottom line:** every variant that looked interesting on pooled full-history
data either (a) is still carried almost entirely by 2015, or (b) depended on
a look-ahead-biased feature. None survives both an ex-2015 check and a
real-time-computable feature check. The original conclusion stands: no
dependable end-of-day reversal signal after a >=1% intraday rally, under any
of the eight filter designs tried so far. Full table:
`filter_variants_summary.csv`.

### 8th variant: 路程/位移比 (path-length / displacement) — `filter_path_ratio.py`

Real-time-computable "how efficient was the move so far" feature: `path(t) =
cumsum |step-to-step price change| from 09:30 to t`, `disp(t) = |price(t) -
open|`, `ratio(t) = path(t)/disp(t) >= 1` (1 = perfectly straight-line rally,
larger = more back-and-forth for the same net gain). Split the up>=1% sample
at the (time-local) median ratio into "smooth" (efficient trend) vs "noisy"
(choppy path) and rescanned:

- **Directionally consistent across contracts**: the noisy/choppy subsample
  shows a bigger, more significant reversal than the smooth subsample on
  every contract (IC t=-1.84 vs -0.22, IF t=-1.75 vs -0.81, IH t=-1.37 vs
  +0.32) — a choppy, low-conviction path to the gain does look more fragile
  than a clean trending one. This is the most *interpretable* result of any
  filter tried.
- **Still doesn't clear the bar**: ex-2015, IF's noisy-group edge vanishes
  (+0.4bp) and IH's flips positive (+7.9bp); only IC stays negative
  (-165bp) but off a thin, year-inconsistent n=31 (one bad 2024 driving it).
- **Continuous check** (corr(ratio(t), fwd(t)) within the up>=1% sample):
  strongly negative right at the 13:00 lunch reopen (-0.41 to -0.50 across
  IC/IF/IH — plausibly a reopen-specific microstructure effect, not a ratio
  effect) but decays to near-zero or flips positive by 14:30-14:59 (IF
  +0.30, IM +0.22 at 14:59). Not a stable relationship through the
  afternoon.

Best lead of everything tried so far, qualitatively sensible, but still not
a robust/tradeable signal once 2015 is excluded. Full table:
`filter_path_ratio_summary.csv`.

## The actual finding: NOT reversal — a robust closing-minutes CONTINUATION

Re-scanning at full minute resolution and zooming into 14:30-15:00
(`deepdive_close_continuation.py`, figs `fig_cc01`/`fig_cc02`) turned up
something different from what the question assumed: on up>=1%-from-open
days, the sign **flips from mild reversal to significant continuation**
right around 14:50-14:52, peaking sharply at **14:55**:

- Plain trigger (`cumret(t)>=1%`), fwd return t->15:00 close, at 14:55: **IF
  t=3.90, IH t=3.58, IM t=5.28** (win-rate for reversal only 28-40%, i.e.
  60-72% of days keep grinding *up* into the close). **IC0000 does not show
  this** — stays flat/mildly negative (t=-0.78) throughout, the outlier
  contract in this cross-section.
- **This is NOT a 2015 artifact — unlike every other filter in this study,
  2015 is one of the WEAKER years here.** IF: positive in 11/12 years
  (ex-2015 weighted mean +3.1bp). IH: positive in 11/12 years (ex-2015
  +5.3bp). IM: positive in all 5 available years (+5.5bp). This is the
  first result in the whole `xn/end` study that survives the ex-2015 check.
- **Window-length screen** (does "still rising in just the last w minutes",
  w=1..30, on top of the >=1%-from-open trigger, sharpen it further?): yes,
  materially. At 14:55: IM peaks at **t=6.17 (w=3min)**, IH at **t=4.47
  (w=10min)**, IF at **t=4.54 (w=2min)** — all comfortably above the plain
  trigger's t-stat. IC stays noisy/inconsistent across every window (no
  robust signal at any w).

**Precision update (`summary_verdict.py`, `plot_summary_verdict.py`,
`plot_continuation_by_year.py`):** recomputing every effect's t-stat as a
single **pooled** ex-2015 sample (rather than a year-weighted average of
per-year point estimates, which was an approximation used earlier) makes the
contrast sharper. See `fig_verdict_forest.png` — full-history t-stat (blue)
vs ex-2015 t-stat (red) for effects A (plain reversal hunt), B (path-ratio
noisy), C (calm-day, real-time-corrected), D (closing continuation), x4
contracts each:

- A and B: **collapse toward zero or flip sign** ex-2015 for IC/IF/IH (e.g.
  IC plain-reversal t: -1.54 -> +0.87; IF path-ratio-noisy t: -1.75 -> +0.06).
  Textbook artifact signature — the gray connector line swings hard across 0.
- C (calm-day, already real-time-corrected): stays mildly negative but
  **never reaches significance** either with or without 2015 (t stays
  between -1.1 and -1.8 throughout) — a weak, inconclusive shrug, not
  an artifact-then-vanish pattern nor a confirmed effect.
- D (closing continuation): the gray connector barely moves, or — for
  IC0000 specifically — **swings the other way**: full-history t=-0.78 (near
  zero) but ex-2015 t=+4.15 (highly significant continuation). **2015 was
  masking IC's continuation effect, not creating a fake one.**
  `fig_verdict_continuation_by_year.png` makes this precise: IC's 2015 bar is
  a lone -78bp outlier against 10/11 otherwise-positive years (2016-2026 all
  small and positive except a near-zero 2016/2017); IF is positive in 11/12
  years, IH in 11/12, IM in 5/5. This is the only effect in the whole study
  where excluding 2015 makes the case *stronger*, and it holds for all four
  contracts, not just three.

**Caveat — magnitude, not existence, is the open question.** The effect is
statistically the most solid one in this whole study (robust across years,
across window definitions, and for 3 of 4 contracts), but it's a ~10-minute
window: mean forward returns are only **+3 to +9bp**, likely inside or close
to the bid-ash spread (recall from `intraminute/`: real round-trip spread is
~0.5-1.0pt, i.e. roughly 5-10bp on these contracts) — net tradeability is
NOT established, only the raw directional/statistical existence of a
last-10-minutes melt-up on already-strong-up days for IF/IH/IM. Files:
`deepdive_close_continuation.py` -> `deepdive_zoom_1430_1500.csv`,
`deepdive_window_screen.csv`; `plot_close_continuation.py` -> figs.

## Profit-taking hypothesis: direct order-flow & open-interest tests

The path/displacement ratio was originally motivated by an intuition — "intraday
traders who bought earlier flatten once price has risen enough, and that
liquidation reverses the close" (获利了结). But path-ratio is a *confounded* proxy
(path smoothness ≈ volatility, not liquidation). So we tested the profit-taking
story **directly**, three ways — all reject it.

### Aggressor order flow (`fetch_day_minutes_flow_ddb.py`, `flow_analysis.py`, `flow_symmetric.py`)

Re-fetched full-history 1-min bars with per-minute aggressor volume
(`m_nActBidVolume`/`m_nActAskVolume`, summed per minute; Act semantics resolved
empirically — `actbid` is BUYER-initiated, corr(minute-ret, actbid−actask)=+0.357).
Feature: trailing-30-min **sell-pressure imbalance** `(sell−buy)/(sell+buy)`, the
direct footprint of longs hitting the bid to flatten. Real-time, no look-ahead.

- **Up≥1% days, split at median sell-pressure at 14:45**
  (`fig_flow_sellpressure_path.png`): high-sell-pressure days do **not** close
  weaker — for IC/IF/IM they close **stronger** (corr(sellpressure, fwd) = +0.08 /
  +0.08 / +0.02), only IH mildly negative. The per-minute scan `fig_flow_tstat.png`
  shows the effect never reaches significance in the hypothesized (negative)
  direction. Interpretation: aggressive selling into an up-day is **absorbed** by
  strong buyers → the melt-up continues (reproduces the 14:55 continuation).
- **Symmetric, both directions** (`fig_flow_symmetric.png`): up-days sell-pressure
  and down-days buy-pressure (counter-move flow) vs signed reversal — mean corr ≈ 0
  on both sides; down-days even mildly positive (bargain-buyers into a drop are also
  absorbed → keeps falling). Same absorption mechanism, symmetric.

### Continuous sweeps: sell-pressure vs P(price up) (`flow_prob_sweep.py`, `flow_prob_sweep_eod.py`)

Instead of the median hi/lo split, sweep the sell-pressure imbalance across its full
range (pooled, all days) and read off P(up). **The sign is horizon-dependent:**

- Fixed horizons (`fig_flow_prob_sweep.png`): **next 1–5 min the curve rises**
  (more sell-pressure → higher P(up): short-term absorption/mean-reversion);
  **next 30 min it falls** (sustained selling → down-momentum). Crossover ~5–30 min.
- To-close / EOD context (`fig_flow_prob_sweep_eod.png`): net-selling → modestly
  **lower** P(close > now), diff only −2 to −4bp (IM the lone flat/positive
  exception). A weak down-**momentum**, not reversal. Magnitudes are tiny
  (P(up) ~0.44–0.56 only at moderate imbalance; ≈0.50 in the bulk) and these are
  minute-pooled (autocorrelated) probabilities — shape is trustworthy, exact
  significance is not.

### Real open-interest confirmation (`fetch_oi_minutes_ddb.py`, `oi_analysis.py`)

Aggressor selling can be *absorbed* (churn) with no net position change; genuine
**position closing shows up as open interest falling**. Minute OI from
`dfs://hft_future_realtime/RealtimeMinKLine` (`position` col; single-contract months
so dominant contract per day = max EOD `totalVolume`; only **2025-01→2026-07** usable
— 2024-H2 partitions throw persistent "too many open files"; afternoon session starts
13:01 not 13:00). Two findings (`fig_oi_confirm.png`):

- **The premise is false: OI GROWS into the close, not shrinks.** 13:01→15:00 OI
  change is **+6 to +8% on up-days, +6 to +9.5% on down-days, +4.6 to +7.4% all
  days**, every contract. These index futures are institutional hedging/overnight
  vehicles; the last hour is net **position-building**, so "day-traders flatten by
  the close" simply doesn't hold here.
- Even where recent OI does decline (trailing-30min, genuine closing), it does **not**
  predict reversal: corr(liquidation, signed fwd) ≈ 0 to slightly **positive**
  (up: +.05/+.02/+.08/−.00, down: +.11/+.09/+.06/+.09 for IC/IF/IH/IM), t<−2 minutes
  at noise level.

**Verdict:** the 获利了结→反转 hypothesis leaves no footprint in aggressor flow,
its symmetric analogue, the probability sweeps, or the direct OI measure. Root cause:
net accumulation into the close + strong-buyer absorption → **continuation (melt-up),
not liquidation reversal**. The intuition may hold for retail intraday equities; it
does not for CFFEX index futures. (path-ratio direction abandoned; the filter-free
sweep `filter_free_pathratio.py` is also weak — dropping the up-filter turns the ratio
into ≈1/|displacement|.)

## Files

- `fetch_day_minutes_ddb.py` — full-history whole-session 1-min bar fetch (run
  with DDB, `dangerouslyDisableSandbox`).
- `day_minutes_full.csv` — raw fetch output (code, day, minute, close).
- `analyze_eod_reversal.py` — per-minute scan + year breakdown; writes
  `scan_by_minute.csv`, `scan_by_year.csv`.
- `plot_eod_reversal.py` — figs (system python3, not `.venv`).
- `common.py` — shared panel-building helpers (cumret/fwd/running-max/rolling-
  change/day-range) used by both `analyze_eod_reversal.py` and
  `filter_variants.py`.
- `filter_variants.py` — the eight alternative filters above; writes
  `filter_variants_summary.csv`.
- `filter_path_ratio.py` — 路程/位移比 (path-length/displacement) split +
  continuous correlation check; writes `filter_path_ratio_summary.csv`.
- `filter_free_pathratio.py` — path-ratio reversal test with the up≥1% filter
  removed (symmetric ±move floor, signed fwd); writes `ff_pathratio_summary.csv`,
  `figs/fig_ff_pathratio_tstat.png`.
- `fetch_day_minutes_flow_ddb.py` — full-history 1-min bars + per-minute aggressor
  volume; writes `day_minutes_flow_full.csv`.
- `flow_analysis.py` / `flow_plot.py` / `flow_tstat_plot.py` — sell-pressure
  profit-taking test on up≥1% days; `flow_summary.csv`,
  `figs/fig_flow_sellpressure_path.png`, `figs/fig_flow_tstat.png`.
- `flow_symmetric.py` — same test for both up (sell-pressure) and down (buy-pressure)
  days; `flow_symmetric_summary.csv`, `figs/fig_flow_symmetric.png`.
- `flow_prob_sweep.py` / `flow_prob_sweep_eod.py` — sweep sell-pressure vs P(price up)
  at fixed horizons / to-close; `flow_prob_sweep*.csv`, `figs/fig_flow_prob_sweep*.png`.
- `fetch_oi_minutes_ddb.py` — minute open-interest (2025-01→2026-07, RealtimeMinKLine);
  writes `oi_minutes.csv`.
- `oi_analysis.py` — real-OI confirmation of the profit-taking test (both directions);
  `oi_summary.csv`, `figs/fig_oi_confirm.png`.
