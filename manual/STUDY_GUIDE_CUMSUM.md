# Study Guide — The Sorted-Cumsum Diagnostic (VOI & OIR), plus the Lattice-Binning Lesson

Hand-built in `xn/manual` (2026-07). Code: `voi_cumsum.py` (compute, factor-agnostic) /
`plot.py` (`WHAT=voi`, integrated with the momentum figure). Side quest: the fig111 MPB
sawtooth fix in `xn/prediction/mpb_return_ddb.py` — full writeup in `NOTE_lattice_binning.md`.

---

## 0. The one-sentence idea
> **Sort every tick by its factor value, smallest first, then walk the sorted list adding
> up each tick's future price change — if the factor predicts, the running total traces a
> check-mark (对勾): down through the sell-signals, bottoming where the factor flips sign,
> up through the buy-signals. No regression, no bins, no assumptions.**

This is the non-parametric cousin of the regressions in `xn/prediction`: where OLS gives
you one slope, the sorted cumsum shows the *entire shape* of the factor→return relation —
monotonicity, saturation, the natural threshold, and tail behavior, all in one curve.

---

## 1. The ingredients (per tick t)

**Factor** (choose with `FACTOR=` env):
- `voi` — Shen (2015) Volume Order Imbalance, the *flow*: ΔV_bid − ΔV_ask with the
  down/same/up rule per side (see `STUDY_GUIDE_VOI.md` §1.1). Integer lots, always.
- `oir` — Order Imbalance Ratio, the *level*: (qb−qa)/(qb+qa) ∈ [−1, +1]. Needs no
  previous tick. QI of paper #2 rescaled: QI = (OIR+1)/2.

**Outcome**:
```
dy_k(t) = P(t+k) − P(t)        in ticks; P = m_nPrice (last trade; PRICE=mid for midquote)
```
Point-to-point, k snapshots (500ms each) ahead, `shift(-k)` **inside gid** so the last k
ticks of each session are dropped, never bridged across lunch/overnight. |dy|>100 = bad
print, dropped. NB: this is NOT Shen's regression target (he uses the *average* mid over
t+1..t+k, ≈ half the endpoint magnitude for a drifting price) — don't compare magnitudes
across the two.

## 2. The mechanics (the part to internalize)

```python
order = np.argsort(voi, kind="stable")   # ranks: indices that would sort the factor
cum   = np.cumsum(dy[order])             # running total in factor order
q     = rank / n                         # percentile = empirical CDF, just counting
```

- **argsort** returns indices, not sorted values — one `order` array consistently reorders
  *every* column. Sort once, apply everywhere.
- **cumsum** = running total: `cum[i]` = "gross ticks from buying at each of the i
  smallest-factor ticks and closing k snapshots later".
- **Percentile** = rank/n. Nothing fancier. Ties (integer VOI!) form contiguous rank
  blocks — a factor *value* maps to a percentile *interval* (VOI=0 owns ~q 0.40–0.60);
  `kind="stable"` keeps chronology inside a block (reproducibility; the block's total is
  order-invariant anyway).
- **Downsampling**: cumsum over all n rows, but store only NPTS=4000 evenly spaced ranks
  (`np.linspace`) — visually identical, 10,000× smaller CSV.
- Memory: code-outer/month-inner loop (sort needs all of one contract; never two at once),
  keep only needed columns, `float32` (exact for integers < 2²⁴).

## 3. How to read the figure (two glances)

1. **Depth of the minimum vs the endpoint = signal.** The valley is *supposed* to be deep
   and negative — we front-loaded the sell-signals. Depth ÷ n = gross ticks per trade.
2. **Endpoint vs zero = sample drift × ~k, NOT signal.** The 100% point = Σ all dy — no
   sorting changes it. (IM 2024 ends −184K: the index fell and overlap counts each move
   ~k times.) To kill drift: demean dy before cumsumming — endpoint pinned to 0.
3. The **minimum's x-position** = where the factor's effect flips sign = the natural
   trading threshold.

**Caveat that goes in every writeup**: adjacent ticks' k-windows overlap (share k−1
snapshots), so the y-axis magnitude is inflated ~k×; the shape and per-tick depth are the
honest statistics. And everything is gross — sub-spread, as always in this project.

**Percentile axis vs raw axis** (`XRAW=1`): percentile answers "how does the effect
accumulate over the *population*" (uniform by construction, hides the tails' rarity);
raw value answers "at which *factor values* does the action live" (needs symlog for VOI —
98% of ticks sit in ±10 lots, extremes reach ±1000; linear would collapse everything into
a stripe at 0). The staircase in raw view = VOI's integer lattice, undistorted. Use both.

## 4. Results (full sample 2020-01…2026-05; pilot = calendar 2024)

**VOI** (`fig_voi_cumsum.png`, K = 1/20/120 in the archived run):
- Clean deep **V** on all four contracts, pilot AND full — the relation is monotone across
  the whole distribution, not tail-driven. Confirms fig81 non-parametrically.
- Minimum always at **VOI ≈ 0** (q 0.535–0.60): threshold = sign(VOI), nothing fancier.
- Per-tick depth (min÷n, k=20): **IM .46, IC .43, IF .24, IH .17** ticks — small-caps
  carry ~2× the per-tick signal (rhymes with the cross-impact IC↔IM finding).
- Horizons: k=1 captures only ~40% of the depth; **k=20 ≈ k=120** — impact fully realized
  by ~10s, slight retrace after (IC).

**OIR** (`fig_oir_cumsum.png`, K = 1/5/10/20):
- Rounded **U** (continuous factor → no tie shelf), min also at ≈0 but fuzzier.
- **~5× weaker than VOI**: IC k=20 depth .073 vs .43 tick/tick. The book's *tilt* informs;
  its *change* is the real signal. Exactly why Shen uses OIR as a supporting regressor.
- Saturates by **~5s** (k=10 ≈ k=20). Pilot-2024's "sell-side stronger" asymmetry was
  drift, not structure — full-sample endpoints ≈ 0 relative to depth (IC even +80K).
- Curiosity that survived the full sample: **IM extreme-ask-tail curl** — the ~2% most
  ask-crushed books precede small *rebounds* at 10s. Exhaustion/limit-move microstructure;
  remember it, don't trade it.

**MPB** (`xn/last/fig_mpb_cumsum[_raw].png`, K = 1/5/10/20, **PRICE=mid — mandatory, see
below**; compute FACTOR=mpb, plot `xn/last/mpb_cumsum_plot.py`):
- Deep symmetric **V**, min at MPB ≈ 0 (q 0.49–0.52). Per-tick depth (k=20):
  **IM .34, IC .29, IF .16, IH .11** — between VOI and OIR, ~⅔ of VOI. Factor ranking:
  **flow (VOI) > trade-side (MPB) > level (OIR)**, ordering IM>IC>IF>IH as in fig111.
- Faster than VOI: k=1 already captures ~⅔ of the k=20 depth (VOI: ~40%) — trade-side
  info is priced almost immediately (fig111's R² peak at 0.5s, non-parametrically).
- Raw view: quarter-tick staircase cliffs, biggest at ±0.5 (trades pinned at bid/ask,
  1-tick spread); fat outlier tails to ±60 ticks ⇒ raw axis needs symlog(linthresh=1).
- Sanity identity confirmed: endpoints match VOI/OIR runs to ~±3K (IC +77K, IF −186K) —
  the endpoint is factor-independent drift, exactly as §3 claims.

**The MPB bounce bug (the most important lesson of the series).** First run used the
default `PRICE=last` → **no check-mark at all** (min at q=1.0; archived:
`fig_mpb_cumsum_pilot_lastprice.png`). MPB *is* the last trade's position vs the mid, so
measuring dy from the last price starts every positive-MPB measurement half a spread
above fair value — the mechanical bid-ask bounce is a reversal of the same size as the
continuation signal, and they cancel ~exactly (both = "where in the spread do prints
sit"). Switch to `PRICE=mid` and the V appears.
> **Rule: the outcome variable must be measurically independent of the factor.**
> Trade-based factor → quote-based outcome. This is why Shen's y is in mids and why
> voi_backtest separates mid-to-mid "signal value" from taker P&L. VOI/OIR survived
> PRICE=last only because they're quote-built: the bounce added noise, not correlated
> bias.

## 5. The lattice-binning lesson (fig111 MPB sawtooth)

The other thing this session taught (full slow version: `NOTE_lattice_binning.md`):

> **Before binning any price-derived variable, ask "what values can it actually take?"
> and put the bin centers there.**

MPB = trade price − avg of two mids → lives on a **quarter-tick lattice**. BW=0.1 bins
misaligned with it, plus `np.round`'s half-to-EVEN tie-breaking (1.25→1.2 but 1.75→1.8),
interleaved giant single-price bins (low mean-y) with tiny multi-price-sweep bins (~2×
mean-y) → regular sawtooth. Fix: `BW=0.25` + `np.floor(x/BW+0.5)*BW` (half-up). Result:
smooth S-curve, saturating beyond ±1.5 ticks, IM>IC>IF>IH, symmetric. The regression R²
panel never changed — pure visualization-layer artifact.

Corollaries: VOI's lattice is the integers (the raw-axis staircase — no rounding in the
code, queue sizes are whole lots). OIR is dense but piles on small-integer ratios (0, ±⅓,
±½…) — if you ever bin OIR, use **quantile bins**, not fixed width. **Periodic wiggle in
any plot ⇒ suspect the measurement grid before the market.**

## 6. Running it

```zsh
cd ~/xn/manual
source ~/xn/.venv/bin/activate          # or prefix commands with ~/xn/.venv/bin/python

PILOT=1 python voi_cumsum.py            # pilot = 2024, VOI
FACTOR=oir PILOT=1 python voi_cumsum.py # pilot, OIR
python voi_cumsum.py                    # full sample (~30-60 min; nohup + log if long)

WHAT=voi python plot.py                 # figure (SUF=_pilot / FACTOR=oir / XRAW=1)
```
Env vars: `PILOT` `FACTOR` `PRICE` `SUF` `WHAT` `XRAW`. Shell gotchas that bit us once:
no spaces around `=` in `VAR=value`; don't paste `# comments` into interactive zsh; bare
`python`/`python3` is not the venv.

Extending to a new factor = one `add_myfactor(df)` function + the `FACTOR` switch line.
Natural next candidates: MPB (expect depth between VOI and OIR; mind its lattice),
spread-scaled `voi/spr` (model B's actual regressor).

## 7. Check-yourself questions

1. Why must the curve's middle be deeply negative *if the factor works*? What would a
   useless factor look like?
2. The endpoint of the IM 2024 curve is −184K. What two things multiply to produce that
   number, and why can't sorting change it?
3. VOI=0 ticks are ~20% of the sample. Where do they appear in the percentile view, and
   where in the raw-value view?
4. Why `np.argsort` + index-once instead of `df.sort_values("voi")`?
5. dy uses `shift(-k)` inside gid. What exact rows does this silently delete, and what
   would go wrong without the gid?
6. Why is `np.round` dangerous for binning half-way lattice points, and what's the
   half-up idiom?
7. k=20 and k=120 VOI curves nearly coincide. What does that say about how fast the
   market absorbs queue-flow information — and what does it imply for a taker who is
   500ms late? (Cross-check: fig84's latency result.)
