"""
summary_verdict.py — consolidated, precise verdict across every effect tested
in xn/end: for each, compute the POOLED (not year-weighted-average) t-stat
both on full history and excluding 2015, at the exact minute/mask already
identified as that effect's strongest point earlier in the study. This is
the single source of truth the forest-plot figure is built from.

Four representative effects, one per contract each (16 rows):
  A. plain "up>=1% from open" reversal-hunt best minute (analyze_eod_reversal.py)
  B. path-ratio "noisy" (choppy path) subsample best minute (filter_path_ratio.py)
  C. "calm day" filter, REAL-TIME-computable version (range up to t, not full-day)
  D. closing-minutes CONTINUATION @ 14:55, plain up>=1% trigger (deepdive script)

Run:  /Users/zhuisabella/xn/.venv/bin/python summary_verdict.py
"""
import numpy as np
import pandas as pd

from common import load, build_panel, day_stats, path_ratio, tod_to_hm, CODES

df = load()
AFTERNOON = list(range(780, 900))
THRESH = 0.01

BEST_MIN_A = {"IC0000": 14 * 60 + 42, "IF0000": 14 * 60 + 39, "IH0000": 13 * 60 + 0, "IM0000": 13 * 60 + 37}
BEST_MIN_B = {"IC0000": 13 * 60 + 0, "IF0000": 14 * 60 + 38, "IH0000": 13 * 60 + 11, "IM0000": 13 * 60 + 37}
BEST_MIN_D = {c: 14 * 60 + 55 for c in CODES}


def pooled_t(mask_col, fwd_col, years, ex2015=False):
    m = mask_col.fillna(False)
    if ex2015:
        m = m & (years != 2015)
    y = (fwd_col * 1e4)[m].dropna()
    if len(y) < 5:
        return dict(n=len(y), mean_bp=np.nan, win_rev=np.nan, t=np.nan)
    mn, sd = y.mean(), y.std(ddof=1)
    t = mn / (sd / np.sqrt(len(y))) if sd > 0 else np.nan
    return dict(n=len(y), mean_bp=round(mn, 3), win_rev=round((y < 0).mean(), 3), t=round(t, 2))


rows = []
for code in CODES:
    piv = build_panel(df, code)
    open_, close_eod, cumret, fwd = day_stats(piv)
    years = pd.Series(cumret.index.year, index=cumret.index)
    ratio = path_ratio(piv)

    # A: plain threshold
    tA = BEST_MIN_A[code]
    maskA = cumret[tA] >= THRESH
    fullA, exA = pooled_t(maskA, fwd[tA], years), pooled_t(maskA, fwd[tA], years, ex2015=True)
    rows.append(dict(effect="A. plain reversal-hunt (best min)", code=code, hm=tod_to_hm(tA), **{f"full_{k}": v for k, v in fullA.items()}, **{f"ex15_{k}": v for k, v in exA.items()}))

    # B: path-ratio noisy subsample
    tB = BEST_MIN_B[code]
    allvals = pd.concat([ratio[t][cumret[t] >= THRESH] for t in AFTERNOON if t in ratio.columns])
    med = allvals.median()
    maskB = (cumret[tB] >= THRESH) & (ratio[tB] > med)
    fullB, exB = pooled_t(maskB, fwd[tB], years), pooled_t(maskB, fwd[tB], years, ex2015=True)
    rows.append(dict(effect="B. path-ratio 'noisy' subsample", code=code, hm=tod_to_hm(tB), **{f"full_{k}": v for k, v in fullB.items()}, **{f"ex15_{k}": v for k, v in exB.items()}))

    # C: calm day, REAL-TIME range (no look-ahead)
    run_max_px, run_min_px = piv.cummax(axis=1), piv.cummin(axis=1)
    range_so_far = run_max_px.sub(run_min_px, axis=0).div(open_, axis=0)
    maskC_full = (cumret >= THRESH) & (range_so_far < 0.015)
    sC = {t: pooled_t(maskC_full[t], fwd[t], years) for t in AFTERNOON if t in maskC_full.columns}
    tC = min((t for t in sC if not np.isnan(sC[t]["t"])), key=lambda t: sC[t]["t"], default=None)
    if tC is not None:
        fullC, exC = pooled_t(maskC_full[tC], fwd[tC], years), pooled_t(maskC_full[tC], fwd[tC], years, ex2015=True)
        rows.append(dict(effect="C. calm-day (real-time range)", code=code, hm=tod_to_hm(tC), **{f"full_{k}": v for k, v in fullC.items()}, **{f"ex15_{k}": v for k, v in exC.items()}))

    # D: closing continuation @ 14:55
    tD = BEST_MIN_D[code]
    maskD = cumret[tD] >= THRESH
    fullD, exD = pooled_t(maskD, fwd[tD], years), pooled_t(maskD, fwd[tD], years, ex2015=True)
    rows.append(dict(effect="D. closing continuation @14:55", code=code, hm=tod_to_hm(tD), **{f"full_{k}": v for k, v in fullD.items()}, **{f"ex15_{k}": v for k, v in exD.items()}))

out = pd.DataFrame(rows)
out.to_csv("/Users/zhuisabella/xn/end/summary_verdict.csv", index=False)
pd.set_option("display.width", 240, "display.max_rows", 60)
print(out.to_string(index=False))
print("\nsaved summary_verdict.csv")
