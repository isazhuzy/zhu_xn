"""
filter_path_ratio.py — condition the >=1% up-day reversal test on 路程/位移比
(path-length / net displacement from open), a real-time-computable "how
efficient/straight was the move so far" feature (ratio=1 is a perfectly
straight-line rally; larger = more zig-zag/noise for the same net gain).

Split at the running median ratio (computed only from qualifying obs at that
same minute, so the split is time-local and comparable) into "smooth" (below
median, efficient trend) vs "noisy" (above median, choppy) subsamples, both
still requiring cumret(t) >= 1%. Hypothesis to test: does a noisy/choppy path
to the gain revert more (exhausted, low-conviction) than a smooth one, or vice
versa (momentum persists better after a clean trend)?

Scans minutes 13:00-14:59, day-clustered, then immediately checks ex-2015 +
year breakdown for the strongest-looking minute in each subsample (same
discipline as filter_variants.py, since this is exactly the kind of feature
that can quietly leak the future if not built carefully).

Run:  /Users/zhuisabella/xn/.venv/bin/python filter_path_ratio.py
"""
import numpy as np
import pandas as pd

from common import load, build_panel, day_stats, path_ratio, scan_minute, tod_to_hm, CODES

AFTERNOON = list(range(780, 900))  # 13:00..14:59
THRESH = 0.01


def year_check(mask, fwd, t_best, years, label):
    rows = []
    for yr in sorted(years.unique()):
        m = mask[t_best] & (years == yr)
        y = (fwd[t_best] * 1e4)[m].dropna()
        if len(y) >= 3:
            mn, s = y.mean(), y.std(ddof=1)
            t_stat = mn / (s / np.sqrt(len(y))) if s > 0 else np.nan
            rows.append(dict(year=yr, n=len(y), mean_bp=round(mn, 3),
                              win_rev=round((y < 0).mean(), 3), t=round(t_stat, 2)))
    tab = pd.DataFrame(rows)
    print(f"  --- {label} @ {tod_to_hm(t_best)} by year ---")
    if len(tab):
        print("  " + tab.to_string(index=False).replace("\n", "\n  "))
        ex = tab[tab.year != 2015]
        if len(ex) and ex["n"].sum():
            wmean = (ex.mean_bp * ex.n).sum() / ex.n.sum()
            print(f"  ex-2015: n={ex.n.sum()} weighted mean={wmean:.3f}bp")
    print()


if __name__ == "__main__":
    df = load()
    results = []
    for code in CODES:
        piv = build_panel(df, code)
        open_, close_eod, cumret, fwd = day_stats(piv)
        ratio = path_ratio(piv)
        years = pd.Series(cumret.index.year, index=cumret.index)
        base = cumret >= THRESH

        print(f"\n===== {code}: 路程/位移比 distribution among up>=1% obs =====")
        allvals = pd.concat([ratio[t][base[t]] for t in AFTERNOON if t in ratio.columns])
        print(f"  n={len(allvals)}  median={allvals.median():.2f}  "
              f"p25={allvals.quantile(.25):.2f}  p75={allvals.quantile(.75):.2f}  "
              f"p95={allvals.quantile(.95):.2f}")
        med = allvals.median()

        smooth_mask = base & (ratio <= med)
        noisy_mask = base & (ratio > med)

        for label, mask in [("smooth (ratio<=median, efficient trend)", smooth_mask),
                             ("noisy (ratio>median, choppy path)", noisy_mask)]:
            s = scan_minute(mask, fwd, AFTERNOON)
            if len(s) == 0:
                print(f"  {label}: no qualifying minutes")
                continue
            best = s.sort_values("t").iloc[0]
            t_best = int(best["tod"])
            print(f"  {label}: best={tod_to_hm(t_best)} n={best.n} mean={best.mean_bp}bp "
                  f"win_rev={best.win_reversal} t={best.t}")
            year_check(mask, fwd, t_best, years, f"{code} {label}")
            results.append(dict(code=code, group=label, best_hm=tod_to_hm(t_best),
                                 n=best.n, mean_bp=best.mean_bp,
                                 win_reversal=best.win_reversal, t=best.t))

        # continuous check: within the up>=1% sample, correlate ratio(t) with
        # fwd(t) directly (does a higher ratio predict a bigger/smaller reversal?)
        print(f"  --- continuous corr(ratio(t), fwd(t)*1e4) within up>=1%, per afternoon minute ---")
        for t in [780, 810, 840, 870, 899]:
            if t not in ratio.columns:
                continue
            m = base[t]
            x = ratio[t][m]
            y = (fwd[t] * 1e4)[m]
            ok = x.notna() & y.notna()
            if ok.sum() >= 20:
                corr = x[ok].corr(y[ok])
                print(f"    {tod_to_hm(t)}: n={ok.sum()} corr={corr:.3f}")

    out = pd.DataFrame(results)
    out.to_csv("/Users/zhuisabella/xn/end/filter_path_ratio_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n\n================ SUMMARY ================")
    print(out.to_string(index=False))
    print("\nsaved /Users/zhuisabella/xn/end/filter_path_ratio_summary.csv")
