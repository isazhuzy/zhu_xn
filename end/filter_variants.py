"""
filter_variants.py — try several alternative day/time filters for finding an
end-of-day reversal, beyond the plain "cumret(t) >= 1%" test in
analyze_eod_reversal.py (which found nothing robust).

Each filter defines a per-(day, minute) mask; for every filter we scan minutes
13:00-14:59, report the strongest (most negative t) minute per contract, and
IMMEDIATELY check it ex-2015 (the earlier study's lesson: a pooled-history
"best minute" can be entirely a 2015-crash artifact) plus a per-year
breakdown, so nothing here can quietly repeat that mistake.

Run:  /Users/zhuisabella/xn/.venv/bin/python filter_variants.py
"""
import numpy as np
import pandas as pd

from common import (load, build_panel, day_stats, running_max, rolling_change,
                     day_range, scan_minute, tod_to_hm, year_breakdown, CODES)

AFTERNOON = list(range(780, 900))  # 13:00..14:59


def summarize(name, mask, fwd, years_index, code):
    s = scan_minute(mask, fwd, AFTERNOON)
    if len(s) == 0:
        print(f"  {code}: no qualifying minutes (n too small)")
        return None
    best = s.sort_values("t").iloc[0]
    t_best = int(best["tod"])
    yb = year_breakdown(mask, fwd, t_best, years_index)
    ex15 = yb[yb.year != 2015] if len(yb) else yb
    if len(ex15):
        n_ex = ex15["n"].sum()
        wmean_ex = (ex15["mean_bp"] * ex15["n"]).sum() / n_ex if n_ex else np.nan
    else:
        n_ex, wmean_ex = 0, np.nan
    print(f"  {code}: best={tod_to_hm(t_best)} n={best.n} mean={best.mean_bp}bp "
          f"win_rev={best.win_reversal} t={best.t}   |  ex-2015: n={n_ex} mean={round(wmean_ex,3) if n_ex else 'NA'}bp")
    return dict(filter=name, code=code, best_hm=tod_to_hm(t_best), n=best.n,
                mean_bp=best.mean_bp, win_reversal=best.win_reversal, t=best.t,
                n_ex2015=n_ex, mean_bp_ex2015=round(wmean_ex, 3) if n_ex else np.nan)


if __name__ == "__main__":
    df = load()
    panels = {code: build_panel(df, code) for code in CODES}
    stats = {code: day_stats(panels[code]) for code in CODES}  # open, close_eod, cumret, fwd
    rmax = {code: running_max(stats[code][2]) for code in CODES}
    rchg30 = {code: rolling_change(stats[code][2], 30) for code in CODES}
    drange = {code: day_range(panels[code]) for code in CODES}
    years = {code: pd.Series(stats[code][2].index.year, index=stats[code][2].index) for code in CODES}

    results = []

    # --- F1: threshold sweep (does a BIGGER rally revert more?) ---
    for thresh in [0.005, 0.01, 0.015, 0.02, 0.03]:
        print(f"\n=== F1 threshold sweep: cumret(t) >= {thresh*100:.1f}% ===")
        for code in CODES:
            _, _, cumret, fwd = stats[code]
            mask = cumret >= thresh
            r = summarize(f"thresh_{thresh}", mask, fwd, years[code], code)
            if r:
                results.append(r)

    # --- F2: "at the fresh high" — cumret(t) >= 1% AND currently AT (within
    #     3bp of) the running intraday max, i.e. momentum still extending ---
    print("\n=== F2: up>=1% AND at/near the running intraday high (fresh momentum) ===")
    for code in CODES:
        _, _, cumret, fwd = stats[code]
        mask = (cumret >= 0.01) & ((rmax[code] - cumret) <= 0.0003)
        r = summarize("at_fresh_high", mask, fwd, years[code], code)
        if r:
            results.append(r)

    # --- F3: "faded off the high" — cumret(t) >= 1% but already pulled back
    #     >=20bp from the running intraday max (momentum already stalling) ---
    print("\n=== F3: up>=1% AND already faded >=20bp off the running intraday high ===")
    for code in CODES:
        _, _, cumret, fwd = stats[code]
        mask = (cumret >= 0.01) & ((rmax[code] - cumret) >= 0.002)
        r = summarize("faded_off_high", mask, fwd, years[code], code)
        if r:
            results.append(r)

    # --- F4: "fast spike" — up>=1% AND most of that gain (>=0.7%) happened in
    #     just the last 30 minutes (sharp recent move) ---
    print("\n=== F4: up>=1% AND >=0.7% of it came in the last 30 minutes (fast spike) ===")
    for code in CODES:
        _, _, cumret, fwd = stats[code]
        mask = (cumret >= 0.01) & (rchg30[code] >= 0.007)
        r = summarize("fast_spike", mask, fwd, years[code], code)
        if r:
            results.append(r)

    # --- F5: "slow grind" — up>=1% but <0.2% of it came in the last 30
    #     minutes (gains are old/stale, not a fresh spike) ---
    print("\n=== F5: up>=1% AND <0.2% of it came in the last 30 minutes (stale/slow grind) ===")
    for code in CODES:
        _, _, cumret, fwd = stats[code]
        mask = (cumret >= 0.01) & (rchg30[code] < 0.002)
        r = summarize("slow_grind", mask, fwd, years[code], code)
        if r:
            results.append(r)

    # --- F6: market-wide confirmation — ALL FOUR contracts up>=1% at the same
    #     time t (broad rally, not idiosyncratic to one contract) ---
    print("\n=== F6: ALL FOUR contracts simultaneously up >=1% (market-wide rally) ===")
    common_days = set(stats[CODES[0]][2].index)
    for code in CODES[1:]:
        common_days &= set(stats[code][2].index)
    common_days = sorted(common_days)
    joint_mask = None
    for code in CODES:
        _, _, cumret, _ = stats[code]
        m = (cumret.reindex(index=common_days) >= 0.01).fillna(False)
        joint_mask = m if joint_mask is None else (joint_mask & m)
    for code in CODES:
        _, _, cumret, fwd = stats[code]
        fwd_c = fwd.reindex(index=common_days)
        years_c = years[code].reindex(common_days)
        r = summarize("market_wide", joint_mask, fwd_c, years_c, code)
        if r:
            results.append(r)

    # --- F7: high-volatility day (day range > 2.5% of open) vs calm day, both
    #     conditioned on up>=1% at time t ---
    print("\n=== F7a: up>=1% AND day range >= 2.5% (choppy/high-vol day) ===")
    for code in CODES:
        _, _, cumret, fwd = stats[code]
        mask = cumret >= 0.01
        day_ok = drange[code] >= 0.025
        s = scan_minute(mask, fwd, AFTERNOON, day_filter=day_ok.reindex(mask.index))
        if len(s):
            best = s.sort_values("t").iloc[0]
            t_best = int(best["tod"])
            print(f"  {code}: best={tod_to_hm(t_best)} n={best.n} mean={best.mean_bp}bp win_rev={best.win_reversal} t={best.t}")
            results.append(dict(filter="highvol_day", code=code, best_hm=tod_to_hm(t_best),
                                 n=best.n, mean_bp=best.mean_bp, win_reversal=best.win_reversal,
                                 t=best.t, n_ex2015=np.nan, mean_bp_ex2015=np.nan))

    print("\n=== F7b: up>=1% AND day range < 2.5% (calm/trend day) ===")
    for code in CODES:
        _, _, cumret, fwd = stats[code]
        mask = cumret >= 0.01
        day_ok = drange[code] < 0.025
        s = scan_minute(mask, fwd, AFTERNOON, day_filter=day_ok.reindex(mask.index))
        if len(s):
            best = s.sort_values("t").iloc[0]
            t_best = int(best["tod"])
            print(f"  {code}: best={tod_to_hm(t_best)} n={best.n} mean={best.mean_bp}bp win_rev={best.win_reversal} t={best.t}")
            results.append(dict(filter="calm_day", code=code, best_hm=tod_to_hm(t_best),
                                 n=best.n, mean_bp=best.mean_bp, win_reversal=best.win_reversal,
                                 t=best.t, n_ex2015=np.nan, mean_bp_ex2015=np.nan))

    out = pd.DataFrame(results)
    out.to_csv("/Users/zhuisabella/xn/end/filter_variants_summary.csv", index=False)
    pd.set_option("display.width", 220, "display.max_rows", 200)
    print("\n\n================ SUMMARY (sorted by |t|) ================")
    print(out.reindex(out.t.abs().sort_values(ascending=False).index).to_string(index=False))
    print("\nsaved /Users/zhuisabella/xn/end/filter_variants_summary.csv")
