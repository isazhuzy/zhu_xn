"""
Per-MINUTE breakdown of selected intraday windows, swept across tick thresholds,
now ALSO split by calendar month (data: 20230104-20230304, so mainly
2023-01 vs 2023-02; 2023-03 only has ~2 trading days, read it with caution).

Windows (from the report's 30-min bucket charts, red=positive, green=negative):
  * pos_open : tallest RED bars  — IC0000/IM0000, 09:30 bucket
  * neg_pm   : tallest GREEN bars — IM0000/IC0000, 13:30 bucket

Output -> one CSV per window, with a `month` column ("all", "2023-01", ...):
    minute_{name}_{HHMM}_{HHMM}.csv   (read by plot_breakdown.py)
"""
import os
import pandas as pd
from matrix import _minute_frame, apply_threshold
from analyze import bucket_profile

RAW = "/Users/zhuisabella/xn/ticker/IC_IF_IH_IM_20230104_20230304.csv"
OUT = "/Users/zhuisabella/xn/ticker/open_breakdown"

# ===== CHANGE ME ============================================================
THRESHOLDS = [5, 10, 15, 20]

WINDOWS = {
    "pos_open": ("09:30", "09:59", ["IC0000", "IM0000"]),
    "neg_pm":   ("13:30", "13:59", ["IM0000", "IC0000"]),
}
# ============================================================================


def minute_profile(R, start, end):
    p = bucket_profile(R, freq_min=1)
    return p[(p.index >= start) & (p.index <= end)]


def month_slices(R):
    """Yield ('all', R) plus one slice per calendar month in the row index."""
    idx = pd.to_datetime(R.index)
    yield "all", R
    for per in sorted(idx.to_period("M").unique()):
        yield str(per), R[(idx.year == per.year) & (idx.month == per.month)]


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(RAW, dtype={"code": "string"}, parse_dates=["m_nDatetime"])

    contracts = sorted({c for _, _, cs in WINDOWS.values() for c in cs})
    frames = {c: _minute_frame(df, c, use_mid=True, mode="momentum", lookback=1)
              for c in contracts}

    for name, (start, end, prods) in WINDOWS.items():
        rows = []
        for c in prods:
            for thr in THRESHOLDS:
                R = apply_threshold(frames[c], thr, "tick")
                R = R[pd.to_datetime(R.index).month != 3] 
                for mon, Rm in month_slices(R):
                    if Rm.empty:
                        continue
                    p = minute_profile(Rm, start, end).reset_index()
                    p = p.rename(columns={"bucket": "minute"})
                    if p.empty:
                        continue
                    p.insert(0, "product", c)
                    p.insert(1, "thr_ticks", thr)
                    p.insert(2, "window", name)
                    p.insert(3, "month", mon)
                    rows.append(p)

        if not rows:
            print(f"[{name}] no data in {start}-{end}, skipped")
            continue
        out = pd.concat(rows, ignore_index=True)
        fn = os.path.join(
            OUT, f"minute_{name}_{start.replace(':','')}_{end.replace(':','')}.csv")
        out.to_csv(fn, index=False)
        print(f"[{name}] {start}-{end} | {prods} | months "
              f"{sorted(out['month'].unique())}")
        print(f"  saved {fn}  shape {out.shape}")
