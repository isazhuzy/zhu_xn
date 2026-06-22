"""
Fetch the FULL-history open bars by looping quarter-by-quarter (each query touches
<=12 partitions = the validated-safe size, avoiding the server's open-files limit).
Concatenate and save to /tmp/open_bars_all.csv.
"""
import time
import pandas as pd
from fetch_open_bars import fetch

QUARTERS = [(f"{y}.{m:02d}.01", f"{y}.{m+2:02d}.{last:02d}")
            for y in range(2015, 2027)
            for m, last in [(1, 31), (4, 30), (7, 30), (10, 31)]]  # Sep ends 30
# trim to the known data range 2015-04 .. 2026-06
QUARTERS = [q for q in QUARTERS if q[0] >= "2015.04.01" and q[0] <= "2026.06.01"]

if __name__ == "__main__":
    parts, t0 = [], time.time()
    for i, (s, e) in enumerate(QUARTERS, 1):
        try:
            df = fetch(s, e)
        except Exception as ex:
            print(f"[{i}/{len(QUARTERS)}] {s}..{e}  ERROR {str(ex)[:80]}", flush=True)
            continue
        n = 0 if df is None else len(df)
        if n:
            parts.append(df)
        print(f"[{i}/{len(QUARTERS)}] {s}..{e}  rows={n}  "
              f"({time.time()-t0:.0f}s)", flush=True)
    allbars = pd.concat(parts, ignore_index=True)
    allbars.to_csv("/tmp/open_bars_all.csv", index=False)
    days = allbars.groupby("code")["d"].nunique().to_dict()
    print("\nSAVED /tmp/open_bars_all.csv  rows:", len(allbars))
    print("distinct days per product:", days)
