"""Diagnostics on the full-history open bars: locate bad ticks / glitch days."""
import numpy as np, pandas as pd
from analyze_open_bars import load_wide, PRODUCTS, ols

w = load_wide("/tmp/open_bars_all.csv")
w["yr"] = w["d"].dt.year
w["m1"] = (w["c31"] / w["c30"] - 1) * 1e4
w["m2"] = (w["c32"] / w["c31"] - 1) * 1e4
pd.set_option("display.width", 200, "display.max_rows", 200)

print("=== m1 (9:30->9:31, bp) distribution per product ===")
print(w.groupby("code")["m1"].describe(percentiles=[.01, .5, .99]).round(1).to_string())

print("\n=== # days with |m1|>150bp (glitch suspects) per product per year ===")
w["glitch"] = w["m1"].abs() > 150
g = w.groupby(["code", "yr"])["glitch"].sum().unstack("yr").fillna(0).astype(int)
print(g.to_string())

print("\n=== IH reversal regression m2~m1, BY YEAR (localize the artifact) ===")
for yr, wy in w[w.code == "IH0000"].dropna(subset=["m1", "m2"]).groupby("yr"):
    print(yr, ols(wy["m1"], wy["m2"]))

print("\n=== worst 8 IH days by |m1| ===")
ih = w[w.code == "IH0000"].copy()
ih["abs1"] = ih["m1"].abs()
print(ih.nlargest(8, "abs1")[["d", "c30", "c31", "c32", "c33", "m1", "m2"]].to_string(index=False))
