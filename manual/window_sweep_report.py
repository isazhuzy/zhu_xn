"""Render the window-sweep results as readable tables.
Reads window_sweep_<factor><suf>.csv (long format from window_sweep.py)."""
import os
import pandas as pd

FACTOR = os.environ.get("FACTOR", "voi")
SUF = "_pilot" if os.environ.get("PILOT") == "1" else ""
D = "/Users/zhuisabella/xn/manual"
df = pd.read_csv(f"{D}/window_sweep_{FACTOR}{SUF}.csv")

CODES = ["IC0000", "IF0000", "IH0000", "IM0000"]
BORDER = ["neg0.1", "neg1", "neg5", "pos5", "pos1", "pos0.1"]
SEC = {1: "0.5s", 5: "2.5s", 20: "10s", 60: "30s", 120: "60s", 240: "120s", 600: "300s"}


def tbl(sub, rowkey, rowvals):
    """print a bucket x contract table for each rowval (J or k)."""
    for rv in rowvals:
        d = sub[sub[rowkey] == rv]
        print(f"\n  {rowkey}={rv} ({SEC[rv]}):")
        piv = d.pivot_table(index="bucket", columns="code", values="mean_dy")
        piv = piv.reindex(BORDER)[[c for c in CODES if c in piv.columns]]
        print(piv.round(2).to_string())


print("=" * 70)
print("EXPERIMENT A — forward-horizon sweep (look-back J=1, instantaneous VOI)")
print("  bucket mean forward return, ticks/trade")
print("=" * 70)
tbl(df[df.J == 1], "k", sorted(df.k.unique()))

print("\n" + "=" * 70)
print("EXPERIMENT B — look-back sweep (forward k=20 = 10s hold)")
print("  factor = rolling VOI sum over past J ticks")
print("=" * 70)
tbl(df[df.k == 20], "J", sorted(df.J.unique()))

# compact peak-horizon summary: for J=1, the pos0.1 & neg0.1 tail vs k
print("\n" + "=" * 70)
print("TAIL vs FORWARD HORIZON (J=1): most-extreme 0.1% buckets")
print("=" * 70)
for code in CODES:
    d = df[(df.J == 1) & (df.code == code)]
    if d.empty:
        continue
    print(f"\n  {code}")
    for b in ["neg0.1", "pos0.1"]:
        s = d[d.bucket == b].set_index("k")["mean_dy"].reindex(sorted(df.k.unique()))
        row = "  ".join(f"{SEC[k]}:{v:+.2f}" for k, v in s.items())
        print(f"    {b:7s} {row}")
