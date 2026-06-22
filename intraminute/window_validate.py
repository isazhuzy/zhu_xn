"""window_validate.py — does the 15-min-window time-of-day structure hold on all 76 months?
For each (contract, window): fraction of months with negative monthly-mean (reversal) vs
positive (continuation), cross-month median, and a sign-consistency read. Compares the full
history to the 12-month IS+OOS finding. Run: python3 window_validate.py"""
import numpy as np, pandas as pd
NAME = {"IC0000": "IC", "IF0000": "IF", "IH0000": "IH", "IM0000": "IM"}
df = pd.read_csv("window_allmonths.csv")
df["wl"] = df["win"].map(lambda t: f"{t//60:02d}:{t%60:02d}")
wins = sorted(df.win.unique())

# per (code, window): frac of months negative, median monthly mean
print("各时段 跨76个月 一致性  (frac_neg=做动量为负即反转的月份占比; med=月度均值中位数)\n")
hdr = "win   " + "".join([f"{NAME[c]:>16}" for c in NAME])
print(hdr)
for w in wins:
    cells = []
    for c in NAME:
        s = df[(df.code == c) & (df.win == w)]["mean"]
        fn = (s < 0).mean()
        cells.append(f"{fn*100:>4.0f}%neg med{s.median():+.2f}")
    print(f"{df[df.win==w]['wl'].iloc[0]}  " + "".join([f"{x:>16}" for x in cells]))

print("\n=== 关键时段裁决 (四合约综合: 多少个合约≥70%月份同号) ===")
for w, name, exp in [(615, "10:15", "neg"), (630, "10:30", "neg"), (645, "10:45", "neg"),
                     (855, "14:15", "neg"), (675, "11:15", "pos"), (780, "13:00", "pos"),
                     (825, "13:45", "pos")]:
    agree = 0
    for c in NAME:
        s = df[(df.code == c) & (df.win == w)]["mean"]
        frac = (s < 0).mean() if exp == "neg" else (s > 0).mean()
        if frac >= 0.70:
            agree += 1
    verdict = "✓稳健" if agree >= 3 else ("~部分" if agree == 2 else "✗不成立")
    print(f"  {name} (预期{'反转' if exp=='neg' else '延续'}): {agree}/4 合约≥70%月份同号  {verdict}")
