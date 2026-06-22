import pandas as pd, numpy as np
df = pd.read_csv("minute_conditional_isoos.csv")
df["pnl"] = np.sign(df.dprev) * df.ret
df["absd"] = df.dprev.abs()
print("sanity:", df.shape[0], "rows; pnl NaN:", df.pnl.isna().sum(),
      "| pnl mean/std:", round(df.pnl.mean(), 4), round(df.pnl.std(), 3))

NAME = {"IC0000": "IC", "IF0000": "IF", "IH0000": "IH", "IM0000": "IM"}

# bucket |dprev| into 5 within each (sample,code,ym) via percentile rank (regime-neutral)
pct = df.groupby(["sample", "code", "ym"])["absd"].rank(pct=True)
df["Q"] = np.ceil(pct * 5).clip(1, 5).astype(int)
print("Q values:", sorted(df.Q.dropna().unique()), "| Q NaN:", df.Q.isna().sum())

for code in NAME:
    print(f"\n=== {NAME[code]} ===  动量P&L 按上一分钟波幅分桶 (Q1=小动 … Q5=大动)")
    print(f"{'桶':>4}{'IS均值':>9}{'IS_t':>7}{'OOS均值':>10}{'OOS_t':>7}")
    for q in [1, 2, 3, 4, 5]:
        out = {}
        for s in ["IS", "OOS"]:
            d = df[(df.code == code) & (df["sample"] == s) & (df.Q == q)]["pnl"]
            out[s] = (d.mean(), d.mean() / (d.std() / np.sqrt(len(d))))
        print(f"{q:>4}{out['IS'][0]:>9.3f}{out['IS'][1]:>7.1f}{out['OOS'][0]:>10.3f}{out['OOS'][1]:>7.1f}")
    for s in ["IS", "OOS"]:
        q5 = df[(df.code == code) & (df["sample"] == s) & (df.Q == 5)]["pnl"]
        q1 = df[(df.code == code) & (df["sample"] == s) & (df.Q == 1)]["pnl"]
        sp = q5.mean() - q1.mean(); ts = sp / np.sqrt(q5.var() / len(q5) + q1.var() / len(q1))
        print(f"   {s} Q5−Q1 = {sp:+.3f}pt (t={ts:+.1f})")
