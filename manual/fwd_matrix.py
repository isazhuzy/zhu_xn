"""fwd_matrix.py — (minute-of-day x day) matrix of the H=30min momentum trade returns.

Same computation as future_30mins.py (sign of minute t's mid move x forward H-min
return, fenced per day only — windows DO cross the lunch break, nothing crosses
overnight), but instead of averaging, every individual (minute, day) trade return is
kept and pivoted into a matrix: rows = clock minute, columns = trading days, cells = bps.
Row-mean of this matrix == the curve plotted by future_30mins.py.
NaN cell = flat signal minute, invalid (limit-up) mid, or missing minute.
Run: PILOT=1 /Users/zhuisabella/xn/.venv/bin/python fwd_matrix.py   (env: CODE)
"""
from tick_to_min import *

PILOT = os.environ.get("PILOT") == "1"
SUF = "_pilot" if PILOT else ""
D = "/Users/zhuisabella/xn/manual"
CODE = os.environ.get("CODE", "IF0000")
H = 30
START, END = ("2024.06.01", "2024.06.30") if PILOT else ("2010.01.01", "2026.07.15")

sess = ddb.session(HOST, PORT); sess.login(USER, PW)
b = to_session_bars(fetch_min_bars(sess, CODE, START, END), CODE)
sess.close()

day = b.ts.dt.normalize()
g = b.groupby(day)["mid_close"]   # day-only fence: windows cross the lunch break
sig = np.sign(g.diff())
sig = sig.replace(0, np.nan)                          # flat minute -> no trade
fwd = (g.shift(-H) - b["mid_close"]) / b["mid_close"] * 1e4
b["strat"] = sig * fwd

b["hm"] = b.ts.dt.strftime("%H:%M")
b["d"] = b.ts.dt.strftime("%Y-%m-%d")
out = b.pivot(index="hm", columns="d", values="strat").dropna(how="all")
out.insert(0, "t", np.arange(1, len(out) + 1))
out.to_csv(f"{D}/fwd{H}_matrix_{CODE}{SUF}.csv", float_format="%.3f")

nan_share = out.drop(columns="t").isna().to_numpy().mean()
print(f"{CODE} {START}..{END}: matrix {len(out)} minutes x {out.shape[1]-1} days, "
      f"NaN share {nan_share:.1%}")
print("row means (bps) first/mid/last:",
      np.round(out.drop(columns='t').mean(axis=1).iloc[[0, len(out)//2, -1]].values, 2))
