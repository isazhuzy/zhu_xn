"""fwd30_matrix.py — raw (minute-of-day x day) matrix of the H=30min momentum exercise.

Day = 240 CONTIGUOUS trading minutes (idx 1..120 = 09:30-11:29, 121..240 = 13:00-14:59).
Rows: signal minutes t = 1..240-H (=210): every t whose H-minute window ends within
the day. Windows are positional, so they CROSS THE LUNCH BREAK: t in 91..120 exits
in the afternoon (the "30-minute" return there includes the 90-min lunch gap);
likewise the t=121 (13:00) signal is the lunch-gap move vs 11:29. t=1 has no prior
minute -> NaN signal.
Cell = sign(mid(t)-mid(t-1)) * (mid(t+H)-mid(t))/mid(t) in bps; NaN where the mid
is missing/invalid (limit-up minutes) or the signal is flat.
Output: fwd<H>_matrix_<CODE><SUF>.csv — 210 rows x (t, hm + one column per day).
Run: PILOT=1 /Users/zhuisabella/xn/.venv/bin/python fwd30_matrix.py   (sandbox OFF)
"""
import os
import sys
import numpy as np
import pandas as pd
import dolphindb as ddb

sys.path.insert(0, "/Users/zhuisabella/xn/prediction")
sys.path.insert(0, "/Users/zhuisabella/xn/last")
from ddb_config import HOST, PORT, USER, PW
from tick2min_ddb import fetch_min_bars, to_session_bars

PILOT = os.environ.get("PILOT") == "1"
SUF = "_pilot" if PILOT else ""
D = "/Users/zhuisabella/xn/last"
CODE = os.environ.get("CODE", "IF0000")
H = int(os.environ.get("H", "30"))
START, END = ("2024.06.01", "2024.06.30") if PILOT else ("2024.01.01", "2024.12.31")

# canonical 240-minute day grid: 09:30..11:29 + 13:00..14:59
GRID = (pd.date_range("09:30", "11:29", freq="1min").strftime("%H:%M").tolist()
        + pd.date_range("13:00", "14:59", freq="1min").strftime("%H:%M").tolist())

sess = ddb.session(HOST, PORT); sess.login(USER, PW)
b = to_session_bars(fetch_min_bars(sess, CODE, START, END), CODE)
sess.close()

b["day"] = b.ts.dt.strftime("%Y-%m-%d")
b["hm"] = b.ts.dt.strftime("%H:%M")
# day x 240-minute grid of mids; missing minutes stay NaN (positional = contiguous)
mid = b.pivot(index="hm", columns="day", values="mid_close").reindex(GRID)
M = mid.to_numpy(float)                                # 240 x ndays

sig = np.sign(M - np.vstack([np.full((1, M.shape[1]), np.nan), M[:-1]]))
sig[sig == 0] = np.nan                                 # flat minute -> no trade
fwd = np.vstack([M[H:], np.full((H, M.shape[1]), np.nan)])
strat = sig * (fwd - M) / M * 1e4                      # bps

out = pd.DataFrame(strat[:240 - H], index=GRID[:240 - H], columns=mid.columns)
out.insert(0, "t", np.arange(1, 240 - H + 1))
out.index.name = "hm"
out.to_csv(f"{D}/fwd{H}_matrix_{CODE}{SUF}.csv", float_format="%.3f")

nan_share = np.isnan(strat[:240 - H]).mean()
print(f"{CODE} {START}..{END}: matrix {240 - H} minutes x {mid.shape[1]} days, "
      f"NaN share {nan_share:.1%}")
print(out.iloc[:3, :4].to_string())
print("row means (bps): first/mid/last =",
      np.round(np.nanmean(strat[:240 - H], axis=1)[[0, 105, 209]], 2))
