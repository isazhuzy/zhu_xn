"""
first_two_minutes.py — focused deep-dive on the open's first two signal minutes.

Framework (identical to matrix.py): 1-min mid-price bars, momentum signal
  signal_move(t) = close(t) - close(t-1)      pos = sign(signal_move)
  fwd(t)         = close(t+1)/close(t) - 1
  r(t)           = pos * fwd   (only if |signal_move| > thr * tick_size)

"First two minutes" map to matrix columns:
  09:31  = trade the 9:30->9:31 move, paid the 9:31->9:32 return
  09:32  = trade the 9:31->9:32 move, paid the 9:32->9:33 return

Runs on the small open-window extract /tmp/open_ticks.csv (09:29-09:34 ticks,
the 4 continuous contracts) so it is fast. Reproduces matrix.py exactly.
"""
import numpy as np
import pandas as pd
from datetime import time as dtime

SRC = "/tmp/open_ticks.csv"
TICK = 0.2                       # CFFEX index-future tick, as used in matrix.py
THRESHOLDS = [5, 10, 15, 20]
PRODUCTS = ["IC0000", "IF0000", "IH0000", "IM0000"]
MINUTES = ["09:31", "09:32"]
EXCLUDE_MONTHS = [3]


def open_bars(df, contract):
    """ticks -> 1-min mid bars (9:30..9:33), momentum signal + fwd return."""
    d = df.loc[df["code"] == contract].copy()
    d["ts"] = pd.to_datetime(d["m_nDatetime"])
    d["px"] = (d["m_nBidPrice"] + d["m_nAskPrice"]) / 2.0
    s = d.set_index("ts").sort_index()["px"]
    bars = s.resample("1min").last().dropna().rename("close").reset_index()
    bars["day"] = bars["ts"].dt.normalize()
    bars["tod"] = bars["ts"].dt.time
    bars = bars[(bars["tod"] >= dtime(9, 30)) & (bars["tod"] <= dtime(11, 30))].copy()
    g = bars.groupby("day")
    bars["px_next"] = g["close"].shift(-1)
    bars["px_prev"] = g["close"].shift(1)
    bars["signal_move"] = bars["close"] - bars["px_prev"]
    bars["fwd"] = bars["px_next"] / bars["close"] - 1.0
    bars["pos"] = np.sign(bars["signal_move"])
    return bars


def momentum_series(bars, minute, thr):
    """per-day momentum return at one minute, threshold in ticks; drop flats."""
    h, m = map(int, minute.split(":"))
    b = bars[bars["tod"] == dtime(h, m)].copy()
    keep = b["signal_move"].abs() > thr * TICK
    r = np.where(keep, b["pos"] * b["fwd"], 0.0)
    out = pd.Series(r, index=b["day"]).replace(0, np.nan).dropna()
    out = out[~out.index.month.isin(EXCLUDE_MONTHS)].sort_index()
    return out


def stats_row(s):
    n = len(s)
    if n == 0:
        return None
    total = s.sum()
    t = s.mean() / (s.std(ddof=1) / np.sqrt(n)) if n > 1 and s.std(ddof=1) > 0 else np.nan
    top3 = s.sort_values(ascending=(total < 0)).iloc[:3].sum()
    jan, feb = s[s.index.month == 1], s[s.index.month == 2]
    return dict(n=n, total_bp=round(total * 1e4, 1), mean_bp=round(s.mean() * 1e4, 2),
                median_bp=round(s.median() * 1e4, 2), hit=round((s > 0).mean(), 3),
                t=round(t, 2), top3_share=round(top3 / total, 2) if total else np.nan,
                jan_bp=round(jan.sum() * 1e4, 1), feb_bp=round(feb.sum() * 1e4, 1))


if __name__ == "__main__":
    df = pd.read_csv(SRC, dtype={"code": "string"}, parse_dates=["m_nDatetime"])
    bars = {c: open_bars(df, c) for c in PRODUCTS}

    rows = []
    for c in PRODUCTS:
        for minute in MINUTES:
            for thr in THRESHOLDS:
                s = momentum_series(bars[c], minute, thr)
                st = stats_row(s)
                if st:
                    rows.append({"product": c, "minute": minute, "thr": thr, **st})
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_rows", 200)
    print("=== Momentum PnL at the first two open minutes (March excluded) ===")
    print("(negative total => momentum LOSES => the move REVERSES)\n")
    print(tab.to_string(index=False))
    tab.to_csv("/Users/zhuisabella/xn/ticker/open_breakdown/first_two_minutes.csv",
               index=False)
