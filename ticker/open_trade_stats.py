"""
open_trade_stats.py — trade count, win rate, and the effect of raising the tick
threshold, for the open's first two minutes.

The first minute is a REVERSAL, so the tradeable signal is CONTRARIAN: when the
9:30->9:31 move exceeds `thr` ticks, take the OPPOSITE side into 9:31->9:32.
We report, per (product, minute, threshold):
    n_trades   = days the move exceeded the threshold (a trade is taken)
    win_rate   = fraction of those trades that are profitable (contrarian)
    avg_bp     = mean P&L per trade, bp (contrarian = -momentum)
    total_bp   = summed P&L, bp
    t          = t-stat of per-trade P&L vs 0
"""
import numpy as np
import pandas as pd
from first_two_minutes import open_bars, momentum_series, PRODUCTS

SRC = "/tmp/open_ticks.csv"
THRESHOLDS = [5, 10, 15, 20, 25, 30]
MINUTES = ["09:31", "09:32"]


def contrarian_row(s):
    """s = momentum per-day return series (drop flats). Contrarian = -s."""
    c = -s
    n = len(c)
    t = c.mean() / (c.std(ddof=1) / np.sqrt(n)) if n > 1 and c.std(ddof=1) > 0 else np.nan
    return dict(n_trades=n, win_rate=round((c > 0).mean(), 3),
                avg_bp=round(c.mean() * 1e4, 2), total_bp=round(c.sum() * 1e4, 1),
                t=round(t, 2))


if __name__ == "__main__":
    df = pd.read_csv(SRC, dtype={"code": "string"}, parse_dates=["m_nDatetime"])
    bars = {c: open_bars(df, c) for c in PRODUCTS}

    rows = []
    for c in PRODUCTS:
        for minute in MINUTES:
            for thr in THRESHOLDS:
                s = momentum_series(bars[c], minute, thr)
                if len(s):
                    rows.append({"product": c, "minute": minute, "thr": thr,
                                 **contrarian_row(s)})
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_rows", 300)
    for minute in MINUTES:
        sub = tab[tab.minute == minute]
        print(f"\n=== CONTRARIAN (fade-the-open) @ {minute} — effect of raising ticks ===")
        print(sub.drop(columns='minute').to_string(index=False))
    tab.to_csv("/Users/zhuisabella/xn/ticker/open_breakdown/open_trade_stats.csv",
               index=False)
