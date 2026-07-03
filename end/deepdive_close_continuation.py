"""
deepdive_close_continuation.py — follow-up to a surprise finding: scanning
minute-by-minute (scan_by_minute.csv), at 14:55 the sign flips from "weak
reversal" to a SIGNIFICANT CONTINUATION (up>=1% days keep grinding higher
into the close) for IF/IH/IM, t up to 5.3, robust across years (NOT a 2015
artifact — 2015 is actually one of the weaker years for IF/IH/IM here).

Two things to nail down at minute resolution, per the user's ask:
 1. Zoom the full afternoon win-rate/t-stat curve into 14:30-15:00 to see
    exactly where "reversal" turns into "continuation".
 2. Instead of the blunt "cumret(t)>=1% from open" trigger, screen recent
    momentum over much shorter lookback windows (1/2/3/5/10/15/20/30 min)
    ending at several close-proximity signal times (14:45..14:59, esp 14:55):
    does "still rising in just the last W minutes" sharpen the continuation
    signal further?

Run:  /Users/zhuisabella/xn/.venv/bin/python deepdive_close_continuation.py
"""
import numpy as np
import pandas as pd

from common import load, build_panel, day_stats, scan_minute, tod_to_hm, CODES

df = load()
CLOSE_ZOOM = list(range(870, 900))     # 14:30..14:59
SIGNAL_TIMES = [14 * 60 + m for m in (45, 48, 50, 52, 53, 55, 57, 58, 59)]
WINDOWS = [1, 2, 3, 5, 10, 15, 20, 30]

zoom_rows, window_rows = [], []
for code in CODES:
    piv = build_panel(df, code)
    open_, close_eod, cumret, fwd = day_stats(piv)
    years = pd.Series(cumret.index.year, index=cumret.index)

    # --- part 1: zoom 14:30-14:59, plain up>=1%-from-open trigger ---
    base = cumret >= 0.01
    s = scan_minute(base, fwd, CLOSE_ZOOM)
    s["code"] = code
    zoom_rows.append(s)

    # --- part 2: window-length screen at close-proximity signal times ---
    for t in SIGNAL_TIMES:
        for w in WINDOWS:
            if (t - w) not in cumret.columns:
                continue
            recent_move = cumret[t] - cumret[t - w]
            # trigger: day already up>=1% from open AND still rising over the
            # last w minutes (recent_move > 0) -- "fresh, still-running" momentum
            mask_t = (cumret[t] >= 0.01) & (recent_move > 0)
            y = (fwd[t] * 1e4)[mask_t].dropna()
            if len(y) < 15:
                continue
            mn, sd = y.mean(), y.std(ddof=1)
            tt = mn / (sd / np.sqrt(len(y))) if sd > 0 else np.nan
            window_rows.append(dict(code=code, tod=t, hm=tod_to_hm(t), w=w, n=len(y),
                                     mean_bp=round(mn, 3), win_rev=round((y < 0).mean(), 3),
                                     t=round(tt, 2)))

zoomtab = pd.concat(zoom_rows, ignore_index=True)
zoomtab["hm"] = zoomtab["tod"].map(tod_to_hm)
zoomtab.to_csv("/Users/zhuisabella/xn/end/deepdive_zoom_1430_1500.csv", index=False)

wintab = pd.DataFrame(window_rows)
wintab.to_csv("/Users/zhuisabella/xn/end/deepdive_window_screen.csv", index=False)

pd.set_option("display.width", 220, "display.max_rows", 400)
print("=== zoom 14:30-14:59, up>=1% from open (plain trigger) ===")
for code in CODES:
    print(f"\n{code}:")
    print(zoomtab[zoomtab.code == code][["hm", "n", "mean_bp", "win_reversal", "t"]].to_string(index=False))

print("\n\n=== window-length screen: up>=1% from open AND still rising over last w min ===")
for code in CODES:
    print(f"\n{code}: (rows=signal time, cols=lookback window w, cell=t-stat)")
    piv = wintab[wintab.code == code].pivot(index="hm", columns="w", values="t")
    print(piv.to_string())

print("\nsaved deepdive_zoom_1430_1500.csv, deepdive_window_screen.csv")
