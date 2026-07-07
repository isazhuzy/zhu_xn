"""
flow_prob_sweep_eod.py — EOD-context version of flow_prob_sweep.py: sweep the aggressor
SELL-PRESSURE imbalance vs P(price is UP at the 15:00 CLOSE relative to the signal minute),
i.e. forward horizon = "to close" instead of a fixed 1/5/30 min. This is the continuous
sweep behind the median hi/lo split in fig_flow_sellpressure_path.

Feature: imb(t)=(sell_W-buy_W)/(sell_W+buy_W), trailing 30min, afternoon-only.
Target: sign(close(15:00) - close(t)).  Pooled over ALL days, full history, two signal windows:
  - whole afternoon 13:30-14:59
  - last 30 min    14:30-14:59
Writes csv + fig (2x2 contracts, two curves each). Run on system python3.
"""
import numpy as np
import pandas as pd
from common import CODES
from flow_analysis import load_flow, panels

W = 30
NB = 30
THRESH_BP = 150
CLOSE_TOD = 900
WINDOWS = {"whole afternoon (13:30-14:59)": (810, 899),
           "last 30 min (14:30-14:59)": (870, 899)}
centers = -1 + (np.arange(NB) + 0.5) * (2 / NB)


def resolve_semantics(df):
    cn = 0.0
    for code in CODES:
        close, ab, aa = panels(df, code)
        ret, sig = close.diff(axis=1), ab - aa
        m = (ret.notna() & sig.notna()).to_numpy()
        if m.sum() > 1000:
            cn += np.corrcoef(ret.to_numpy()[m], sig.to_numpy()[m])[0, 1]
    return cn > 0


if __name__ == "__main__":
    df = load_flow()
    actbid_is_buy = resolve_semantics(df)
    print(f"actbid is {'BUYER' if actbid_is_buy else 'SELLER'}-initiated\n")

    res = {}
    for code in CODES:
        close, ab, aa = panels(df, code)
        buy = ab if actbid_is_buy else aa
        sell = aa if actbid_is_buy else ab
        steps = close.diff(axis=1).div(close.shift(1, axis=1)).abs() * 1e4
        keep = ~steps.gt(THRESH_BP).any(axis=1)
        close, buy, sell = close.loc[keep], buy.loc[keep], sell.loc[keep]
        acols = [c for c in sorted(close.columns) if 780 <= c <= CLOSE_TOD]
        close_a, buy_a, sell_a = close[acols], buy[acols], sell[acols]
        buyW = buy_a.T.rolling(W, min_periods=W).sum().T
        sellW = sell_a.T.rolling(W, min_periods=W).sum().T
        imb = (sellW - buyW) / (sellW + buyW)
        # forward-to-close return for every afternoon minute
        c_close = close_a[CLOSE_TOD]
        fwd = close_a.rsub(c_close, axis=0)          # close(15:00) - close(t)

        res[code] = {}
        for wlab, (t0, t1) in WINDOWS.items():
            scols = [c for c in acols if t0 <= c <= t1]
            X = imb[scols].to_numpy().ravel()
            F = fwd[scols].to_numpy().ravel()
            valid = np.isfinite(X) & np.isfinite(F) & (F != 0)
            x, up = X[valid], (F[valid] > 0).astype(float)
            b = np.clip(((x + 1) / 2 * NB).astype(int), 0, NB - 1)
            n = np.bincount(b, minlength=NB).astype(float)
            k = np.bincount(b, weights=up, minlength=NB)
            res[code][wlab] = (n, k)

    rows = []
    for code in CODES:
        for wlab in WINDOWS:
            n, k = res[code][wlab]
            for i in range(NB):
                if n[i] > 0:
                    rows.append(dict(code=code, window=wlab, imb=round(centers[i], 3),
                                     n=int(n[i]), p_up_close=round(k[i] / n[i], 4)))
    pd.DataFrame(rows).to_csv("/Users/zhuisabella/xn/end/flow_prob_sweep_eod.csv", index=False)

    print("=== P(up-to-close): net-buying bins (imb<-0.1) vs net-selling bins (imb>+0.1) ===")
    for code in CODES:
        for wlab in WINDOWS:
            n, k = res[code][wlab]
            p = k / np.where(n > 0, n, np.nan)
            lo = np.nansum(k[centers < -0.1]) / max(np.nansum(n[centers < -0.1]), 1)
            hi = np.nansum(k[centers > 0.1]) / max(np.nansum(n[centers > 0.1]), 1)
            print(f" {code} [{wlab[:14]}]: buy-heavy P(up)={lo:.3f}  sell-heavy P(up)={hi:.3f}  diff={hi-lo:+.3f}")

    import matplotlib.pyplot as plt
    STYLE = {"whole afternoon (13:30-14:59)": ("#3182bd", "-", "o"),
             "last 30 min (14:30-14:59)": ("#e6550d", "--", "s")}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, code in zip(axes.flat, CODES):
        for wlab in WINDOWS:
            n, k = res[code][wlab]
            m = n > 200
            col, ls, mk = STYLE[wlab]
            ax.plot(centers[m], (k / np.where(n > 0, n, np.nan))[m], ls=ls, marker=mk, ms=3,
                    color=col, label=wlab)
        ax.axhline(0.5, color="0.5", lw=0.8); ax.axvline(0, color="0.5", lw=0.8)
        ax.set_title(code, fontsize=12)
        ax.set_xlabel("sell-pressure imbalance (trailing 30min)", fontsize=9)
        ax.set_ylabel("P( close 15:00 > price now )", fontsize=9)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("EOD context: aggressor SELL-PRESSURE vs probability the 15:00 CLOSE is HIGHER "
                 "than now\n(all days, full history; right=more net selling; falling=selling→lower "
                 "close, rising=absorption/bounce)", fontsize=12)
    plt.tight_layout()
    plt.savefig("/Users/zhuisabella/xn/end/figs/fig_flow_prob_sweep_eod.png", dpi=140)
    print("\nsaved flow_prob_sweep_eod.csv + fig_flow_prob_sweep_eod.png")
