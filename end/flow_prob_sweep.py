"""
flow_prob_sweep.py — sweep the aggressor SELL-PRESSURE imbalance across its full range and
plot P(price goes UP over the next h minutes), instead of the median hi/lo split used in
fig_flow_sellpressure_path. This is a fig71-style probability curve for the flow signal.

Feature: imb(t) = (sell_W - buy_W)/(sell_W + buy_W), trailing W=30min, afternoon-only so the
window never spans the lunch gap. Pooled over 13:31-14:59 and ALL days (bad-tick days removed),
full history. Target: sign(close(t+h) - close(t)) for h in {1,5,30} min (ties dropped).
Bin imb into NB bins; P(up) = n_up/n per bin. Writes csv + fig (2x2 contracts).
Run on system python3.
"""
import numpy as np
import pandas as pd
from common import CODES
from flow_analysis import load_flow, panels

W = 30
NB = 30
HOR = [1, 5, 30]                 # forward horizons in minutes
THRESH_BP = 150
AFT0, AFT1 = 13 * 60, 15 * 60    # afternoon 13:00-15:00
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
    print(f"actbid is {'BUYER' if actbid_is_buy else 'SELLER'}-initiated (buy=aggressor buy)\n")

    res = {}
    for code in CODES:
        close, ab, aa = panels(df, code)
        buy = ab if actbid_is_buy else aa
        sell = aa if actbid_is_buy else ab
        # remove bad-tick days (session-wide 150bp single-minute move)
        steps = close.diff(axis=1).div(close.shift(1, axis=1)).abs() * 1e4
        keep = ~steps.gt(THRESH_BP).any(axis=1)
        close, buy, sell = close.loc[keep], buy.loc[keep], sell.loc[keep]
        # afternoon-only columns so the trailing window never crosses lunch
        acols = [c for c in sorted(close.columns) if AFT0 <= c <= AFT1]
        close_a, buy_a, sell_a = close[acols], buy[acols], sell[acols]
        buyW = buy_a.T.rolling(W, min_periods=W).sum().T
        sellW = sell_a.T.rolling(W, min_periods=W).sum().T
        imb = (sellW - buyW) / (sellW + buyW)

        res[code] = {}
        for h in HOR:
            fwd = close_a.T.shift(-h).T - close_a       # close(t+h) - close(t), within afternoon
            valid = imb.notna() & fwd.notna() & (fwd != 0)
            x = imb.to_numpy()[valid.to_numpy()]
            up = (fwd.to_numpy()[valid.to_numpy()] > 0).astype(float)
            b = np.clip(((x + 1) / 2 * NB).astype(int), 0, NB - 1)
            n = np.bincount(b, minlength=NB).astype(float)
            k = np.bincount(b, weights=up, minlength=NB)
            res[code][h] = (n, k)

    # save + print monotonic summary
    rows = []
    for code in CODES:
        for h in HOR:
            n, k = res[code][h]
            for i in range(NB):
                if n[i] > 0:
                    rows.append(dict(code=code, h=h, imb=round(centers[i], 3),
                                     n=int(n[i]), p_up=round(k[i] / n[i], 4)))
    pd.DataFrame(rows).to_csv("/Users/zhuisabella/xn/end/flow_prob_sweep.csv", index=False)
    print("=== P(up) at low vs high sell-pressure (h=5min), + Spearman-ish slope sign ===")
    for code in CODES:
        n, k = res[code][5]
        p = k / np.where(n > 0, n, np.nan)
        loP = np.nanmean(p[:NB // 3]); hiP = np.nanmean(p[-NB // 3:])
        print(f" {code}: P(up) low-sell={loP:.3f}  high-sell={hiP:.3f}  diff(high-low)={hiP-loP:+.3f}")

    # ---- figure ----
    import matplotlib.pyplot as plt
    COL = {1: "#2c7fb8", 5: "#41ab5d", 30: "#e6550d"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, code in zip(axes.flat, CODES):
        for h in HOR:
            n, k = res[code][h]
            m = n > 200
            ax.plot(centers[m], (k / np.where(n > 0, n, np.nan))[m], marker="o", ms=3,
                    color=COL[h], label=f"next {h} min")
        ax.axhline(0.5, color="0.5", lw=0.8); ax.axvline(0, color="0.5", lw=0.8)
        ax.set_title(code, fontsize=12)
        ax.set_xlabel("sell-pressure imbalance  (sell−buy)/(sell+buy), trailing 30min", fontsize=9)
        ax.set_ylabel("P( price UP over next h min )", fontsize=9)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Sweep of aggressor SELL-PRESSURE vs probability of a price INCREASE "
                 "(afternoon, all days, full history)\n"
                 "right = more net selling; falling curve = selling predicts down (momentum), "
                 "flat/rising = absorption", fontsize=12)
    plt.tight_layout()
    plt.savefig("/Users/zhuisabella/xn/end/figs/fig_flow_prob_sweep.png", dpi=140)
    print("\nsaved flow_prob_sweep.csv + fig_flow_prob_sweep.png")
