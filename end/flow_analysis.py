"""
flow_analysis.py — test the "涨到一定程度就日内平仓" (profit-taking) hypothesis DIRECTLY,
using aggressor order flow instead of the (confounded) path/displacement ratio.

Feature (real-time, no look-ahead): over a trailing W-minute window ending at t, the
net SELLING-PRESSURE imbalance
      imb(t) = (sell_W - buy_W) / (sell_W + buy_W)
where buy/sell are aggressor-initiated volumes. imb>0 = net aggressive selling (the
footprint of longs hitting the bid to flatten). Semantics of the two Act* columns are
resolved empirically (sign of corr(minute return, actbid-actask)).

Event test (day-clustered): on days up>=1% from open at minute t, does high sell-pressure
predict a weaker forward return to the 15:00 close?  Split up-days at the median imb(t),
compare mean fwd(t->close); scan 13:00-14:59; check per-year and pooled ex-2015.
Also: continuous corr(imb(t), fwd(t)) within up-days.

Writes flow_summary.csv (+ prints semantics + verdict). Run on .venv or system python3.
"""
import numpy as np
import pandas as pd
from common import day_stats, CODES

IN = "/Users/zhuisabella/xn/end/day_minutes_flow_full.csv"
W = 30                       # trailing window (minutes) for the flow imbalance
WINDOW_START = 14 * 60 + 45  # 14:45 (last-15-min focus, mirrors the clean_* figs)
AFT = list(range(13 * 60, 15 * 60))  # 13:00-14:59 scan


def load_flow():
    df = pd.read_csv(IN)
    df["d"] = pd.to_datetime(df["d"])
    mn = pd.to_datetime(df["mn"])
    df["tod"] = mn.dt.hour * 60 + mn.dt.minute
    df["year"] = df["d"].dt.year
    return df


def panels(df, code):
    sub = df[df["code"] == code]
    close = sub.pivot_table(index="d", columns="tod", values="close").sort_index()
    actbid = sub.pivot_table(index="d", columns="tod", values="actbid").sort_index()
    actask = sub.pivot_table(index="d", columns="tod", values="actask").sort_index()
    return close, actbid, actask


def trailing_sum(x, w):
    """sum over the last w available (positional) trading minutes, per day (row)."""
    return x.T.rolling(w, min_periods=1).sum().T


def two_sample_t(a, b):
    a, b = a.dropna(), b.dropna()
    if len(a) < 5 or len(b) < 5:
        return np.nan
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / len(a) + vb / len(b))
    return (a.mean() - b.mean()) / se if se > 0 else np.nan


if __name__ == "__main__":
    df = load_flow()

    # ---- resolve Act* semantics globally: which column is buyer-initiated? ----
    corr_num = 0.0
    for code in CODES:
        close, ab, aa = panels(df, code)
        ret = close.diff(axis=1)                       # minute-to-minute mid change
        sig = (ab - aa)
        m = ret.notna() & sig.notna()
        if m.to_numpy().sum() > 1000:
            corr_num += np.corrcoef(ret.to_numpy()[m.to_numpy()],
                                    sig.to_numpy()[m.to_numpy()])[0, 1]
    actbid_is_buy = corr_num > 0    # if (actbid-actask) moves WITH price, actbid=buyer-init
    print(f"semantics: corr(ret, actbid-actask) sum over contracts = {corr_num:.3f} "
          f"=> actbid is {'BUYER' if actbid_is_buy else 'SELLER'}-initiated\n")

    rows = []
    for code in CODES:
        close, ab, aa = panels(df, code)
        buy = ab if actbid_is_buy else aa
        sell = aa if actbid_is_buy else ab
        _, _, cumret, fwd = day_stats(close)
        buyW, sellW = trailing_sum(buy, W), trailing_sum(sell, W)
        imb = (sellW - buyW) / (sellW + buyW)          # >0 = net aggressive selling

        years = pd.Series(close.index.year, index=close.index)
        for t in AFT:
            if t not in cumret.columns or t not in fwd.columns or t not in imb.columns:
                continue
            up = cumret[t] >= 0.01
            y = (fwd[t] * 1e4)[up]
            x = imb[t][up]
            d = pd.DataFrame({"y": y, "x": x, "yr": years[up]}).dropna()
            if len(d) < 20:
                continue
            med = d["x"].median()
            hi = d[d["x"] > med]["y"]     # high sell-pressure days
            lo = d[d["x"] <= med]["y"]    # low sell-pressure days
            d_ex = d[d["yr"] != 2015]
            hi_ex = d_ex[d_ex["x"] > med]["y"]; lo_ex = d_ex[d_ex["x"] <= med]["y"]
            rr = np.corrcoef(d["x"], d["y"])[0, 1] if len(d) > 5 else np.nan
            rows.append(dict(
                code=code, tod=t, hm=f"{t//60:02d}:{t%60:02d}", n=len(d),
                mean_hi=round(hi.mean(), 2), mean_lo=round(lo.mean(), 2),
                t_hi_minus_lo=round(two_sample_t(hi, lo), 2),
                corr_imb_fwd=round(rr, 3),
                t_hi_minus_lo_ex15=round(two_sample_t(hi_ex, lo_ex), 2),
                mean_hi_ex15=round(hi_ex.mean(), 2), n_ex15=len(d_ex)))
    out = pd.DataFrame(rows)
    out.to_csv("/Users/zhuisabella/xn/end/flow_summary.csv", index=False)

    # ---- verdict print: the last-15-min signal minute + the strongest afternoon minute ----
    print("=== at 14:45 (last-15-min trigger) ===")
    print(out[out.tod == WINDOW_START][
        ["code", "hm", "n", "mean_hi", "mean_lo", "t_hi_minus_lo",
         "corr_imb_fwd", "t_hi_minus_lo_ex15"]].to_string(index=False))
    print("\n=== strongest reversal minute per contract (most negative hi-lo t) ===")
    for code in CODES:
        c = out[out.code == code]
        if len(c):
            b = c.loc[c["t_hi_minus_lo"].idxmin()]
            print(f" {code} {b.hm}: hi={b.mean_hi}bp lo={b.mean_lo}bp "
                  f"t(hi-lo)={b.t_hi_minus_lo} (ex15 {b.t_hi_minus_lo_ex15}) "
                  f"corr={b.corr_imb_fwd} n={int(b.n)}")
    print("\nsaved flow_summary.csv")
