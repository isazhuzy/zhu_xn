"""
flow_symmetric.py — symmetric version of the aggressor-flow reversal test, covering BOTH
price increases and decreases (user request "account for price decrease").

UP days   (cumret(t) >= +1%): does trailing SELL-pressure (profit-taking) -> weaker close?
DOWN days (cumret(t) <= -1%): does trailing  BUY-pressure (short-cover / bargain) -> bounce?

Unified with a SIGNED move: let dir = sign(cumret(t)). "Pressure AGAINST the move" is the
liquidation footprint on either side:
    against(t) = dir * (buy_W - sell_W) / (buy_W + sell_W)
       up-day (dir=+1): against>0 means net BUYING (continuation flow); against<0 = net SELLING
       down-day(dir=-1): sign flips so against<0 = net BUYING into the drop = counter-move flow
We test the SIGNED forward return sfwd(t) = fwd(t)*dir  (+=continuation, -=reversal) against
the "counter-pressure" = -against (i.e. flow opposing the current move). Hypothesis: more
counter-pressure (profit-taking / covering) -> more reversal -> negative corr.

Reuses day_minutes_flow_full.csv. Day-clustered, full history + ex-2015. Writes csv + fig.
Run on system python3.
"""
import numpy as np
import pandas as pd
from common import day_stats, CODES, tod_to_hm
from flow_analysis import load_flow, panels, trailing_sum, W, two_sample_t

AFT = list(range(13 * 60, 15 * 60))
THRESH_BP = 150


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

    rows = []
    for code in CODES:
        close, ab, aa = panels(df, code)
        buy = ab if actbid_is_buy else aa
        sell = aa if actbid_is_buy else ab
        steps = close.diff(axis=1).div(close.shift(1, axis=1)).abs() * 1e4
        keep = ~steps.gt(THRESH_BP).any(axis=1)
        close, buy, sell = close.loc[keep], buy.loc[keep], sell.loc[keep]
        _, _, cumret, fwd = day_stats(close)
        buyW, sellW = trailing_sum(buy, W), trailing_sum(sell, W)
        years = pd.Series(close.index.year, index=close.index)

        for side, sign_sel in [("up", +1), ("down", -1)]:
            for t in AFT:
                if t not in cumret.columns or t not in fwd.columns:
                    continue
                if sign_sel > 0:
                    ev = cumret[t] >= 0.01
                else:
                    ev = cumret[t] <= -0.01
                # counter-pressure = aggressor flow OPPOSING the current move direction
                # up-move: opposing = selling; down-move: opposing = buying
                counter = (sellW[t] - buyW[t]) / (sellW[t] + buyW[t])
                counter = counter * sign_sel        # >0 = flow against the move (liquidation)
                sfwd = (fwd[t] * sign_sel) * 1e4     # signed: + continuation, - reversal
                d = pd.DataFrame({"c": counter, "s": sfwd, "yr": years})[ev].dropna()
                if len(d) < 20:
                    continue
                med = d["c"].median()
                hi, lo = d[d["c"] > med]["s"], d[d["c"] <= med]["s"]   # hi = high counter-pressure
                dex = d[d["yr"] != 2015]
                hiex, loex = dex[dex["c"] > med]["s"], dex[dex["c"] <= med]["s"]
                rows.append(dict(
                    code=code, side=side, tod=t, hm=tod_to_hm(t), n=len(d),
                    mean_hicounter=round(hi.mean(), 2), mean_locounter=round(lo.mean(), 2),
                    t_hi_minus_lo=round(two_sample_t(hi, lo), 2),
                    corr_counter_sfwd=round(np.corrcoef(d["c"], d["s"])[0, 1], 3),
                    t_ex15=round(two_sample_t(hiex, loex), 2)))
    out = pd.DataFrame(rows)
    out.to_csv("/Users/zhuisabella/xn/end/flow_symmetric_summary.csv", index=False)

    for side in ["up", "down"]:
        print(f"=== {side.upper()} days: counter-pressure -> reversal? "
              f"(neg corr/t = liquidation flow -> reversal) ===")
        for code in CODES:
            c = out[(out.code == code) & (out.side == side)]
            if len(c):
                print(f" {code}: mean corr={c.corr_counter_sfwd.mean():+.3f}  "
                      f"min t(hi-lo) full={c.t_hi_minus_lo.min():+.2f} ex15={c.t_ex15.min():+.2f}  "
                      f"mean n/min={int(c.n.mean())}")
        print()

    # ---- figure: 2 rows (up / down), corr(counter-pressure, signed fwd) across afternoon ----
    import matplotlib.pyplot as plt
    COL = {"IC0000": "tab:blue", "IF0000": "tab:orange", "IH0000": "tab:green", "IM0000": "tab:red"}
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for ax, side, ttl in [(axes[0], "up", "UP days (up≥1%): SELL-pressure = counter"),
                          (axes[1], "down", "DOWN days (down≤−1%): BUY-pressure = counter")]:
        for code in CODES:
            s = out[(out.code == code) & (out.side == side)].sort_values("tod")
            ax.plot(s["hm"], s["corr_counter_sfwd"], color=COL[code], lw=1.6, label=code)
        ax.axhline(0, color="k", lw=0.8); ax.grid(True, alpha=0.3)
        ax.set_ylabel("corr(counter-pressure, signed fwd→close)", fontsize=9)
        ax.set_title(ttl + "   —   negative = liquidation flow → reversal", fontsize=11)
        ax.legend(fontsize=8, ncol=4, loc="upper left")
    lab = out[(out.code == "IC0000") & (out.side == "up")].sort_values("tod")["hm"].tolist()
    ticks = [i for i in range(len(lab)) if i % 5 == 0]
    axes[1].set_xticks(ticks); axes[1].set_xticklabels([lab[i] for i in ticks], rotation=45)
    axes[1].set_xlabel("signal minute (HH:MM)", fontsize=11)
    fig.suptitle("Symmetric aggressor-flow reversal test — does liquidation flow (against the "
                 "move) predict a reversal into the 15:00 close?", fontsize=12)
    plt.tight_layout()
    plt.savefig("/Users/zhuisabella/xn/end/figs/fig_flow_symmetric.png", dpi=140)
    print("saved flow_symmetric_summary.csv + fig_flow_symmetric.png")
