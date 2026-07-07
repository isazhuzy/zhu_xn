"""
oi_analysis.py — the cleanest confirmation of the profit-taking hypothesis, using real
minute-level OPEN INTEREST (oi_minutes.csv, 2025-01..2026-07, dominant contract per day).

"真平仓 vs 换手": aggressor selling can be absorbed (churn) without any net position change.
OPEN INTEREST falling = positions actually being CLOSED. So we use recent OI change as the
direct liquidation signal, with NO look-ahead:
    liq(t) = -(oi(t) - oi(t-W)) / oi(t-W)      (>0 = recent net position CLOSING)
Symmetric reversal test (accounts for price decrease):
    up>=1%   : does recent liquidation -> weaker close (reversal down)?
    down<=-1%: does recent liquidation -> bounce (reversal up)?
Unified via signed fwd:  sfwd(t) = fwd(t->close)*sign(cumret(t))  (+continuation, -reversal)
Hypothesis: more liquidation -> more reversal -> negative corr(liq, sfwd).
Also prints the descriptive OI-into-close path (do positions really shrink before 15:00?).
Run on system python3.
"""
import numpy as np
import pandas as pd

IN = "/Users/zhuisabella/xn/end/oi_minutes.csv"
FAM = ["IC", "IF", "IH", "IM"]
NAME = {"IC": "IC0000", "IF": "IF0000", "IH": "IH0000", "IM": "IM0000"}
COL = {"IC": "tab:blue", "IF": "tab:orange", "IH": "tab:green", "IM": "tab:red"}
W = 30
AFT = list(range(13 * 60, 15 * 60))


def two_sample_t(a, b):
    a, b = a.dropna(), b.dropna()
    if len(a) < 5 or len(b) < 5:
        return np.nan
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return (a.mean() - b.mean()) / se if se > 0 else np.nan


def trailing_change(x, w):
    """oi(t) - oi(t-w), positional over sorted trading minutes, per day (row)."""
    return x - x.T.shift(w).T


def dominant_panels(df, fam):
    sub = df[df["code_init"] == fam].copy()
    # dominant contract per day = the one with the largest end-of-day cumulative volume
    eod = sub.groupby(["d", "code"])["tvol"].max().reset_index()
    dom = eod.loc[eod.groupby("d")["tvol"].idxmax(), ["d", "code"]]
    sub = sub.merge(dom, on=["d", "code"])          # keep only dominant-contract rows
    close = sub.pivot_table(index="d", columns="tod", values="close").sort_index()
    oi = sub.pivot_table(index="d", columns="tod", values="oi").sort_index()
    return close, oi


if __name__ == "__main__":
    df = pd.read_csv(IN)
    df["d"] = pd.to_datetime(df["d"])
    mn = pd.to_datetime(df["mn"])
    df["tod"] = mn.dt.hour * 60 + mn.dt.minute

    rows, desc = [], []
    for fam in FAM:
        close, oi = dominant_panels(df, fam)
        open_ = close[570] if 570 in close.columns else close[min(close.columns)]
        close900 = close[900] if 900 in close.columns else close[max(close.columns)]
        cumret = close.div(open_, axis=0) - 1.0
        fwd = pd.DataFrame(close900.to_numpy()[:, None] / close.to_numpy() - 1.0,
                           index=close.index, columns=close.columns)
        liq = -trailing_change(oi, W).div(oi.T.shift(W).T)   # >0 = recent net closing

        # descriptive: mean OI change from 13:00 to 15:00 on up vs down days (context)
        if 780 in oi.columns and 900 in oi.columns:
            up = cumret[780] >= 0.01
            dn = cumret[780] <= -0.01
            doi = (oi[900] / oi[780] - 1.0) * 100
            desc.append(dict(fam=fam, up_days=int(up.sum()), dn_days=int(dn.sum()),
                             oi_chg_up_pm=round(doi[up].mean(), 2),
                             oi_chg_dn_pm=round(doi[dn].mean(), 2),
                             oi_chg_all_pm=round(doi.mean(), 2)))

        for side, sgn in [("up", +1), ("down", -1)]:
            for t in AFT:
                if t not in cumret.columns or t not in fwd.columns or t not in liq.columns:
                    continue
                ev = (cumret[t] >= 0.01) if sgn > 0 else (cumret[t] <= -0.01)
                sfwd = (fwd[t] * sgn) * 1e4
                d = pd.DataFrame({"l": liq[t], "s": sfwd})[ev].dropna()
                d = d[np.isfinite(d["l"])]
                if len(d) < 15:
                    continue
                med = d["l"].median()
                hi, lo = d[d["l"] > med]["s"], d[d["l"] <= med]["s"]   # hi = more liquidation
                rows.append(dict(
                    fam=fam, side=side, tod=t, hm=f"{t//60:02d}:{t%60:02d}", n=len(d),
                    mean_hiLiq=round(hi.mean(), 2), mean_loLiq=round(lo.mean(), 2),
                    t_hi_minus_lo=round(two_sample_t(hi, lo), 2),
                    corr_liq_sfwd=round(np.corrcoef(d["l"], d["s"])[0, 1], 3)))
    out = pd.DataFrame(rows)
    out.to_csv("/Users/zhuisabella/xn/end/oi_summary.csv", index=False)

    print("=== descriptive: OI change 13:00->15:00 (%), does OI really shrink into close? ===")
    print(pd.DataFrame(desc).to_string(index=False))
    for side in ["up", "down"]:
        print(f"\n=== {side.upper()} days: does recent LIQUIDATION -> reversal? "
              f"(neg corr/t = liquidation predicts reversal) ===")
        for fam in FAM:
            c = out[(out.fam == fam) & (out.side == side)]
            if len(c):
                print(f" {NAME[fam]}: mean corr={c.corr_liq_sfwd.mean():+.3f}  "
                      f"min t(hi-lo)={c.t_hi_minus_lo.min():+.2f}  "
                      f"minutes t<-2: {(c.t_hi_minus_lo<-2).sum()}/{len(c)}  mean n/min={int(c.n.mean())}")

    # ---- figure ----
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for ax, side, ttl in [(axes[0], "up", "UP days (up≥1%)"),
                          (axes[1], "down", "DOWN days (down≤−1%)")]:
        for fam in FAM:
            s = out[(out.fam == fam) & (out.side == side)].sort_values("tod")
            ax.plot(s["hm"], s["corr_liq_sfwd"], color=COL[fam], lw=1.6, label=NAME[fam])
        ax.axhline(0, color="k", lw=0.8); ax.grid(True, alpha=0.3)
        ax.set_ylabel("corr(liquidation, signed fwd→close)", fontsize=9)
        ax.set_title(f"{ttl}   —   negative = OI-drop (real position closing) → reversal", fontsize=11)
        ax.legend(fontsize=8, ncol=4, loc="upper left")
    any_side = out[(out.fam == "IC") & (out.side == "up")].sort_values("tod")
    lab = any_side["hm"].tolist()
    ticks = [i for i in range(len(lab)) if i % 5 == 0]
    axes[1].set_xticks(ticks); axes[1].set_xticklabels([lab[i] for i in ticks], rotation=45)
    axes[1].set_xlabel("signal minute (HH:MM)", fontsize=11)
    fig.suptitle(f"Real open-interest confirmation (2025-01..2026-07, dominant contract): does recent "
                 f"OI decline\n(trailing {W}min, genuine position closing) predict a reversal into the "
                 f"15:00 close?", fontsize=12)
    plt.tight_layout()
    plt.savefig("/Users/zhuisabella/xn/end/figs/fig_oi_confirm.png", dpi=140)
    print("\nsaved oi_summary.csv + fig_oi_confirm.png")
