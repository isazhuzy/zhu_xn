"""
filter_free_pathratio.py — study the path/displacement ratio as a reversal signal
WITHOUT the up>=1% filter (user request). Instead of conditioning on a >=1% up move,
we take ALL days (both directions) and, to keep the ratio meaningful, only require a
small symmetric net-move floor |cumret(t)| >= MIN_DISP (NOT the old 1% up gate).

Reversal is defined with a SIGNED forward return so it still makes sense without the
up-direction assumption:
    sfwd(t) = fwd(t->close) * sign(cumret(t))      (+ = continuation, - = reversal)
Hypothesis (path-ratio idea): a choppier path (high ratio) -> more reversal -> negative
corr(ratio, sfwd) and noisy-group mean < smooth-group mean.

Scans 13:00-14:59, day-clustered, full history + ex-2015. Also runs a NO-FLOOR variant
to expose the ratio~1/|disp| confound. Writes ff_pathratio_summary.csv and a verdict fig.
Run on system python3 (matplotlib).
"""
import numpy as np
import pandas as pd
from common import load, build_panel, day_stats, path_ratio, CODES, tod_to_hm

MIN_DISP = 0.005                      # 0.5% net move, either direction (the meaningful floor)
AFT = list(range(13 * 60, 15 * 60))   # 13:00-14:59
THRESH_BP = 150                       # bad-tick day removal


def two_sample_t(a, b):
    a, b = a.dropna(), b.dropna()
    if len(a) < 5 or len(b) < 5:
        return np.nan
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return (a.mean() - b.mean()) / se if se > 0 else np.nan


def scan(df, min_disp):
    rows = []
    for code in CODES:
        piv = build_panel(df, code)
        steps = piv.diff(axis=1).div(piv.shift(1, axis=1)).abs() * 1e4
        piv = piv.loc[~steps.gt(THRESH_BP).any(axis=1)]
        _, _, cumret, fwd = day_stats(piv)
        ratio = path_ratio(piv)
        years = pd.Series(piv.index.year, index=piv.index)
        for t in AFT:
            if t not in cumret.columns or t not in fwd.columns or t not in ratio.columns:
                continue
            moved = cumret[t].abs() >= min_disp
            sfwd = (fwd[t] * np.sign(cumret[t])) * 1e4
            d = pd.DataFrame({"r": ratio[t], "s": sfwd, "yr": years})[moved].dropna()
            d = d[np.isfinite(d["r"])]
            if len(d) < 30:
                continue
            med = d["r"].median()
            hi, lo = d[d["r"] > med]["s"], d[d["r"] <= med]["s"]   # hi = noisy/choppy
            dex = d[d["yr"] != 2015]
            hiex, loex = dex[dex["r"] > med]["s"], dex[dex["r"] <= med]["s"]
            rows.append(dict(
                code=code, tod=t, hm=tod_to_hm(t), n=len(d),
                mean_noisy=round(hi.mean(), 2), mean_smooth=round(lo.mean(), 2),
                t_noisy_minus_smooth=round(two_sample_t(hi, lo), 2),
                corr_ratio_sfwd=round(np.corrcoef(d["r"], d["s"])[0, 1], 3),
                t_ex15=round(two_sample_t(hiex, loex), 2)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load()
    out = scan(df, MIN_DISP)
    out.to_csv("/Users/zhuisabella/xn/end/ff_pathratio_summary.csv", index=False)

    print(f"=== path-ratio as reversal signal, NO up-filter (|cumret|>={MIN_DISP:.1%} floor) ===")
    print("(negative corr / negative t = choppier path -> more reversal)\n")
    for code in CODES:
        c = out[out.code == code]
        print(f"{code}: mean corr(ratio,sfwd) full={c.corr_ratio_sfwd.mean():+.3f}  "
              f"minutes with t<-2: full={ (c.t_noisy_minus_smooth<-2).sum() }/{len(c)}  "
              f"ex15={ (c.t_ex15<-2).sum() }/{len(c)}  "
              f"mean n/min={int(c.n.mean())}")
    print("\n=== same at 14:45 ===")
    print(out[out.tod == 14 * 60 + 45][
        ["code", "n", "mean_noisy", "mean_smooth", "t_noisy_minus_smooth",
         "corr_ratio_sfwd", "t_ex15"]].to_string(index=False))

    # confound demo: no floor at all -> ratio ~ 1/|disp|
    nofloor = scan(df, 0.0)
    print("\n=== NO-FLOOR variant (exposes ratio~1/|disp| confound) mean corr(ratio,sfwd) ===")
    for code in CODES:
        print(f" {code}: {nofloor[nofloor.code==code].corr_ratio_sfwd.mean():+.3f} "
              f"(mean n/min={int(nofloor[nofloor.code==code].n.mean())})")

    # ---- verdict figure ----
    import matplotlib.pyplot as plt
    COL = {"IC0000": "tab:blue", "IF0000": "tab:orange", "IH0000": "tab:green", "IM0000": "tab:red"}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for code in CODES:
        s = out[out.code == code].sort_values("tod")
        ax1.plot(s["hm"], s["t_noisy_minus_smooth"], color=COL[code], lw=1.6, label=f"{code} full")
        ax1.plot(s["hm"], s["t_ex15"], color=COL[code], lw=1.3, ls="--", alpha=0.8)
        ax2.plot(s["hm"], s["corr_ratio_sfwd"], color=COL[code], lw=1.6, label=code)
    ax1.axhspan(-2, 2, color="gray", alpha=0.12); ax1.axhline(0, color="k", lw=0.8)
    for y in (-2, 2):
        ax1.axhline(y, color="gray", lw=0.7, ls=":")
    ax1.set_ylabel("t: noisy − smooth signed-fwd\n(negative = choppy → reversal)", fontsize=10)
    ax1.set_title(f"Path/displacement ratio as a reversal signal — NO up-filter "
                  f"(|net move|≥{MIN_DISP:.1%}, both directions)\n"
                  f"signed fwd→15:00 close; solid=full history, dashed=ex-2015; shaded |t|<2", fontsize=12)
    ax1.legend(fontsize=8, ncol=4, loc="upper left"); ax1.grid(True, alpha=0.3)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_ylabel("corr(ratio, signed fwd→close)", fontsize=10)
    ax2.set_xlabel("signal minute (HH:MM)", fontsize=11)
    ax2.legend(fontsize=8, ncol=4, loc="upper left"); ax2.grid(True, alpha=0.3)
    lab = out[out.code == "IC0000"].sort_values("tod")["hm"].tolist()
    ticks = [i for i in range(len(lab)) if i % 5 == 0]
    ax2.set_xticks(ticks); ax2.set_xticklabels([lab[i] for i in ticks], rotation=45)
    plt.tight_layout()
    plt.savefig("/Users/zhuisabella/xn/end/figs/fig_ff_pathratio_tstat.png", dpi=140)
    print("\nsaved ff_pathratio_summary.csv + fig_ff_pathratio_tstat.png")
