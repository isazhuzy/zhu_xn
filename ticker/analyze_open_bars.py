"""
analyze_open_bars.py — first-two-minute stats from a server-aggregated open-bars
CSV (columns: code, d, mn, close) produced by fetch_open_bars.py.

Reproduces the matrix.py momentum/contrarian methodology on minutes 09:31 & 09:32:
  09:31: signal = close(9:31)-close(9:30), paid close(9:32)/close(9:31)-1
  09:32: signal = close(9:32)-close(9:31), paid close(9:33)/close(9:32)-1
Contrarian = take the opposite of the (losing) momentum sign.

Outputs the contrarian trade table (n, win_rate, avg_bp, total_bp, t) and the
threshold-free reversal regression (m2~m1).  Use --by-year to add a yearly split.
"""
import sys
import numpy as np
import pandas as pd

TICK = 0.2
THRESHOLDS = [5, 10, 15, 20, 25, 30]
PRODUCTS = ["IC0000", "IF0000", "IH0000", "IM0000"]


def load_wide(path):
    df = pd.read_csv(path)
    df["d"] = pd.to_datetime(df["d"])
    df["hm"] = pd.to_datetime(df["mn"]).dt.strftime("%H:%M")
    w = df.pivot_table(index=["code", "d"], columns="hm", values="close").reset_index()
    w = w.rename(columns={"09:30": "c30", "09:31": "c31", "09:32": "c32", "09:33": "c33"})
    return w


def clean(w, cap_bp=150):
    """drop glitch days: any adjacent open-minute move beyond cap_bp (e.g. the
    2024-10-08 IH bad tick where the 9:31 mid halved). Returns filtered copy."""
    mv = pd.concat([(w["c31"] / w["c30"] - 1), (w["c32"] / w["c31"] - 1),
                    (w["c33"] / w["c32"] - 1)], axis=1).abs().max(axis=1) * 1e4
    bad = mv > cap_bp
    if bad.sum():
        print(f"[clean] dropping {int(bad.sum())} glitch day-rows (|move|>{cap_bp}bp)")
    return w[~bad].copy()


def momentum_at(w, minute, thr):
    """per-day momentum return at `minute`, threshold in ticks; drop flats."""
    if minute == "09:31":
        move, fwd = w["c31"] - w["c30"], w["c32"] / w["c31"] - 1.0
    else:  # 09:32
        move, fwd = w["c32"] - w["c31"], w["c33"] / w["c32"] - 1.0
    keep = move.abs() > thr * TICK
    r = np.where(keep, np.sign(move) * fwd, 0.0)
    s = pd.Series(r, index=w["d"]).replace(0, np.nan).dropna()
    return s.sort_index()


def contrarian_row(s):
    c = -s
    n = len(c)
    t = c.mean() / (c.std(ddof=1) / np.sqrt(n)) if n > 1 and c.std(ddof=1) > 0 else np.nan
    return dict(n=n, win=round((c > 0).mean(), 3), avg_bp=round(c.mean() * 1e4, 2),
                total_bp=round(c.sum() * 1e4, 1), t=round(t, 2))


def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y)); x, y = x[m], y[m]
    n = len(x)
    b1, b0 = np.polyfit(x, y, 1)
    res = y - (b0 + b1 * x)
    r2 = 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()
    se = np.sqrt((res ** 2).sum() / (n - 2)) / np.sqrt(((x - x.mean()) ** 2).sum())
    return dict(n=n, beta=round(b1, 3), t=round(b1 / se, 2), r2=round(r2, 3),
                corr=round(np.corrcoef(x, y)[0, 1], 3))


def run(w, label=""):
    pd.set_option("display.width", 200, "display.max_rows", 400)
    print(f"\n########## {label}  (n_days per product) ##########")
    print({p: int(w[w.code == p]["c30"].notna().sum()) for p in PRODUCTS})

    rows = []
    for p in PRODUCTS:
        wp = w[w.code == p]
        for minute in ["09:31", "09:32"]:
            for thr in THRESHOLDS:
                s = momentum_at(wp, minute, thr)
                if len(s):
                    rows.append({"product": p, "minute": minute, "thr": thr, **contrarian_row(s)})
    tab = pd.DataFrame(rows)
    for minute in ["09:31", "09:32"]:
        print(f"\n=== CONTRARIAN (fade-the-open) @ {minute} ===")
        print(tab[tab.minute == minute].drop(columns="minute").to_string(index=False))

    print("\n=== Reversal regression  m2 ~ m1   (m_i in bp; beta<0 => reversal) ===")
    rr = []
    for p in PRODUCTS:
        wp = w[w.code == p].dropna(subset=["c30", "c31", "c32"])
        m1 = (wp["c31"] / wp["c30"] - 1) * 1e4
        m2 = (wp["c32"] / wp["c31"] - 1) * 1e4
        rr.append({"product": p, **ols(m1, m2)})
    print(pd.DataFrame(rr).to_string(index=False))
    return tab


def by_year_summary(w, minute="09:31", thr=10):
    """compact yearly stability of one contrarian config."""
    print(f"\n=== BY-YEAR contrarian stability @ {minute}, {thr} ticks "
          f"(win | avg_bp | t) ===")
    rows = []
    for yr, wy in w.groupby(w["d"].dt.year):
        rec = {"year": yr}
        for p in PRODUCTS:
            s = momentum_at(wy[wy.code == p], minute, thr)
            if len(s) >= 5:
                c = contrarian_row(s)  # s is momentum; contrarian_row negates inside
                rec[p] = f"{c['win']:.2f}/{c['avg_bp']:+.1f}/{c['t']:+.1f}"
            else:
                rec[p] = "."
        rows.append(rec)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/open_bars_2023.csv"
    by_year = "--by-year" in sys.argv
    w = load_wide(path)
    w = w[w["d"].dt.month != 3] if "--keep-march" not in sys.argv else w
    if "--cap" in sys.argv:
        w = clean(w, float(sys.argv[sys.argv.index("--cap") + 1]))
    run(w, label=f"ALL  {path}")
    if by_year:
        by_year_summary(w, "09:31", 10)
        by_year_summary(w, "09:32", 10)
