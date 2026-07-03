"""voi_backtest.py — net-of-fees backtest of the Shen (2015) threshold strategy.
Closes the loop on paper #3: we know the forecast has OOS R²≈1.5% and 60%+ hit rate
on strong signals — this script answers "how far inside the spread is the edge, in
ticks and CNY?"

Strategy (per contract, test period 2025-01..2026-05, signals from FROZEN 2020-24
model-B k=20 coefficients in voi_coefs.csv):
  - at snapshot t compute yhat (predicted avg mid change over next 20 snapshots, ticks)
  - if |yhat| > q  and flat -> open 1 lot in direction sign(yhat)
  - close exactly k=20 snapshots later (same session only); non-overlapping (no
    re-entry while a position is open)
Execution/cost tiers per trade (each answered separately):
  mid0    : enter & exit at the mid of the signal snapshot / exit snapshot (signal value)
  taker0  : cross the spread, same snapshot (buy@ask sell@bid) — optimistic latency
  taker1  : cross the spread at snapshot t+1 — realistic latency (~500ms)
  fee_yz  : taker1 + commission 0.23bp open + 0.23bp close-yesterday (平昨) = 0.46bp RT
  fee_jt  : taker1 + commission 0.23bp open + 3.45bp close-today   (平今) = 3.68bp RT
  (rates are the current CFFEX schedule order-of-magnitude; constants below)
Outputs: bt_results[_pilot].csv  (code, q, tier, n, avg_ticks, hit, tstat)
         bt_daily[_pilot].csv    (code, day, pnl_cny at q=0.5, tier=fee_yz)
Run: PILOT=1 /Users/zhuisabella/xn/.venv/bin/python voi_backtest.py   (sandbox OFF)
"""
import calendar
import os
import numpy as np, pandas as pd, dolphindb as ddb
from ddb_config import HOST, PORT, USER, PW
from lob_common import CODES, MULT, TICK, fetch_l1, prep_l1
from voi_ddb import build, design, varnames, NLAG

PILOT = os.environ.get("PILOT") == "1"
SUF = "_pilot" if PILOT else ""
D = "/Users/zhuisabella/xn/prediction"
K = 20                                     # holding = forecast horizon (10s)
QS = [0.1, 0.2, 0.5, 1.0, 2.0]             # signal thresholds, ticks
FEE_OPEN, FEE_YZ, FEE_JT = 0.23e-4, 0.23e-4, 3.45e-4   # of notional, per side
MONTHS = [(2024, 7)] if PILOT else [(y, m) for y in range(2025, 2027) for m in range(1, 13)
                                    if (2025, 1) <= (y, m) <= (2026, 5)]


def load_betas():
    cf = pd.read_csv(f"{D}/voi_coefs{SUF}.csv")
    cf = cf[(cf.model == "B") & (cf.k == K)]
    order = varnames("B")
    out = {}
    for code in CODES:
        s = cf[cf.code == code].set_index("var")["beta"]
        out[code] = np.array([s[v] for v in order])
    return out


def trades_for(yhat, last_idx, q):
    """greedy non-overlapping entries: returns entry rows i (exit = i+K, same gid)."""
    fired = np.where(np.isfinite(yhat) & (np.abs(yhat) > q))[0]
    entries = []
    nxt = 0
    while nxt < len(fired):
        i = fired[nxt]
        if i + K <= last_idx[i]:
            entries.append(i)
            nxt = np.searchsorted(fired, i + K)
        else:
            nxt = np.searchsorted(fired, last_idx[i] + 1)
    return np.asarray(entries, dtype=int)


if __name__ == "__main__":
    betas = load_betas()
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    acc = {}                    # (code,q,tier) -> [sum, sumsq, n, nhit]
    daily = {}                  # (code, day) -> pnl CNY  (q=0.5, tier=fee_yz)
    for yr, mo in MONTHS:
        last = calendar.monthrange(yr, mo)[1]
        for code in CODES:
            df = fetch_l1(sess, code, f"{yr}.{mo:02d}.01", f"{yr}.{mo:02d}.{last:02d}")
            if not len(df):
                continue
            df = prep_l1(df)
            if df.empty:
                continue
            df = build(df, code)
            need = [f"voi{l}" for l in range(NLAG)] + [f"oir{l}" for l in range(NLAG)] + ["mpb"]
            ok = df[need].notna().all(axis=1).to_numpy()
            X = design(df, "B")
            yhat = np.where(ok, X @ betas[code], np.nan)
            gid = df["gid"].to_numpy()
            # last row index of each session block (gid is monotone within a month)
            chg = np.r_[np.where(gid[1:] != gid[:-1])[0], len(gid) - 1]
            sizes = np.diff(np.r_[-1, chg])
            last_idx = np.repeat(chg, sizes)
            pb, pa = df.pb.to_numpy(), df.pa.to_numpy()
            mid = df.mid_tk.to_numpy()
            day = df["ts"].dt.normalize().to_numpy()
            mult = MULT[code[:2]]
            for q in QS:
                i = trades_for(yhat, last_idx, q)
                if not len(i):
                    continue
                j = i + K
                side = np.sign(yhat[i])                       # +1 long, -1 short
                pnl_mid = side * (mid[j] - mid[i])            # ticks
                # taker prices: long buys ask sells bid; short mirror
                p_in0 = np.where(side > 0, pa[i], pb[i])
                p_in1 = np.where(side > 0, pa[i + 1], pb[i + 1])   # i+1<=j<=last_idx
                p_out = np.where(side > 0, pb[j], pa[j])
                pnl_tk0 = side * (p_out - p_in0) / TICK
                pnl_tk1 = side * (p_out - p_in1) / TICK
                fee_tk = lambda rt: (p_in1 + p_out) * rt / TICK    # per-side rate rt
                pnl_yz = pnl_tk1 - fee_tk((FEE_OPEN + FEE_YZ) / 2)
                pnl_jt = pnl_tk1 - fee_tk((FEE_OPEN + FEE_JT) / 2)
                for tier, v in [("mid0", pnl_mid), ("taker0", pnl_tk0), ("taker1", pnl_tk1),
                                ("fee_yz", pnl_yz), ("fee_jt", pnl_jt)]:
                    a = acc.setdefault((code, q, tier), [0.0, 0.0, 0, 0])
                    a[0] += v.sum(); a[1] += (v * v).sum(); a[2] += len(v)
                    a[3] += int((pnl_mid > 0).sum())          # hit on mid-to-mid
                if q == 0.5:
                    cny = pnl_yz * TICK * mult
                    for d_, p_ in zip(day[i], cny):
                        daily[(code, d_)] = daily.get((code, d_), 0.0) + p_
        print(f"{yr}-{mo:02d} done", flush=True)
    sess.close()

    rows = []
    for (code, q, tier), (s, ss, n, nh) in sorted(acc.items()):
        m = s / n
        sd = np.sqrt(max(ss / n - m * m, 1e-12))
        rows.append(dict(code=code, q=q, tier=tier, n_trades=n, avg_ticks=m,
                         hit=nh / n, tstat=m / (sd / np.sqrt(n))))
    pd.DataFrame(rows).to_csv(f"{D}/bt_results{SUF}.csv", index=False)
    drows = [dict(code=c, day=pd.Timestamp(d_), pnl_cny=p) for (c, d_), p in sorted(daily.items())]
    pd.DataFrame(drows).to_csv(f"{D}/bt_daily{SUF}.csv", index=False)
    print(f"saved bt_results{SUF}.csv bt_daily{SUF}.csv")
