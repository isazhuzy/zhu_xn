"""xc_icim_backtest_ddb.py — can IM's order flow make money trading IC? net-of-cost backtest.
Signal at 1s bin t (frozen 2022-24 coefficients):
  own  : ŷ_IC(t+1) = a0 + a1·OFI_IC(t)
  pair : ŷ_IC(t+1) = b0 + b1·OFI_IC(t) + b2·OFI_IM(t)     ← "use IM to predict IC"
Trade IC when |ŷ|>q, hold 1 bin (=1s, the prediction horizon), non-overlapping. Five
execution/cost tiers per trade (same ladder as voi_backtest / fig85):
  mid0   : mid→mid (frictionless signal value)
  taker0 : cross spread same bin (optimistic latency)
  taker1 : cross spread +1 bin latency (realistic)
  fee_yz : taker1 + 平昨 fees   fee_jt : taker1 + 平今 fees
Compare own vs pair to see if IM's flow adds NET profit (not just R²).
Test 2025-01..2026-05 (pilot: train 2024-06 → test 2024-07).
Outputs: xc_icim_bt_results[_pilot].csv, xc_icim_bt_daily[_pilot].csv
Run: PILOT=1 /Users/zhuisabella/xn/.venv/bin/python xc_icim_backtest_ddb.py   (sandbox OFF)
"""
import calendar
import os
import numpy as np, pandas as pd, dolphindb as ddb
from ddb_config import HOST, PORT, USER, PW
from lob_common import TICK, fetch_l1, prep_l1
from crossimpact_ddb import add_ofi

PILOT = os.environ.get("PILOT") == "1"
SUF = "_pilot" if PILOT else ""
D = "/Users/zhuisabella/xn/prediction"
W = 1
MULT = 200                                    # IC multiplier (yuan/pt)
QS = [0.05, 0.1, 0.2, 0.5]
FEE_OPEN, FEE_YZ, FEE_JT = 0.23e-4, 0.23e-4, 3.45e-4
TREND = (2024, 6) if PILOT else (2024, 12)
DAILY_Q, DAILY_TIER = 0.1, "fee_yz"


def per_bin_px(df):
    b = (df["ts"].astype("int64") // 10**6 // (W * 1000)).astype("int64")
    return df.assign(b=b).groupby(["gid", "b"], sort=True).agg(
        ofi=("ofi", "sum"), mid=("mid_tk", "last"), pb=("pb", "last"), pa=("pa", "last")).reset_index()


def sess_arrays(pbic, pbim, gid):
    a = pbic[pbic.gid == gid].set_index("b"); c = pbim[pbim.gid == gid].set_index("b")
    if len(a) < 3 or len(c) < 1:
        return None
    lo = min(a.index.min(), c.index.min()); hi = max(a.index.max(), c.index.max())
    idx = np.arange(lo, hi + 1)
    a = a.reindex(idx); c = c.reindex(idx)
    mid = a["mid"].ffill().to_numpy(float)
    pb = a["pb"].ffill().to_numpy(float); pa = a["pa"].ffill().to_numpy(float)
    oic = a["ofi"].fillna(0.0).to_numpy(float); oim = c["ofi"].fillna(0.0).to_numpy(float)
    return mid, pb, pa, oic, oim


if __name__ == "__main__":
    months = [(2024, 6), (2024, 7)] if PILOT else [
        (y, m) for y in range(2022, 2027) for m in range(1, 13) if (2022, 7) <= (y, m) <= (2026, 5)]
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    Mown = dict(XtX=np.zeros((2, 2)), Xty=np.zeros(2), n=0)
    Mpair = dict(XtX=np.zeros((3, 3)), Xty=np.zeros(3), n=0)
    beta = {}
    acc = {}                                  # (signal,q,tier) -> [sum, sumsq, n, nhit]
    daily = {}                                # (signal, day) -> pnl CNY
    for yr, mo in months:
        phase = "train" if (yr, mo) <= TREND else "test"
        if phase == "test" and not beta:
            beta["own"] = np.linalg.solve(Mown["XtX"], Mown["Xty"])
            beta["pair"] = np.linalg.solve(Mpair["XtX"], Mpair["Xty"])
        last = calendar.monthrange(yr, mo)[1]
        dfic = fetch_l1(sess, "IC0000", f"{yr}.{mo:02d}.01", f"{yr}.{mo:02d}.{last:02d}")
        dfim = fetch_l1(sess, "IM0000", f"{yr}.{mo:02d}.01", f"{yr}.{mo:02d}.{last:02d}")
        if not len(dfic) or not len(dfim):
            print(f"{yr}-{mo:02d} skip", flush=True); continue
        pbic = per_bin_px(add_ofi(prep_l1(dfic))); pbim = per_bin_px(add_ofi(prep_l1(dfim)))
        gids = pd.unique(pd.concat([pbic["gid"], pbim["gid"]]))
        for gid in gids:
            A = sess_arrays(pbic, pbim, gid)
            if A is None:
                continue
            mid, pb, pa, oic, oim = A
            ok = np.isfinite(mid) & np.isfinite(pb) & np.isfinite(pa)   # leading-NaN guard
            rnext = np.roll(np.r_[np.nan, np.diff(mid)], -1)   # r_IC(t+1) = mid[i+1]-mid[i]
            if phase == "train":
                m = np.isfinite(rnext)
                Xo = np.column_stack([np.ones(m.sum()), oic[m]])
                Xp = np.column_stack([np.ones(m.sum()), oic[m], oim[m]])
                y = rnext[m]
                Mown["XtX"] += Xo.T @ Xo; Mown["Xty"] += Xo.T @ y; Mown["n"] += len(y)
                Mpair["XtX"] += Xp.T @ Xp; Mpair["Xty"] += Xp.T @ y; Mpair["n"] += len(y)
                continue
            day = pd.Timestamp(gid // 2, unit="D") if False else None   # gid encodes day; use index below
            L = len(mid)
            yhat = {"own": beta["own"][0] + beta["own"][1] * oic,
                    "pair": beta["pair"][0] + beta["pair"][1] * oic + beta["pair"][2] * oim}
            for sig in ("own", "pair"):
                yh = yhat[sig]
                for q in QS:
                    nxt = 0; i = 0
                    while i < L - 2:
                        if abs(yh[i]) > q and ok[i] and ok[i + 1] and ok[i + 2]:
                            s = np.sign(yh[i])
                            pnl_mid = s * (mid[i + 1] - mid[i])
                            p_in0 = pa[i] if s > 0 else pb[i]; p_out0 = pb[i + 1] if s > 0 else pa[i + 1]
                            p_in1 = pa[i + 1] if s > 0 else pb[i + 1]; p_out1 = pb[i + 2] if s > 0 else pa[i + 2]
                            pnl_tk0 = s * (p_out0 - p_in0) / TICK
                            pnl_tk1 = s * (p_out1 - p_in1) / TICK
                            fee = lambda rt: (p_in1 + p_out1) * rt / TICK
                            pnl_yz = pnl_tk1 - fee((FEE_OPEN + FEE_YZ) / 2)
                            pnl_jt = pnl_tk1 - fee((FEE_OPEN + FEE_JT) / 2)
                            for tier, v in [("mid0", pnl_mid), ("taker0", pnl_tk0), ("taker1", pnl_tk1),
                                            ("fee_yz", pnl_yz), ("fee_jt", pnl_jt)]:
                                a = acc.setdefault((sig, q, tier), [0.0, 0.0, 0, 0])
                                a[0] += v; a[1] += v * v; a[2] += 1; a[3] += int(pnl_mid > 0)
                            if q == DAILY_Q:
                                dkey = (sig, gid // 2)
                                daily[dkey] = daily.get(dkey, 0.0) + pnl_yz * TICK * MULT
                            i += 3
                        else:
                            i += 1
        print(f"{yr}-{mo:02d} done ({phase})", flush=True)
    sess.close()

    rows = []
    for (sig, q, tier), (s, ss, n, nh) in sorted(acc.items()):
        m = s / n; sd = np.sqrt(max(ss / n - m * m, 1e-12))
        rows.append(dict(signal=sig, q=q, tier=tier, n_trades=n, avg_ticks=m,
                         hit=nh / n, tstat=m / (sd / np.sqrt(n))))
    pd.DataFrame(rows).to_csv(f"{D}/xc_icim_bt_results{SUF}.csv", index=False)
    drows = [dict(signal=sig, gidday=gd, pnl_cny=v) for (sig, gd), v in sorted(daily.items())]
    pd.DataFrame(drows).to_csv(f"{D}/xc_icim_bt_daily{SUF}.csv", index=False)
    print(f"saved xc_icim_bt_results{SUF}.csv xc_icim_bt_daily{SUF}.csv")
