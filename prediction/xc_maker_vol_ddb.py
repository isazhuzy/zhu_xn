"""xc_maker_vol_ddb.py — round 3: does predicting VOLATILITY improve the maker?
Extends fig106's best quote center (P_micro + cross-drift). Hypothesis: adverse selection
spikes in high-vol bursts, so PULLING quotes when high vol is predicted lifts per-fill edge.

Predicted-vol proxy = trailing 1-min realized variance (RV autocorrelated ⇒ predicts next vol).
On train we freeze g*_state, flow coefs, AND the 33/67 percentile cutoffs of log(trailing RV).
On test (2025-26), for the cross center we split every fill into predicted-vol terciles
(low/mid/high) and report markout/fill per tercile, plus the "pull top tercile" aggregate.
Outputs: xc_maker_vol_results[_pilot].csv (regime, n_fills, markout_tk)
Run: PILOT=1 /Users/zhuisabella/xn/.venv/bin/python xc_maker_vol_ddb.py   (sandbox OFF)
"""
import calendar
import os
import numpy as np, pandas as pd, dolphindb as ddb
from ddb_config import HOST, PORT, USER, PW
from lob_common import TICK, fetch_l1, prep_l1
from crossimpact_ddb import add_ofi
from xc_maker_ddb import fetch_ic, per_bin_ic, per_bin_ofi, state_of, NI, NS

PILOT = os.environ.get("PILOT") == "1"
SUF = "_pilot" if PILOT else ""
D = "/Users/zhuisabella/xn/prediction"
H, RVWIN, TAU = 4, 60, 0.25                       # markout bins / trailing-RV bins(1min) / skew
TREND = (2024, 6) if PILOT else (2024, 12)


if __name__ == "__main__":
    months = [(2024, 6), (2024, 7)] if PILOT else [
        (y, m) for y in range(2022, 2027) for m in range(1, 13) if (2022, 7) <= (y, m) <= (2026, 5)]
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    gsum = np.zeros(NI * NS); gcnt = np.zeros(NI * NS)
    Mf = dict(XtX=np.zeros((3, 3)), Xty=np.zeros(3), n=0)
    volsamp = []                                  # subsample of log trailing-RV on train
    frozen = {}
    acc = {}                                      # regime -> [markout_sum, n]
    for yr, mo in months:
        phase = "train" if (yr, mo) <= TREND else "test"
        if phase == "test" and not frozen:
            frozen["g"] = np.where(gcnt > 0, gsum / np.maximum(gcnt, 1), 0.0)
            frozen["c"] = np.linalg.solve(Mf["XtX"], Mf["Xty"])
            v = np.array(volsamp)
            frozen["q33"], frozen["q67"] = np.percentile(v, [33, 67])
        last = calendar.monthrange(yr, mo)[1]
        dic = fetch_ic(sess, f"{yr}.{mo:02d}.01", f"{yr}.{mo:02d}.{last:02d}")
        dim = fetch_l1(sess, "IM0000", f"{yr}.{mo:02d}.01", f"{yr}.{mo:02d}.{last:02d}")
        if not len(dic) or not len(dim):
            print(f"{yr}-{mo:02d} skip", flush=True); continue
        pic = per_bin_ic(add_ofi(prep_l1(dic))); pim = per_bin_ofi(add_ofi(prep_l1(dim)))
        for gid in pd.unique(pic["gid"]):
            a = pic[pic.gid == gid].set_index("b"); c = pim[pim.gid == gid].set_index("b")
            if len(a) < H + RVWIN:
                continue
            lo = min(a.index.min(), c.index.min() if len(c) else a.index.min())
            hi = max(a.index.max(), c.index.max() if len(c) else a.index.max())
            idx = np.arange(lo, hi + 1); a = a.reindex(idx); c = c.reindex(idx)
            mid = a["mid"].ffill().to_numpy(float)
            pb = (a["pb"].ffill() / TICK).to_numpy(float); pa = (a["pa"].ffill() / TICK).to_numpy(float)
            qb = a["qb"].ffill().to_numpy(float); qa = a["qa"].ffill().to_numpy(float)
            spr = a["spr"].ffill().to_numpy(float)
            oic = a["ofi"].fillna(0.0).to_numpy(float); oim = c["ofi"].fillna(0.0).to_numpy(float)
            actb = a["actb"].fillna(0.0).to_numpy(float); acta = a["acta"].fillna(0.0).to_numpy(float)
            ok = np.isfinite(mid) & np.isfinite(pb) & np.isfinite(pa) & (qb + qa > 0)
            st = np.zeros(len(mid), int); st[ok] = state_of(qb[ok], qa[ok], spr[ok].astype(int))
            rbin = np.r_[np.nan, np.diff(mid)]
            trv = pd.Series(rbin ** 2).rolling(RVWIN).sum().to_numpy()   # trailing 1-min RV (predicted vol)
            fut = np.full(len(mid), np.nan); fut[:-H] = mid[H:] - mid[:-H]
            midH = mid + fut
            if phase == "train":
                m = ok & np.isfinite(fut)
                np.add.at(gsum, st[m], fut[m]); np.add.at(gcnt, st[m], 1.0)
                X = np.column_stack([np.ones(m.sum()), oic[m], oim[m]]); y = fut[m]
                Mf["XtX"] += X.T @ X; Mf["Xty"] += X.T @ y; Mf["n"] += len(y)
                lv = np.log(trv[np.isfinite(trv) & (trv > 0)] + 1e-9)
                volsamp.extend(lv[::20].tolist())
                continue
            g, cc = frozen["g"], frozen["c"]
            F = g[st] + cc[1] * oic + cc[2] * oim                       # cross center: F - mid
            qbid = F >= -TAU; qask = F <= TAU
            lv = np.log(trv + 1e-9)
            reg = np.where(lv <= frozen["q33"], "low", np.where(lv >= frozen["q67"], "high", "mid"))
            valid = ok & np.isfinite(fut) & np.isfinite(trv)
            for idxs, pnl, quoted in [(None, midH - pb, qbid & (acta > 0)),   # bid fills
                                      (None, pa - midH, qask & (actb > 0))]:  # ask fills
                fill = valid & quoted
                for r in ("low", "mid", "high"):
                    sel = fill & (reg == r)
                    if sel.any():
                        s = acc.setdefault(r, [0.0, 0.0, 0])           # sum, sumsq, n
                        s[0] += float(pnl[sel].sum()); s[1] += float((pnl[sel] ** 2).sum()); s[2] += int(sel.sum())
        print(f"{yr}-{mo:02d} done ({phase})", flush=True)
    sess.close()

    def stat(s):
        mean = s[0] / s[2]; var = max(s[1] / s[2] - mean * mean, 1e-12)
        return mean, np.sqrt(var), mean / np.sqrt(var)                 # mean, std, risk-adj (per-fill Sharpe)

    rows = []
    for r in ("low", "mid", "high"):
        if r in acc:
            m, sd, ra = stat(acc[r])
            rows.append(dict(regime=r, n_fills=acc[r][2], markout_tk=m, std_tk=sd, risk_adj=ra))
    all_s = list(np.sum([acc[r] for r in acc], axis=0))
    keep = list(np.sum([acc[r] for r in ("low", "mid") if r in acc], axis=0))
    for tag, s in [("ALL(不过滤)", all_s), ("PULL_HIGH(撤高波动)", keep)]:
        m, sd, ra = stat(s)
        rows.append(dict(regime=tag, n_fills=int(s[2]), markout_tk=m, std_tk=sd, risk_adj=ra))
    pd.DataFrame(rows).to_csv(f"{D}/xc_maker_vol_results{SUF}.csv", index=False)
    print(f"saved xc_maker_vol_results{SUF}.csv")
    print(pd.DataFrame(rows).to_string(index=False))
