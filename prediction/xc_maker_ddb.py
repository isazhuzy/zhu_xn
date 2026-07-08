"""xc_maker_ddb.py — the capstone: market-making IC with P_micro (paper #4) as quote
center + cross-drift (paper #5, IM's flow) as skew. Does combining them beat naive mid quoting?

Maker model (1s grid, honest fills from aggressor volume):
  - rest at best bid pb and best ask pa each bin.
  - a BID fills when aggressive SELLs hit it (Σ m_nActAskVolume > 0);
    an ASK fills when aggressive BUYs lift it (Σ m_nActBidVolume > 0).
  - P&L per fill = markout: bid → mid(t+H) − pb ; ask → pa − mid(t+H)   (ticks).
    half-spread capture minus adverse selection, baked in.
  - a fair-value estimate F decides WHICH side to quote (skew):
      F > mid+τ (up)  → keep bid, PULL ask (don't sell before it rises)
      F < mid−τ (dn)  → keep ask, PULL bid
      else            → quote both
Three centers compared:
  mid   : F = mid                              (naive, never skews — baseline)
  micro : F = mid + g*_IC(state)               (paper #4 level correction)
  cross : F = mid + g*_IC(state) + drift_flow  (+ paper #5: c_IC·OFI_IC + c_IM·OFI_IM)
g*_state (E[Δmid | imbalance,spread]) and the flow coefficients are FROZEN on train
(2022-07..2024-12), tested 2025-01..2026-05. H = 4 bins (~4s hold/markout).
Outputs: xc_maker_results[_pilot].csv (center, tau, side, n_fills, avg_markout_tk, total_tk)
Run: PILOT=1 /Users/zhuisabella/xn/.venv/bin/python xc_maker_ddb.py   (sandbox OFF)
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
W, H = 1, 4                                     # 1s bins; markout/hold = 4 bins (~4s)
NI, NS = 10, 3
TAUS = [0.1, 0.25, 0.5]                         # skew thresholds (ticks)
TREND = (2024, 6) if PILOT else (2024, 12)


def fetch_ic(sess, start, end):
    q = f"""
    pt=loadTable("dfs://hft_future_ts","TickPartitioned")
    select m_nDatetime as ts, m_nBidPrice as pb, m_nBidVolume as qb,
           m_nAskPrice as pa, m_nAskVolume as qa,
           m_nActBidVolume as actb, m_nActAskVolume as acta
    from pt where code_init=`IC, code=`IC0000,
          m_nDatetime>={start}T00:00:00, m_nDatetime<={end}T23:59:59,
          minute(m_nDatetime) between 09:30m:15:00m
    """
    return sess.run(q)


def per_bin_ic(df):
    b = (df["ts"].astype("int64") // 10**6 // (W * 1000)).astype("int64")
    return df.assign(b=b).groupby(["gid", "b"], sort=True).agg(
        mid=("mid_tk", "last"), pb=("pb", "last"), pa=("pa", "last"),
        qb=("qb", "last"), qa=("qa", "last"), spr=("spr", "last"),
        ofi=("ofi", "sum"), actb=("actb", "sum"), acta=("acta", "sum")).reset_index()


def per_bin_ofi(df):
    b = (df["ts"].astype("int64") // 10**6 // (W * 1000)).astype("int64")
    return df.assign(b=b).groupby(["gid", "b"], sort=True).agg(ofi=("ofi", "sum")).reset_index()


def state_of(qb, qa, spr):
    I = qb / (qb + qa)
    ib = np.clip((I * NI).astype(int), 0, NI - 1)
    sb = np.clip(spr, 1, NS) - 1
    return sb * NI + ib


if __name__ == "__main__":
    months = [(2024, 6), (2024, 7)] if PILOT else [
        (y, m) for y in range(2022, 2027) for m in range(1, 13) if (2022, 7) <= (y, m) <= (2026, 5)]
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    gsum = np.zeros(NI * NS); gcnt = np.zeros(NI * NS)         # g*_state accumulation
    Mf = dict(XtX=np.zeros((3, 3)), Xty=np.zeros(3), n=0)      # flow drift: dmid ~ 1+OFI_IC+OFI_IM
    frozen = {}
    acc = {}                                                  # (center,tau,side) -> [pnl_sum, n]
    for yr, mo in months:
        phase = "train" if (yr, mo) <= TREND else "test"
        if phase == "test" and not frozen:
            frozen["g"] = np.where(gcnt > 0, gsum / np.maximum(gcnt, 1), 0.0)
            frozen["c"] = np.linalg.solve(Mf["XtX"], Mf["Xty"])
        last = calendar.monthrange(yr, mo)[1]
        dic = fetch_ic(sess, f"{yr}.{mo:02d}.01", f"{yr}.{mo:02d}.{last:02d}")
        dim = fetch_l1(sess, "IM0000", f"{yr}.{mo:02d}.01", f"{yr}.{mo:02d}.{last:02d}")
        if not len(dic) or not len(dim):
            print(f"{yr}-{mo:02d} skip", flush=True); continue
        pic = per_bin_ic(add_ofi(prep_l1(dic))); pim = per_bin_ofi(add_ofi(prep_l1(dim)))
        gids = pd.unique(pic["gid"])
        for gid in gids:
            a = pic[pic.gid == gid].set_index("b"); c = pim[pim.gid == gid].set_index("b")
            if len(a) < H + 1:
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
            st = np.zeros(len(mid), int)
            st[ok] = state_of(qb[ok], qa[ok], spr[ok].astype(int))
            fut = np.full(len(mid), np.nan)
            fut[:-H] = mid[H:] - mid[:-H]                      # mid(t+H) - mid(t)
            if phase == "train":
                m = ok & np.isfinite(fut)
                np.add.at(gsum, st[m], fut[m]); np.add.at(gcnt, st[m], 1.0)
                X = np.column_stack([np.ones(m.sum()), oic[m], oim[m]]); y = fut[m]
                Mf["XtX"] += X.T @ X; Mf["Xty"] += X.T @ y; Mf["n"] += len(y)
                continue
            g = frozen["g"]; cc = frozen["c"]
            micro_adj = g[st]
            flow = cc[1] * oic + cc[2] * oim
            centers = {"mid": np.zeros(len(mid)), "micro": micro_adj, "cross": micro_adj + flow}
            valid = ok & np.isfinite(fut)
            for cen, Fadj in centers.items():
                diff = Fadj                                    # F - mid
                midH = mid + fut                               # = mid(t+H)
                for tau in TAUS:
                    qbid = diff >= -tau                        # pull bid when down expected
                    qask = diff <= tau                         # pull ask when up expected
                    ib = np.where(valid & qbid & (acta > 0))[0]   # bid filled by aggressive sells
                    ia = np.where(valid & qask & (actb > 0))[0]   # ask filled by aggressive buys
                    for idxs, pnl, side in [(ib, midH[ib] - pb[ib], "bid"),
                                            (ia, pa[ia] - midH[ia], "ask")]:
                        if len(idxs) == 0:
                            continue
                        s = acc.setdefault((cen, tau, side), [0.0, 0])
                        s[0] += float(np.sum(pnl)); s[1] += len(idxs)
        print(f"{yr}-{mo:02d} done ({phase})", flush=True)
    sess.close()

    rows = []
    tot = {}
    for (cen, tau, side), (psum, n) in acc.items():
        rows.append(dict(center=cen, tau=tau, side=side, n_fills=n,
                         avg_markout_tk=psum / n if n else np.nan, total_tk=psum))
        t = tot.setdefault((cen, tau), [0.0, 0]); t[0] += psum; t[1] += n
    for (cen, tau), (psum, n) in tot.items():
        rows.append(dict(center=cen, tau=tau, side="all", n_fills=n,
                         avg_markout_tk=psum / n if n else np.nan, total_tk=psum))
    pd.DataFrame(rows).to_csv(f"{D}/xc_maker_results{SUF}.csv", index=False)
    print(f"saved xc_maker_results{SUF}.csv")
