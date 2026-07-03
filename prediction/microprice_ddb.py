"""microprice_ddb.py — Paper 4: S. Stoikov (2018), *The micro-price: a high-frequency
estimator of future prices*, Quantitative Finance 18(12).

Idea: the 'fair' price is not the mid — it is the expected mid at the time of future
price moves, conditional on the book state x = (imbalance bin, spread state).
  I = qb/(qb+qa) in 10 bins; spread state = 1 / 2 / 3+ ticks  ->  30 states.
Markov estimator (the paper's construction, adapted to 500ms snapshots):
  no-move transitions  T[x,x'] ; move events: Rsum[x]=sum dM, Rcnt[x], B[x,x']
  G1 = (I - T)^(-1) rhat  (expected mid change at the FIRST move | state)
  g* = G1 + B G1 + ... + B^6 G1, then antisymmetrized in the imbalance bin.
  micro-price = mid + g*(x).
Train 2020-01..2024-12, test 2025-01..2026-05 (pilot: 2024-06 / 2024-07).
Out-of-sample horse race vs mid and weighted mid  Pw = I*Pa + (1-I)*Pb :
  RMSE of (predictor - mid_{t+h}) for h in {1,4,20,120} snapshots, and the paper's
  signature plot: E[mid_{t+20} - predictor | imbalance decile].

Outputs: mp_gstar[_pilot].csv (code, spr, ibin, icenter, n_state, G1, gstar ticks)
         mp_rmse[_pilot].csv  (code, h, predictor, rmse_ticks, n)
         mp_bias[_pilot].csv  (code, predictor, ibin, drift_ticks, n)   [h=20]
Run: PILOT=1 /Users/zhuisabella/xn/.venv/bin/python microprice_ddb.py  (sandbox OFF)
"""
import calendar
import os
import numpy as np, pandas as pd, dolphindb as ddb
from ddb_config import HOST, PORT, USER, PW
from lob_common import CODES, TICK, fetch_l1, prep_l1, month_windows, train_end

PILOT = os.environ.get("PILOT") == "1"
SUF = "_pilot" if PILOT else ""
D = "/Users/zhuisabella/xn/prediction"
NI, NS = 10, 3                            # imbalance bins x spread states
NST = NI * NS
H = [1, 4, 20, 120]
HBIAS = 20
TREND = train_end(PILOT)


def annotate(df):
    df = df[(df.qb > 0) & (df.qa > 0)].copy()
    df["I"] = df.qb / (df.qb + df.qa)
    df["ibin"] = np.clip((df["I"] * NI).astype(int), 0, NI - 1)
    df["state"] = (np.clip(df["spr"], 1, NS) - 1) * NI + df["ibin"]
    return df


def gstar_from(Tc, Bc, Rsum, Rcnt):
    N = Tc.sum(1) + Rcnt
    ok = N > 0
    That = np.zeros_like(Tc); rhat = np.zeros(NST)
    That[ok] = Tc[ok] / N[ok, None]; rhat[ok] = Rsum[ok] / N[ok]
    G1 = np.linalg.solve(np.eye(NST) - That, rhat)
    Bhat = np.zeros_like(Bc)
    okb = Rcnt > 0
    Bhat[okb] = Bc[okb] / Rcnt[okb, None]
    g = G1.copy(); term = G1.copy()
    for _ in range(6):
        term = Bhat @ term
        g += term
    gs = g.reshape(NS, NI)
    gsym = ((gs - gs[:, ::-1]) / 2).ravel()   # enforce buy/sell symmetry
    return G1, gsym, N


if __name__ == "__main__":
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    Tc = {c: np.zeros((NST, NST)) for c in CODES}
    Bc = {c: np.zeros((NST, NST)) for c in CODES}
    Rsum = {c: np.zeros(NST) for c in CODES}
    Rcnt = {c: np.zeros(NST) for c in CODES}
    gtab = {}                              # code -> (G1, gstar, N) frozen after train
    rmse = {}                              # (code,h,pred) -> [sse, n]
    bias = {}                              # (code,pred,ibin) -> [sum drift, n]
    for yr, mo in month_windows(PILOT):
        phase = "train" if (yr, mo) <= TREND else "test"
        if phase == "test" and not gtab:
            for c in CODES:
                gtab[c] = gstar_from(Tc[c], Bc[c], Rsum[c], Rcnt[c])
        last = calendar.monthrange(yr, mo)[1]
        for code in CODES:
            df = fetch_l1(sess, code, f"{yr}.{mo:02d}.01", f"{yr}.{mo:02d}.{last:02d}")
            if not len(df):
                continue
            df = prep_l1(df)
            if df.empty:
                continue
            df = annotate(df)
            g = df.groupby("gid", sort=False)
            if phase == "train":
                nxt = g["state"].shift(-1)
                dM = g["mid_tk"].shift(-1) - df["mid_tk"]
                m = nxt.notna() & dM.notna() & (dM.abs() <= 50)
                x0 = df.loc[m, "state"].to_numpy(int)
                x1 = nxt[m].to_numpy(int)
                d = dM[m].to_numpy(float)
                mv = np.abs(d) >= 0.25     # mid moves in 0.5-tick multiples
                np.add.at(Tc[code], (x0[~mv], x1[~mv]), 1.0)
                np.add.at(Bc[code], (x0[mv], x1[mv]), 1.0)
                np.add.at(Rsum[code], x0[mv], d[mv])
                np.add.at(Rcnt[code], x0[mv], 1.0)
            else:
                _, gstar, _ = gtab[code]
                mid = df["mid_tk"].to_numpy(float)
                wmid = (df["I"] * df.pa + (1 - df["I"]) * df.pb).to_numpy(float) / TICK
                micro = mid + gstar[df["state"].to_numpy(int)]
                preds = {"mid": mid, "wmid": wmid, "micro": micro}
                for h in H:
                    fm = g["mid_tk"].shift(-h).to_numpy(float)
                    m = np.isfinite(fm) & (np.abs(fm - mid) <= 100)
                    for pn, pv in preds.items():
                        e = pv[m] - fm[m]
                        r = rmse.setdefault((code, h, pn), [0.0, 0])
                        r[0] += float(e @ e); r[1] += len(e)
                        if h == HBIAS:
                            drift = fm[m] - pv[m]
                            ib = df.loc[m, "ibin"].to_numpy(int)
                            for b in range(NI):
                                sel = ib == b
                                if sel.any():
                                    s = bias.setdefault((code, pn, b), [0.0, 0])
                                    s[0] += float(drift[sel].sum()); s[1] += int(sel.sum())
        print(f"{yr}-{mo:02d} done ({phase})", flush=True)
    sess.close()
    if not gtab:                            # no test month reached
        for c in CODES:
            gtab[c] = gstar_from(Tc[c], Bc[c], Rsum[c], Rcnt[c])

    grows = []
    for code in CODES:
        G1, gstar, N = gtab[code]
        for s in range(NS):
            for i in range(NI):
                st = s * NI + i
                grows.append(dict(code=code, spr=s + 1, ibin=i, icenter=(i + .5) / NI,
                                  n_state=int(N[st]), G1=G1[st], gstar=gstar[st]))
    rrows = [dict(code=c, h=h, secs=h * 0.5, predictor=p,
                  rmse_ticks=np.sqrt(v[0] / v[1]) if v[1] else np.nan, n=v[1])
             for (c, h, p), v in sorted(rmse.items())]
    brows = [dict(code=c, predictor=p, ibin=b, icenter=(b + .5) / NI,
                  drift_ticks=v[0] / v[1] if v[1] else np.nan, n=v[1])
             for (c, p, b), v in sorted(bias.items())]
    pd.DataFrame(grows).to_csv(f"{D}/mp_gstar{SUF}.csv", index=False)
    pd.DataFrame(rrows).to_csv(f"{D}/mp_rmse{SUF}.csv", index=False)
    pd.DataFrame(brows).to_csv(f"{D}/mp_bias{SUF}.csv", index=False)
    print(f"saved mp_gstar{SUF} mp_rmse{SUF} mp_bias{SUF}")
