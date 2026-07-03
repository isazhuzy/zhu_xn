"""voi_ddb.py — Paper 3: D. Shen (2015), *Order Imbalance Based Strategy in High
Frequency Trading*, Oxford MSc thesis (built on Chinese futures L1 data, like ours).

Factors per 500ms snapshot (everything price-like in TICK units, tick=0.2):
  VOI_t : volume order imbalance  = dV_bid - dV_ask, where
          dV_bid = 0 if Pb down, qb - qb_prev if Pb same, qb if Pb up  (mirror for ask)
  OIR_t : depth imbalance ratio   = (qb - qa)/(qb + qa)
  MPB_t : mid-price basis         = avg trade price - (mid_t + mid_{t-1})/2,
          avg trade price = dTurnover/(dVolume*MULT), carried forward when no trades.
Target: y_k(t) = mean(mid_{t+1..t+k}) - mid_t   for k in {1,4,20,120} snapshots.
Models (Shen's Model 1 / Model 2):
  A: y_k ~ 1 + VOI_{t..t-5}
  B: y_k ~ 1 + VOI/s_{t..t-5} + OIR/s_{t..t-5} + MPB/s_t     (s = spread in ticks)
Train 2020-01..2024-12, test 2025-01..2026-05 (pilot: 2024-06 / 2024-07). Pooled OLS
via accumulated moments (XtX, Xty, yty); exact out-of-sample R2 on test moments with
train betas; row-level OOS sign hit rates (model B, k=20) at |yhat| thresholds.

Outputs: voi_results[_pilot].csv, voi_coefs[_pilot].csv, voi_permonth[_pilot].csv,
         voi_hitrate[_pilot].csv
Run: PILOT=1 /Users/zhuisabella/xn/.venv/bin/python voi_ddb.py   (sandbox OFF)
"""
import calendar
import os
import numpy as np, pandas as pd, dolphindb as ddb
from ddb_config import HOST, PORT, USER, PW
from lob_common import CODES, MULT, TICK, fetch_l1, prep_l1, month_windows, train_end

PILOT = os.environ.get("PILOT") == "1"
SUF = "_pilot" if PILOT else ""
D = "/Users/zhuisabella/xn/prediction"
K = [1, 4, 20, 120]
NLAG = 6                                  # lags 0..5
THRS = [0.0, 0.1, 0.2, 0.5]               # |yhat| thresholds (ticks) for hit rates
TREND = train_end(PILOT)


def build(df, code):
    """factor columns + targets; returns df with voi0..5, oir0..5, mpbs, spr, y_k."""
    g = df.groupby("gid", sort=False)
    pbi = np.rint(df.pb / TICK).astype(int); pai = np.rint(df.pa / TICK).astype(int)
    pb1 = g["pb"].shift(1); pa1 = g["pa"].shift(1)
    pb1i = np.rint(pb1 / TICK); pa1i = np.rint(pa1 / TICK)
    qb1 = g["qb"].shift(1); qa1 = g["qa"].shift(1)
    dvb = np.where(pbi < pb1i, 0.0, np.where(pbi == pb1i, df.qb - qb1, df.qb))
    dva = np.where(pai > pa1i, 0.0, np.where(pai == pa1i, df.qa - qa1, df.qa))
    df["voi"] = np.where(pb1.notna(), dvb - dva, np.nan)
    df["oir"] = (df.qb - df.qa) / (df.qb + df.qa)
    tp = np.where(df.vol > 0, df.amt / (df.vol * MULT[code[:2]]), np.nan) / TICK
    tp = pd.Series(tp, index=df.index).groupby(df["gid"]).ffill().fillna(df["mid_tk"])
    df["mpb"] = tp - (df["mid_tk"] + g["mid_tk"].shift(1)) / 2
    for l in range(NLAG):
        df[f"voi{l}"] = g["voi"].shift(l)
        df[f"oir{l}"] = g["oir"].shift(l)
    gm = df.groupby("gid", sort=False)["mid_tk"]
    for k in K:
        df[f"y{k}"] = gm.transform(lambda s, k=k: s.rolling(k).mean().shift(-k)) - df["mid_tk"]
        df.loc[df[f"y{k}"].abs() > 100, f"y{k}"] = np.nan     # bad-tick guard
    return df


def design(df, model):
    s = df["spr"].to_numpy(float)
    cols = [np.ones(len(df))]
    if model == "A":
        cols += [df[f"voi{l}"].to_numpy(float) for l in range(NLAG)]
    else:
        cols += [df[f"voi{l}"].to_numpy(float) / s for l in range(NLAG)]
        cols += [df[f"oir{l}"].to_numpy(float) / s for l in range(NLAG)]
        cols += [df["mpb"].to_numpy(float) / s]
    return np.column_stack(cols)


def varnames(model):
    if model == "A":
        return ["const"] + [f"voi{l}" for l in range(NLAG)]
    return (["const"] + [f"voi{l}/s" for l in range(NLAG)]
            + [f"oir{l}/s" for l in range(NLAG)] + ["mpb/s"])


def r2_from(mom, beta=None):
    XtX, Xty, yty, n = mom["XtX"], mom["Xty"], mom["yty"], mom["n"]
    if n < 1000:
        return np.nan
    if beta is None:
        beta = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
    sse = yty - 2 * beta @ Xty + beta @ XtX @ beta
    sst = yty - Xty[0] ** 2 / n            # Xty[0] = sum(y) via const column
    return 1.0 - sse / sst if sst > 0 else np.nan


def zero_mom(p):
    return dict(XtX=np.zeros((p, p)), Xty=np.zeros(p), yty=0.0, n=0)


if __name__ == "__main__":
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    P = {"A": 1 + NLAG, "B": 1 + 2 * NLAG + 1}
    acc = {}                              # (code,k,model,phase) -> moments
    beta_tr = {}                          # (code,k,model) -> train beta (frozen)
    hit = {}                              # (code,thr) -> [nhit, ncov, ntot]
    monrows = []
    for yr, mo in month_windows(PILOT):
        phase = "train" if (yr, mo) <= TREND else "test"
        if phase == "test" and not beta_tr:
            for (c, k, m, ph), mom in list(acc.items()):
                if ph == "train" and mom["n"] >= 1000:
                    beta_tr[(c, k, m)] = np.linalg.lstsq(mom["XtX"], mom["Xty"], rcond=None)[0]
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
            for k in K:
                v = df.dropna(subset=need + [f"y{k}"])
                if len(v) < 1000:
                    continue
                y = v[f"y{k}"].to_numpy(float)
                for model in ("A", "B"):
                    X = design(v, model)
                    key = (code, k, model, phase)
                    if key not in acc:
                        acc[key] = zero_mom(P[model])
                    a = acc[key]
                    a["XtX"] += X.T @ X; a["Xty"] += X.T @ y
                    a["yty"] += y @ y; a["n"] += len(y)
                    if k == 20 and model == "B":
                        mom_m = dict(XtX=X.T @ X, Xty=X.T @ y, yty=y @ y, n=len(y))
                        row = dict(code=code, year=yr, month=mo, phase=phase,
                                   r2_refit=r2_from(mom_m), r2_oos=np.nan, n=len(y))
                        if phase == "test" and (code, 20, "B") in beta_tr:
                            b = beta_tr[(code, 20, "B")]
                            yh = X @ b
                            row["r2_oos"] = r2_from(mom_m, b)
                            for thr in THRS:
                                mcov = np.abs(yh) > thr
                                msgn = mcov & (y != 0)
                                h = hit.setdefault((code, thr), [0, 0, 0])
                                h[0] += int((np.sign(yh[msgn]) == np.sign(y[msgn])).sum())
                                h[1] += int(msgn.sum()); h[2] += len(y)
                        monrows.append(row)
        print(f"{yr}-{mo:02d} done ({phase})", flush=True)
    sess.close()

    resrows, coefrows = [], []
    for code in CODES:
        for k in K:
            for model in ("A", "B"):
                tr = acc.get((code, k, model, "train"), zero_mom(P[model]))
                te = acc.get((code, k, model, "test"), zero_mom(P[model]))
                b = beta_tr.get((code, k, model))
                resrows.append(dict(
                    code=code, k=k, secs=k * 0.5, model=model,
                    n_train=tr["n"], r2_train=r2_from(tr),
                    n_test=te["n"],
                    r2_test_oos=r2_from(te, b) if b is not None else np.nan,
                    r2_test_refit=r2_from(te)))
                if b is not None:
                    for name, bv in zip(varnames(model), b):
                        coefrows.append(dict(code=code, k=k, model=model, var=name, beta=bv))
    hitrows = [dict(code=c, thr=t, hitrate=v[0] / v[1] if v[1] else np.nan,
                    coverage=v[1] / v[2] if v[2] else np.nan, n_signal=v[1], n_total=v[2])
               for (c, t), v in sorted(hit.items())]
    pd.DataFrame(resrows).to_csv(f"{D}/voi_results{SUF}.csv", index=False)
    pd.DataFrame(coefrows).to_csv(f"{D}/voi_coefs{SUF}.csv", index=False)
    pd.DataFrame(monrows).to_csv(f"{D}/voi_permonth{SUF}.csv", index=False)
    pd.DataFrame(hitrows).to_csv(f"{D}/voi_hitrate{SUF}.csv", index=False)
    print(f"saved voi_results{SUF} voi_coefs{SUF} voi_permonth{SUF} voi_hitrate{SUF}")
