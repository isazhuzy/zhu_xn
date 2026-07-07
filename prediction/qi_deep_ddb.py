"""qi_deep_ddb.py — deeper QI (queue-imbalance) study, two extensions in ONE data pass:

  #2 HORIZON DECAY.  qi_ddb.py only used the *first* future mid change. Here we ask how
     far the sign(I) edge reaches: for k in {1,2,5,10,20,60,120} snapshots (0.5s..60s) we
     bin I and record P(mid_{t+k} > mid_t), then fit the same grouped logistic and compute
     the sign hit rate — overall and by spread state. -> qi_horizon[_pilot].csv

  #5 ADVERSE SELECTION conditional on QI (the maker question).  At snapshot t you quote
     passively on both sides using I(t). In the NEXT interval a trade prints; its VWAP vs
     mid(t) says which quote filled: seller-initiated (VWAP<mid) fills your BID (you bought
     ~pb), buyer-initiated (VWAP>mid) fills your ASK (you sold ~pa). Markout at horizon h is
     dmid = mid_{t+h} - mid_t; net maker edge = spr/2 + dmid (bid fill) or spr/2 - dmid (ask
     fill). We bucket dmid, half-spread and counts by I and side. -> qi_markout[_pilot].csv

Trade VWAP + side reuse voi_ddb conventions (amt/(vol*MULT)/TICK, per-interval vol/amt).
Run: PILOT=1 /Users/zhuisabella/xn/.venv/bin/python qi_deep_ddb.py   (sandbox OFF)
"""
import calendar
import os
import numpy as np, pandas as pd, dolphindb as ddb
from ddb_config import HOST, PORT, USER, PW
from lob_common import CODES, MULT, TICK, fetch_l1, prep_l1, month_windows
from qi_ddb import fit_logit

PILOT = os.environ.get("PILOT") == "1"
SUF = "_pilot" if PILOT else ""
D = "/Users/zhuisabella/xn/prediction"
NB = 40                                    # I bins for direction, over [-1,1]; slot NB = I==0
NBM = 40                                   # I bins for markout
NSPR = 3                                   # spread states 1 / 2 / 3+
KDIR = [1, 2, 5, 10, 20, 60, 120]          # direction horizons (snapshots); *0.5 = seconds
HMK = [1, 4, 20]                           # markout horizons (0.5s, 2s, 10s)
GUARD = 200                                # |mid move| tick guard (loose: horizons up to 60s)

centers = np.r_[-1 + (np.arange(NB) + 0.5) * (2 / NB), 0.0]        # bin centers + I==0 slot
cmk = -1 + (np.arange(NBM) + 0.5) * (2 / NBM)


if __name__ == "__main__":
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    # direction: per (code,k) -> (spread, bin) counts and up-counts
    cnt = {(c, k): np.zeros((NSPR, NB + 1)) for c in CODES for k in KDIR}
    up = {(c, k): np.zeros((NSPR, NB + 1)) for c in CODES for k in KDIR}
    # markout: per (code,h) -> per (side 0=bid/1=ask, Ibin) sum dmid, sum halfspread, count
    mkN = {(c, h): np.zeros((2, NBM)) for c in CODES for h in HMK}
    mkD = {(c, h): np.zeros((2, NBM)) for c in CODES for h in HMK}
    mkH = {(c, h): np.zeros((2, NBM)) for c in CODES for h in HMK}

    for yr, mo in month_windows(PILOT):
        last = calendar.monthrange(yr, mo)[1]
        for code in CODES:
            df = fetch_l1(sess, code, f"{yr}.{mo:02d}.01", f"{yr}.{mo:02d}.{last:02d}")
            if not len(df):
                continue
            df = prep_l1(df)
            if df.empty:
                continue
            g = df.groupby("gid", sort=False)
            mid = df["mid_tk"]
            I = ((df.qb - df.qa) / (df.qb + df.qa)).to_numpy()
            Ibin = np.clip(((I + 1) / 2 * NB).astype(int), 0, NB - 1)
            Ibin = np.where(I == 0, NB, Ibin)
            Imk = np.clip(((I + 1) / 2 * NBM).astype(int), 0, NBM - 1)
            sidx = np.clip(df["spr"].to_numpy(), 1, NSPR) - 1
            hs = df["spr"].to_numpy(float) / 2.0                    # half-spread in ticks

            # ---- #2 direction at each horizon ----
            for k in KDIR:
                dk = (g["mid_tk"].shift(-k) - mid).to_numpy()
                m = np.isfinite(dk) & (np.abs(dk) <= GUARD) & (dk != 0)
                np.add.at(cnt[(code, k)], (sidx[m], Ibin[m]), 1.0)
                np.add.at(up[(code, k)], (sidx[m], Ibin[m]), (dk[m] > 0).astype(float))

            # ---- #5 markout / adverse selection ----
            with np.errstate(divide="ignore", invalid="ignore"):
                tp = np.where(df.vol.to_numpy() > 0,
                              df.amt.to_numpy() / (df.vol.to_numpy() * MULT[code[:2]]), np.nan) / TICK
            tp = pd.Series(tp, index=df.index)
            # next-interval trade VWAP and volume (fill happens in (t, t+1])
            voln = g["vol"].shift(-1).to_numpy()
            tpn = tp.groupby(df["gid"]).shift(-1).to_numpy()
            midv = mid.to_numpy()
            traded = np.isfinite(tpn) & (voln > 0)
            side = np.where(tpn > midv, 1, np.where(tpn < midv, 0, -1))   # 1=ask fill,0=bid fill,-1=none
            for h in HMK:
                dh = (g["mid_tk"].shift(-h) - mid).to_numpy()
                ok = traded & (side >= 0) & np.isfinite(dh) & (np.abs(dh) <= GUARD)
                s, b, d, hh = side[ok], Imk[ok], dh[ok], hs[ok]
                np.add.at(mkN[(code, h)], (s, b), 1.0)
                np.add.at(mkD[(code, h)], (s, b), d)
                np.add.at(mkH[(code, h)], (s, b), hh)
        print(f"{yr}-{mo:02d} done", flush=True)
    sess.close()

    # ---- write #2 ----
    rows = []
    for code in CODES:
        for k in KDIR:
            variants = [("all", cnt[(code, k)].sum(0), up[(code, k)].sum(0))]
            variants += [(str(s + 1), cnt[(code, k)][s], up[(code, k)][s]) for s in range(NSPR)]
            for tag, n, kk in variants:
                a, b, r2 = fit_logit(centers, n, kk)
                pos, neg = centers > 0, centers < 0
                nhit = kk[pos].sum() + (n[neg] - kk[neg]).sum()
                ntot = n[pos].sum() + n[neg].sum()
                rows.append(dict(code=code, sprstate=tag, k=k, secs=k * 0.5, a=a, b=b,
                                 pseudoR2=r2, hitrate=nhit / ntot if ntot else np.nan,
                                 n=int(n.sum())))
    pd.DataFrame(rows).to_csv(f"{D}/qi_horizon{SUF}.csv", index=False)

    # ---- write #5 ----
    rows = []
    for code in CODES:
        for h in HMK:
            for si, sname in [(0, "bid_fill"), (1, "ask_fill")]:
                n = mkN[(code, h)][si]; sd = mkD[(code, h)][si]; sh = mkH[(code, h)][si]
                for i in range(NBM):
                    if n[i] > 0:
                        md = sd[i] / n[i]                          # mean markout (ticks)
                        mh = sh[i] / n[i]                          # mean half-spread (ticks)
                        edge = mh + md if si == 0 else mh - md     # net maker edge (ticks)
                        rows.append(dict(code=code, h=h, secs=h * 0.5, side=sname,
                                         Ibin=cmk[i], n=int(n[i]), mean_markout=md,
                                         mean_halfspread=mh, mean_netedge=edge))
    pd.DataFrame(rows).to_csv(f"{D}/qi_markout{SUF}.csv", index=False)
    print(f"saved qi_horizon{SUF}.csv qi_markout{SUF}.csv")
