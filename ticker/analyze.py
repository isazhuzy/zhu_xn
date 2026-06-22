"""
Sweep momentum over ALL contracts in the file x ALL tick thresholds, and
summarise the three questions:
  Q1  profitable in most minutes?       -> summary.csv : pct_profitable
  Q2/Q3  profitable / losing segments?  -> buckets_all.csv : per 30-min block
Builds the per-contract minute frame ONCE, then applies each threshold cheaply.
"""
import os
import numpy as np
import pandas as pd
from matrix import _minute_frame, apply_threshold, split_by_month


def overall_stats(R):
    a = R.values[(R != 0).values]
    a = a[~np.isnan(a)]
    return {
        "active_mins":     int(a.size),
        "pct_profitable":  float((a > 0).mean()) if a.size else np.nan,
        "mean_ret_active": float(a.mean()) if a.size else np.nan,
        "mean_daily_pnl":  float(R.sum(axis=1).mean()),
    }


def bucket_profile(R, freq_min=30):
    long = R.stack().rename("r").reset_index()
    long.columns = ["day", "tod", "r"]
    long = long.dropna(subset=["r"])
    mins = long["tod"].apply(lambda t: t.hour * 60 + t.minute)
    long["bucket"] = ((mins // freq_min) * freq_min).apply(
        lambda m: f"{m // 60:02d}:{m % 60:02d}")
    act = long[long["r"] != 0]
    g = act.groupby("bucket")["r"]
    return pd.DataFrame({
        "active":   g.size(),
        "hit_rate": g.apply(lambda s: (s > 0).mean()),
        "mean_ret": g.mean(),
        "total":    g.sum(),
    }).sort_index()


if __name__ == "__main__":
    df = pd.read_csv(
        "/Users/zhuisabella/xn/ICIFIHIM/IC_IF_IH_IM_20230104_20230304.csv",
        dtype={"code": "string"}, parse_dates=["m_nDatetime"],
    )

    # ---- contract universe: EVERY code in the file ----
    contracts = sorted(df["code"].unique())
    # optional subsets (uncomment to narrow):
    # contracts = [c for c in contracts if not c.endswith("Ind")]      # drop spot index
    # contracts = [c for c in contracts if c[2:].isdigit()]           # dated contracts only

    # ---- all tick thresholds ----
    thresholds = [1, 2, 3, 5, 8, 10, 15, 20]     # in TICKS; edit freely

    out_dir = "/Users/zhuisabella/xn/ICIFIHIM/momentum_stats"
    os.makedirs(out_dir, exist_ok=True)

    summary_rows, bucket_rows, skipped = [], [], []
    for c in contracts:
        try:
            frame = _minute_frame(df, c, use_mid=True, mode="momentum", lookback=1)
        except Exception as e:                       # empty / malformed contract
            skipped.append((c, "frame", str(e)[:60]))
            continue

        for thr in thresholds:
            R = apply_threshold(frame, thr, "tick")
            st = overall_stats(R)
            if st["active_mins"] == 0:               # threshold killed everything
                skipped.append((c, thr, "no active minutes"))
                continue

            summary_rows.append({"product": c, "thr_ticks": thr,
                                 "month": "ALL", **st})
            for m, Rm in split_by_month(R).items():
                summary_rows.append({"product": c, "thr_ticks": thr,
                                     "month": m, **overall_stats(Rm)})

            # Q2/Q3 + stability: bucket profile for whole window AND each month
            month_mats = {"ALL": R}
            month_mats.update(split_by_month(R))
            for mlabel, Rm in month_mats.items():
                if Rm.shape[0] == 0:
                    continue
                prof = bucket_profile(Rm, 30).reset_index()
                if prof.empty:
                    continue
                prof.insert(0, "month", mlabel)
                prof.insert(0, "thr_ticks", thr)
                prof.insert(0, "product", c)
                bucket_rows.append(prof)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    pd.concat(bucket_rows, ignore_index=True).to_csv(
        os.path.join(out_dir, "buckets_all.csv"), index=False)

    print(f"swept {len(contracts)} contracts x {len(thresholds)} thresholds")
    print(f"summary rows: {len(summary)}  |  skipped (contract,thr): {len(skipped)}")
    if skipped:
        print("examples skipped:", skipped[:6])

    # quick peek: best/worst contracts at one threshold by profitable-minute share
    look = summary[(summary.month == "ALL") & (summary.thr_ticks == 5)]
    look = look.sort_values("pct_profitable", ascending=False)
    print("\nmost profitable-minute share @ 5 ticks (top 5):")
    print(look[["product", "pct_profitable", "active_mins"]].head(5).to_string(index=False))
