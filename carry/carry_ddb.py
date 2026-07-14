"""carry_ddb.py — Direction A: index-futures basis carry (吃贴水), NO spot feed needed.
There is no usable spot index in the warehouse ({P}Ind does NOT converge at expiry —
it's some weighted futures composite), so we use two spot-free constructions:

1. Term-structure slope carry (Koijen-Moskowitz-Pedersen-Vrugt style):
     ann slope(i,j) = (F_i - F_j)/F_j * 365/(days_j - days_i) * 100   (i nearer)
   Positive = backwardated = long far month rolls up = positive carry.
2. Realized carry anchored at delivery settlements. CFFEX delivery settlement =
   2h average of the SPOT index on expiry day, so DS is an exact spot anchor.
   Roll cycle k: buy next-front contract k at close on expiry day of k-1, hold to
   its own expiry, take DS_k.  fut_ret = DS_k/F_k(E_{k-1}) - 1,
   spot_ret = DS_k/DS_{k-1} - 1, realized carry = fut_ret - spot_ret.
   Dividends handled automatically (price index anchors on both legs).

Outputs: carry_slope.csv    (date,product,pair,ann_pct)
         carry_realized.csv (product,code,entry_date,exit_date,window_d,entry_basis_pct,
                             fut_ret_pct,spot_ret_pct,carry_pct,ann_pct,exp_month)
Run: /Users/zhuisabella/xn/.venv/bin/python carry_ddb.py   (sandbox OFF, ~seconds)
"""
import re
import numpy as np, pandas as pd, dolphindb as ddb
from ddb_config import HOST, PORT, USER, PW

D = "/Users/zhuisabella/xn/carry"
MONTHLY = re.compile(r"^(IC|IF|IH|IM)\d{4}$")

sess = ddb.session(HOST, PORT); sess.login(USER, PW)
k = sess.run("""
select date, code, close, settlement from loadTable("dfs://future_day_kline","DayKLine")
where code_init in [`IC,`IF,`IH,`IM]
""")
info = sess.run("""
select code, max(last_trade_day) as ltd from loadTable("dfs://future_day_kline","DailyInstrumentInfo")
where code_init in [`IC,`IF,`IH,`IM] group by code
""")
sess.close()

k["date"] = pd.to_datetime(k["date"]).dt.normalize()
k = k[k.close > 0].drop_duplicates(["date", "code"])
fut = k[k.code.str.match(MONTHLY)].copy()
fut["product"] = fut.code.str[:2]
info["ltd"] = pd.to_datetime(info["ltd"].astype(str), format="%Y%m%d")
fut = fut.merge(info[["code", "ltd"]], on="code", how="left").dropna(subset=["ltd"])
fut["days"] = (fut["ltd"] - fut["date"]).dt.days
fut = fut[fut["days"] >= 0]
fut["rank"] = fut.groupby(["date", "product"])["days"].rank(method="first").astype(int)

# ---- leg 1: daily term-structure slope carry --------------------------------
w = fut[fut["rank"] <= 4].pivot_table(index=["date", "product"], columns="rank",
                                      values=["close", "days"])
rows = []
for (i, j) in [(1, 2), (1, 4)]:
    fi, fj = w[("close", i)], w[("close", j)]
    di, dj = w[("days", i)], w[("days", j)]
    ok = fi.notna() & fj.notna() & (di >= 3) & ((dj - di) >= 10)
    ann = (fi - fj) / fj * 365.0 / (dj - di) * 100
    s = ann[ok].reset_index()
    s.columns = ["date", "product", "ann_pct"]
    s["pair"] = f"{i}-{j}"
    rows.append(s)
slope = pd.concat(rows).sort_values(["product", "pair", "date"])
slope.to_csv(f"{D}/carry_slope.csv", index=False)

# ---- leg 2: realized carry anchored at delivery settlements -----------------
res = []
for prod, g in fut.groupby("product", sort=True):
    # delivery settlement: settlement on the contract's own last trade day
    exp = {}
    for code, gc in g.groupby("code"):
        gl = gc[gc.date == gc.ltd]
        if len(gl) and np.isfinite(gl.settlement.iloc[0]) and gl.settlement.iloc[0] > 0:
            exp[code] = (gl.date.iloc[0], float(gl.settlement.iloc[0]))
    order = sorted(exp, key=lambda c: exp[c][0])
    px = g.pivot_table(index="date", columns="code", values="close")
    for prev, cur in zip(order[:-1], order[1:]):
        e0, ds0 = exp[prev]; e1, ds1 = exp[cur]
        if cur not in px.columns or e0 not in px.index:
            continue
        f0 = px.at[e0, cur]
        if not np.isfinite(f0):
            continue
        wind = (e1 - e0).days
        if not 20 <= wind <= 45:                 # keep the monthly roll chain only
            continue
        fr = ds1 / f0 - 1; sr = ds1 / ds0 - 1
        res.append(dict(product=prod, code=cur, entry_date=e0, exit_date=e1,
                        window_d=wind, entry_basis_pct=(f0 - ds0) / ds0 * 100,
                        fut_ret_pct=fr * 100, spot_ret_pct=sr * 100,
                        carry_pct=(fr - sr) * 100,
                        ann_pct=(fr - sr) * 100 * 365 / wind,
                        exp_month=e1.month))
real = pd.DataFrame(res).sort_values(["product", "exit_date"])
real.to_csv(f"{D}/carry_realized.csv", index=False)
print(f"saved carry_slope.csv ({len(slope)}) carry_realized.csv ({len(real)})")
print(real.groupby("product")["carry_pct"].agg(["count", "mean", "std"]).round(3))
