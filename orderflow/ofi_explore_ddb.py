"""ofi_explore_ddb.py — 方向A 首发：订单流失衡 (OFI) 与价格 (IM, 分年度)。
两种订单流：
  OFI (Cont-Kukanov-Stoikov 盘口事件流，由最优买卖价+量变化构造)：
    e_n = 1[Pb≥Pb⁻]·qb − 1[Pb≤Pb⁻]·qb⁻ − 1[Pa≤Pa⁻]·qa + 1[Pa≥Pa⁻]·qa⁻   (>0=买压)
  TFI (trade-flow): 主动买量 − 主动卖量 = ActBidVol − ActAskVol            (>0=主买)
两种关系（关键区分）：
  ① 同期 contemp（价格冲击/解释，应很强，复现 CKS、验证OFI算对）：
       X = 同一窗口 [i,i+L] 的订单流和；Y = mid[i+L]−mid[i]。看 R²(=corr²)、斜率β。
  ② 预测 pred（真·alpha，预测未来，应该弱）：
       X = 过去 w 个 tick 的订单流和；Y = 未来 mid[i+x]−mid[i]。看 corr、方向命中率(剔除Y=0)。
源头滤坏报价 bid>0&ask>0；窗口不跨午休/隔夜。IM, 2022-07..2026-05.
Output ofi_explore.csv: year,mode(contemp/pred),kind(ofi/tfi),w,x, n,Sx,Sy,Sxx,Syy,Sxy, hit,nynz
Run: /Users/zhuisabella/xn/.venv/bin/python ofi_explore_ddb.py   (sandbox OFF)
"""
import calendar
import numpy as np
import pandas as pd
import dolphindb as ddb
from ddb_config import HOST, PORT, USER, PW

CODE = "IM0000"
LS = [1, 5, 10, 20, 50]                   # contemp 同期窗口
WS = [5, 10, 20]                          # pred 过去订单流窗口
XS = [5, 10, 20]                          # pred 未来价格窗口
WINDOWS = [(y, m) for y in range(2022, 2027) for m in range(1, 13)
           if (2022, 7) <= (y, m) <= (2026, 5)]
OUT = "/Users/zhuisabella/xn/orderflow/ofi_explore.csv"


def fetch(sess, start, end):
    q = f"""
    pt=loadTable("dfs://hft_future_ts","TickPartitioned")
    select m_nDatetime as ts, m_nBidPrice as bid, m_nAskPrice as ask,
           m_nBidVolume as bidv, m_nAskVolume as askv,
           m_nActBidVolume as actbidv, m_nActAskVolume as actaskv
    from pt where code_init=`IM, m_nDatetime>={start}T00:00:00,
          m_nDatetime<={end}T23:59:59, code=`{CODE},
          m_nBidPrice>0, m_nAskPrice>0,
          minute(m_nDatetime) between 09:30m:15:00m
    """
    return sess.run(q)


def blocks(df):
    df = df.drop_duplicates("ts").sort_values("ts").copy()
    tod = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    df = df[((tod >= 570) & (tod <= 690)) | ((tod >= 780) & (tod <= 900))]
    if df.empty:
        return
    df["day"] = df["ts"].dt.normalize()
    df["session"] = np.where(tod[df.index] <= 690, "AM", "PM")
    for _, g in df.groupby(["day", "session"], sort=False):
        yield (g["bid"].to_numpy(float), g["ask"].to_numpy(float),
               g["bidv"].to_numpy(float), g["askv"].to_numpy(float),
               g["actbidv"].to_numpy(float), g["actaskv"].to_numpy(float))


def per_tick_ofi(bid, ask, bidv, askv):
    L = len(bid); e = np.zeros(L)
    db = bid[1:] - bid[:-1]; da = ask[1:] - ask[:-1]
    e[1:] = ((db >= 0) * bidv[1:] - (db <= 0) * bidv[:-1]
             - (da <= 0) * askv[1:] + (da >= 0) * askv[:-1])
    return e


def _acc(acc, key, X, Y):
    ok = np.isfinite(X) & np.isfinite(Y)
    X, Y = X[ok], Y[ok]
    if not X.size:
        return
    nz = Y != 0
    a = acc.setdefault(key, [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0])
    a[0] += int(X.size)
    a[1] += X.sum(); a[2] += Y.sum()
    a[3] += float((X * X).sum()); a[4] += float((Y * Y).sum()); a[5] += float((X * Y).sum())
    a[6] += int(((np.sign(X) == np.sign(Y)) & nz).sum())
    a[7] += int(nz.sum())


def accumulate(arrs, acc, yr):
    bid, ask, bidv, askv, actbidv, actaskv = arrs
    L = len(bid)
    if L < max(LS) + max(XS) + max(WS) + 2:
        return
    mid = (bid + ask) / 2.0
    for kind, pt in (("ofi", per_tick_ofi(bid, ask, bidv, askv)), ("tfi", actbidv - actaskv)):
        cs = np.concatenate([[0.0], np.cumsum(pt)])
        # ① 同期 contemp
        for Lw in LS:
            i = np.arange(0, L - Lw)
            X = cs[i + Lw + 1] - cs[i + 1]          # 订单流 over ticks (i, i+Lw]
            Y = mid[i + Lw] - mid[i]
            _acc(acc, (yr, "contemp", kind, Lw, Lw), X, Y)
        # ② 预测 pred
        Xpast = {}
        for w in WS:
            xw = np.full(L, np.nan); xw[w - 1:] = cs[w:L + 1] - cs[0:L - w + 1]
            Xpast[w] = xw
        for w in WS:
            for x in XS:
                i = np.arange(max(w - 1, 1), L - x)
                _acc(acc, (yr, "pred", kind, w, x), Xpast[w][i], mid[i + x] - mid[i])


def stats(g):
    n = g.n.sum(); Sx = g.Sx.sum(); Sy = g.Sy.sum()
    Sxx = g.Sxx.sum(); Syy = g.Syy.sum(); Sxy = g.Sxy.sum()
    hit = g.hit.sum(); nynz = g.nynz.sum()
    cov = Sxy - Sx * Sy / n; vx = Sxx - Sx * Sx / n; vy = Syy - Sy * Sy / n
    corr = cov / np.sqrt(vx * vy) if vx > 0 and vy > 0 else float("nan")
    beta = cov / vx if vx > 0 else float("nan")
    return corr, beta, 100 * hit / nynz if nynz else float("nan"), int(n)


if __name__ == "__main__":
    sess = ddb.session(HOST, PORT); sess.login(USER, PW)
    acc = {}
    for yr, mo in WINDOWS:
        last = calendar.monthrange(yr, mo)[1]
        start, end = f"{yr}.{mo:02d}.01", f"{yr}.{mo:02d}.{last:02d}"
        df = fetch(sess, start, end)
        if not len(df):
            continue
        for arrs in blocks(df):
            accumulate(arrs, acc, yr)
        print(f"{yr}-{mo:02d} done", flush=True)
    sess.close()
    rows = [{"year": yr, "mode": md, "kind": k, "w": w, "x": x, "n": a[0],
             "Sx": a[1], "Sy": a[2], "Sxx": a[3], "Syy": a[4], "Sxy": a[5],
             "hit": a[6], "nynz": a[7]} for (yr, md, k, w, x), a in acc.items()]
    d = pd.DataFrame(rows)
    d.sort_values(["mode", "kind", "w", "x", "year"]).to_csv(OUT, index=False)
    print(f"saved {OUT} ({len(rows)})")

    print("\n=== ① 同期 contemp（价格冲击/解释力，应强）: R² | beta | n ===")
    for kind in ["ofi", "tfi"]:
        print(f"--- {kind.upper()} ---")
        for Lw in LS:
            g = d[(d["mode"] == "contemp") & (d.kind == kind) & (d.w == Lw)]
            corr, beta, _, n = stats(g)
            print(f"  窗口 L={Lw:>2}: R²={corr**2:.3f}  beta={beta:+.4f}  n={n//1000}k")
    print("\n=== ② 预测 pred（过去订单流→未来价格，真alpha，应弱）: corr | 命中% | n ===")
    for kind in ["ofi", "tfi"]:
        print(f"--- {kind.upper()} ---")
        for w in WS:
            for x in XS:
                g = d[(d["mode"] == "pred") & (d.kind == kind) & (d.w == w) & (d.x == x)]
                corr, beta, hitp, n = stats(g)
                print(f"  过去w={w:>2} 未来x={x:>2}: corr={corr:+.3f}  命中={hitp:4.1f}%  n={n//1000}k")
