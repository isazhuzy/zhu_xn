"""
replication_sample.py — selective out-of-sample check of the previous-minute reversal.

Re-run the EXACT step-4 trade (prev-minute signal, fade, enter at tick N, hold a full
minute = exit at tick N of next minute) on several SEPARATE Jan–Feb windows across years,
at fixed N=10 and N=20. Question: does IH's reversal replicate year to year, or was 2023
a fluke? Selective sampling (6 windows, not all 11 years).

Compute only (DolphinDB on .venv); replication_plot.py (3.14) draws it.
Run:  /Users/zhuisabella/xn/.venv/bin/python replication_sample.py   (sandbox off)
"""
import pandas as pd
import dolphindb as ddb
import intraminute_wholeday_ddb as iw

iw.N_LIST = [16]                          # N=16 = the 2023 two-month highest-return entry delay
WINDOWS = [("2016.01.01", "2016.02.28"),  # NB: 2016–17 CFFEX index futures were restricted
           ("2019.01.01", "2019.02.28"),
           ("2022.01.01", "2022.02.28"),
           ("2023.01.01", "2023.02.28"),  # the original window
           ("2024.01.01", "2024.02.28"),
           ("2025.01.01", "2025.02.28")]
OUTCSV = "/Users/zhuisabella/xn/ticker/open_breakdown/replication_sample.csv"


if __name__ == "__main__":
    sess = ddb.session(iw.HOST, iw.PORT); sess.login(iw.USER, iw.PW)
    rows = []
    for start, end in WINDOWS:
        yr = start[:4]
        for code in iw.CODES:
            df = iw.fetch(sess, code, start, end)
            if df is None or len(df) < 5000:
                print(f"{yr} {code}: no/low data ({0 if df is None else len(df)}), skip", flush=True)
                continue
            t = iw.contract_trades(df, code)
            ndays = t["day"].nunique()
            for N in iw.N_LIST:
                st = iw.agg(t[t.N == N])
                st["se_bp"] = round(abs(st["mean_bp"] / st["t"]), 3) if st["t"] else float("nan")
                rows.append({"year": yr, "code": code, "N": N, "ndays": ndays, **st})
            print(f"{yr} {code}: {len(df)} ticks, {ndays} days  "
                  f"IH-style done", flush=True)
    sess.close()
    tab = pd.DataFrame(rows)
    tab.to_csv(OUTCSV, index=False)
    pd.set_option("display.width", 220, "display.max_rows", 300)
    for N in iw.N_LIST:
        print(f"\n=== 反转 mean_bp / t ｜ N={N} ｜ 各年 1–2月 × 各合约 ===")
        sub = tab[tab.N == N]
        print("mean_bp:")
        print(sub.pivot(index="year", columns="code", values="mean_bp").to_string())
        print("t:")
        print(sub.pivot(index="year", columns="code", values="t").to_string())
    print("\nsaved", OUTCSV)
