"""minute_open_window_pnl.py — actual P&L of the momentum strategy in the OPEN 30 minutes.
Strategy: each minute M in 09:30-10:00 with valid signal, take position sign(close(M-1)-close(M-2)),
hold the whole minute, P&L = sign(dprev)*(close-open). Reports gross, net after costs, in 元.
Run: python3 minute_open_window_pnl.py"""
import pandas as pd, numpy as np

MULT = {"IC0000": 200, "IF0000": 300, "IH0000": 300, "IM0000": 200}   # 元/指数点
NAME = {"IC0000": "IC 中证500", "IF0000": "IF 沪深300", "IH0000": "IH 上证50", "IM0000": "IM 中证1000"}
COSTS = [0.0, 0.1, 0.2, 0.4]            # 单次往返成本(指数点): 0=毛, 0.2≈跨一次价差
OPEN_LO, OPEN_HI = 570, 600             # 09:30..10:00

df = pd.read_csv("/Users/zhuisabella/xn/intraminute/minute_conditional_isoos.csv")
df = df[(df.tod >= OPEN_LO) & (df.tod <= OPEN_HI)].copy()
df["pnl"] = np.sign(df.dprev) * df.ret

print(f"开盘窗口 09:30-10:00 | 数据=12个平静月(6 IS+6 OOS) | 策略=动量(信号×整分钟)\n")
for code in NAME:
    c = df[df.code == code]
    ndays = c.date.nunique(); ntr = len(c)
    gross = c.pnl.sum(); per_tr = c.pnl.mean(); hit = (c.pnl > 0).mean()
    per_day = gross / ndays
    m = MULT[code]
    print(f"=== {NAME[code]} (乘数 {m} 元/点) ===")
    print(f"  天数={ndays}  交易数={ntr}  每天约{ntr/ndays:.0f}笔  胜率={hit*100:.1f}%")
    print(f"  毛利合计={gross:+.1f}点  每笔={per_tr:+.4f}点  每天={per_day:+.3f}点")
    print(f"  毛利折钱: 合计={gross*m:+,.0f}元  每天={per_day*m:+,.0f}元  年化(242天)={per_day*m*242:+,.0f}元/手")
    for cost in COSTS:
        net = gross - ntr * cost
        tag = "毛" if cost == 0 else f"成本{cost}点/往返"
        print(f"    净利({tag}): 合计={net:+.1f}点 = {net*m:+,.0f}元  每天={net/ndays*m:+,.0f}元  年化={net/ndays*242*m:+,.0f}元/手")
    print()

# 组合：四合约等手数合计
allc = df.copy()
print("=== 四合约各1手 组合 ===")
for cost in COSTS:
    tot_day = 0; tot_year = 0
    for code in NAME:
        c = df[df.code == code]; ndays = c.date.nunique()
        net = c.pnl.sum() - len(c) * cost
        tot_day += net / ndays * MULT[code]; tot_year += net / ndays * 242 * MULT[code]
    tag = "毛" if cost == 0 else f"成本{cost}点/往返"
    print(f"  {tag}: 每天合计={tot_day:+,.0f}元  年化合计={tot_year:+,.0f}元")
