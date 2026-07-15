import sys
sys.path.insert(0, "/Users/zhuisabella/xn/manual")
from tick_to_min import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

sys.path.insert(0, "/Users/zhuisabella/xn/prediction")
from ddb_config import HOST, PORT, USER, PW

_av = {f.name for f in fm.fontManager.ttflist}
for _f in ["Arial Unicode MS", "PingFang HK", "Heiti TC", "STHeiti", "Songti SC"]:
    if _f in _av:
        matplotlib.rcParams["font.sans-serif"] = [_f]; break
matplotlib.rcParams["axes.unicode_minus"] = False

PILOT = os.environ.get("PILOT") == "1"
SUF = "_pilot" if PILOT else ""
D = "/Users/zhuisabella/xn/last"
CODES = ["IF0000", "IC0000", "IH0000", "IM0000"]
H = int(os.environ.get("H", "30"))
THRESH = float(os.environ.get("THRESH", "0.001"))     # min |minute-t return| to trade
START, END = ("2024.06.01", "2024.06.30") if PILOT else ("2024.01.01", "2024.12.31")

sess = ddb.session(HOST, PORT); sess.login(USER, PW)
bars = {c: to_session_bars(fetch_min_bars(sess, c, START, END), c) for c in CODES}
sess.close()

for CODE in CODES:
    b = bars[CODE]

    # signal & forward return, both inside (day, session) so nothing crosses a break
    day = b.ts.dt.normalize()
    pm = (b.ts.dt.hour >= 13).astype(int)
    g = b.groupby([day, pm])["mid_close"]
    ret = g.pct_change()                                  # minute t's own return
    sig = np.sign(ret).where(ret.abs() > THRESH)          # trade only if move > THRESH
    fwd = (g.shift(-H) - b["mid_close"]) / b["mid_close"] * 1e4   # next-H-min move, bps
    b["strat"] = sig * fwd

    # average per minute-of-day: one sample per day per cell -> clean se/t
    b["hm"] = b.ts.dt.strftime("%H:%M")
    s = (b.dropna(subset=["strat"]).groupby("hm")["strat"]
           .agg(mean="mean", sd="std", n="count").reset_index())
    s["se"] = s.sd / np.sqrt(s.n)
    s["t"] = s["mean"] / s["se"]
    s.to_csv(f"{D}/fwd{H}_momentum_thr_{CODE}{SUF}.csv", index=False)

    pooled = b["strat"].dropna()
    print(f"{CODE} {START}..{END}  H={H}min  thresh={THRESH:.4f}  trades={len(pooled)}  days={day.nunique()}")
    print(f"pooled mean {pooled.mean():+.2f} bps (indicative t={pooled.mean()/pooled.std()*np.sqrt(len(pooled)):.1f}, overlap-inflated)")
    print(f"minutes with |t|>=2: {(s.t.abs() >= 2).sum()} / {len(s)}")

    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(s))
    ax.axhline(0, color="0.5", lw=.7)
    ax.plot(x, s["mean"], lw=1.4, color="#4C72B0", label="均值")
    tick = [i for i, h in enumerate(s.hm) if h.endswith(("00", "30"))]
    ax.set_xticks(tick); ax.set_xticklabels(s.hm.iloc[tick], fontsize=8)
    ax.set_xlabel("日内分钟 t（信号 = 第 t 分钟方向，|涨跌幅| > 阈值才交易）")
    ax.set_ylabel(f"动量策略收益（bps，持有 {H} 分钟）")
    ax.set_title(f"{CODE} 分钟动量（阈值 {THRESH:.1%}）：sign(第t分钟涨跌) × 未来{H}分钟平均收益"
                 f"（{START}–{END}）", fontsize=11, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=.3)
    fig.tight_layout(); fig.savefig(f"{D}/fig_fwd{H}_momentum_thr_{CODE}{SUF}.png", dpi=135)
    plt.close(fig)
    print(f"saved fwd{H}_momentum_thr_{CODE}{SUF}.csv + fig")
