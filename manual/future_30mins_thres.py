from tick_to_min import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

sys.path.insert(0, "/Users/zhuisabella/xn/prediction")
sys.path.insert(0, "/Users/zhuisabella/xn/manual")
from ddb_config import HOST, PORT, USER, PW

_av = {f.name for f in fm.fontManager.ttflist}
for _f in ["Arial Unicode MS", "PingFang HK", "Heiti TC", "STHeiti", "Songti SC"]:
    if _f in _av:
        matplotlib.rcParams["font.sans-serif"] = [_f]; break
matplotlib.rcParams["axes.unicode_minus"] = False

PILOT = os.environ.get("PILOT") == "1" #pilot = june only fast run
SUF = "_pilot" if PILOT else ""
D = "/Users/zhuisabella/xn/manual"
CODES = ["IF0000", "IC0000", "IH0000", "IM0000"]
H = 30
THRESH = 0.001
START, END = ("2024.06.01", "2024.06.30") if PILOT else ("2010.01.01", "2026.07.15")

sess = ddb.session(HOST, PORT); sess.login(USER, PW)
bars = {c: to_session_bars(fetch_min_bars(sess, c, START, END), c) for c in CODES}
sess.close()

for CODE in CODES:
    b = bars[CODE]

    day = b.ts.dt.normalize()
    pm = (b.ts.dt.hour >= 13).astype(int) #0 for morning, 1 for afternoon
    g = b.groupby([day, pm])["mid_close"]
    ret = g.pct_change(fill_method=None) # minute t's return; don't ffill NaN (limit-up) mids
    sig = np.sign(ret).where(ret.abs() > THRESH)
    fwd = (g.shift(-H) - b["mid_close"]) / b["mid_close"] * 1e4
    b["strat"] = sig * fwd

    b["hm"] = b.ts.dt.strftime("%H:%M")
    s = (b.dropna(subset=["strat"]).groupby("hm")["strat"]
           .agg(mean="mean", sd="std", n="count").reset_index())
    s["se"] = s.sd / np.sqrt(s.n)
    s["t"] = s["mean"] / s["se"]
    s.to_csv(f"{D}/fwd{H}_momentum_{CODE}{SUF}_thres.csv", index=False)

    pooled = b["strat"].dropna()
    print(f"{CODE} {START}..{END}  H={H}min  trades={len(pooled)}  days={day.nunique()}")
    print(f"pooled mean {pooled.mean():+.2f} bps (indicative t={pooled.mean()/pooled.std()*np.sqrt(len(pooled)):.1f}, overlap-inflated)")
    print(f"minutes with |t|>=2: {(s.t.abs() >= 2).sum()} / {len(s)}")

    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(s))
    ax.axhline(0, color="0.5", lw=.7)
    # ax.fill_between(x, s["mean"] - 2 * s.se, s["mean"] + 2 * s.se,
    #                 color="#4C72B0", alpha=.25, label="±2·se")
    ax.plot(x, s["mean"], lw=1.4, color="#4C72B0", label="均值")
    tick = [i for i, h in enumerate(s.hm) if h.endswith(("00", "15", "30", "45"))]
    ax.set_xticks(tick); ax.set_xticklabels(s.hm.iloc[tick], fontsize=7)
    ax.set_xlabel("日内分钟 t（信号 = 第 t 分钟方向）")
    ax.set_ylabel(f"动量策略收益（bps，持有 {H} 分钟）")
    ax.set_title(f"{CODE} 分钟动量：sign(第t分钟涨跌) × 未来{H}分钟平均收益, 阈值 {THRESH:.1%}"
                 f"（{START}–{END}）", fontsize=11, fontweight="bold")
    ax.xaxis.set_minor_locator(AutoMinorLocator(3))    # 2 minor ticks between majors -> 10-min steps
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))    # halve each y interval
    ax.grid(True, which="major", alpha=.35)
    ax.grid(True, which="minor", alpha=.12, lw=.5)     # fainter so it doesn't drown the data
    ax.legend()
    fig.tight_layout(); fig.savefig(f"{D}/fig_fwd{H}_momentum_{CODE}{SUF}_thres.png", dpi=135)
    plt.close(fig)
    print(f"saved fwd{H}_momentum_{CODE}{SUF}_thres.csv + fig")

