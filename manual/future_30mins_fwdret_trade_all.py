from tick_to_min import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.ticker import AutoMinorLocator

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
PX = "close"       # last traded price, not mid
START, END = ("2024.06.01", "2024.06.30") if PILOT else ("2010.01.01", "2026.07.15")

sess = ddb.session(HOST, PORT); sess.login(USER, PW)
bars = {c: to_session_bars(fetch_min_bars(sess, c, START, END), c) for c in CODES}
sess.close()

for CODE in CODES:
    b = bars[CODE]

    # signed forward return: trade minute t's direction, raw fraction (NOT bps)
    day = b.ts.dt.normalize()
    g = b.groupby(day)[PX]   # day-only fence: windows cross the lunch break
    sig = np.sign(g.diff())                       # minute t's own direction
    sig = sig.replace(0, np.nan)                  # flat minute -> no trade
    b["fwdret"] = sig * (g.shift(-H) - b[PX]) / b[PX]   # hold H min in sig's direction

    b["hm"] = b.ts.dt.strftime("%H:%M")
    s = (b.dropna(subset=["fwdret"]).groupby("hm")["fwdret"]
           .agg(mean="mean", sd="std", n="count").reset_index())
    s["se"] = s.sd / np.sqrt(s.n)
    s["t"] = s["mean"] / s["se"]
    s.to_csv(f"{D}/fwd{H}_ret_trade_all_{CODE}{SUF}.csv", index=False)

    pooled = b["fwdret"].dropna()
    print(f"{CODE} {START}..{END}  H={H}min  px={PX}  obs={len(pooled)}  days={day.nunique()}")
    print(f"pooled mean {pooled.mean():+.6f} (indicative t={pooled.mean()/pooled.std()*np.sqrt(len(pooled)):.1f}, overlap-inflated)")
    print(f"minutes with |t|>=2: {(s.t.abs() >= 2).sum()} / {len(s)}")

    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(s))
    ax.axhline(0, color="0.5", lw=.7)
    ax.plot(x, s["mean"], lw=1.4, color="#4C72B0", label="均值")
    tick = [i for i, h in enumerate(s.hm) if h.endswith(("00", "15", "30", "45"))]
    ax.set_xticks(tick); ax.set_xticklabels(s.hm.iloc[tick], fontsize=7)
    ax.xaxis.set_minor_locator(AutoMinorLocator(3))   # 5-min minor gridlines
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlabel("日内分钟 t（信号 = 第 t 分钟方向）")
    ax.set_ylabel(f"未来{H}分钟平均收益（原始比例）")
    ax.set_title(f"{CODE} sign(第t分钟涨跌) × 未来{H}分钟平均收益（最新价，原始比例）"
                 f"（{START}–{END}）", fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(True, which="major", alpha=.35)
    ax.grid(True, which="minor", alpha=.12, lw=.5)
    fig.tight_layout(); fig.savefig(f"{D}/fig_fwd{H}_ret_trade_all_{CODE}{SUF}.png", dpi=135)
    plt.close(fig)
    print(f"saved fwd{H}_ret_trade_all_{CODE}{SUF}.csv + fig")
