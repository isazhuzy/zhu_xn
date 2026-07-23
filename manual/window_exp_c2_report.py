"""Render the three deliverables from window_exp_c2_all.csv:
  (1) per-contract FORWARD table  (look-back fixed = instant, cols = hold time)
  (2) per-contract LOOK-BACK table (forward fixed = 10s,   cols = look-back time)
  (3) STRONGEST-window note-style table: per factor pick the (J,k) with the largest
      cross-contract signal |S|=|最正0.1%-最负0.1%|/2, then buckets x contracts.
Price basis: voi/oir = LAST, mpb = MID (per the note's rule; stored in the csv).
"""
import numpy as np
import pandas as pd

D = "/Users/zhuisabella/xn/manual"
df = pd.read_csv(f"{D}/window_exp_c2_all.csv")
SEC = {1: "0.5s", 5: "2.5s", 20: "10s", 60: "30s", 120: "60s", 240: "120s", 600: "300s"}
TIMES = [1, 5, 20, 60, 120, 240, 600]
CODES = ["IC0000", "IF0000", "IH0000", "IM0000"]
FACS = ["voi", "mpb", "oir"]
BAS = {"voi": "last", "oir": "last", "mpb": "mid"}
LAB = {"neg0.1": "最负0.1%", "neg1": "最负1%", "pos1": "最正1%", "pos0.1": "最正0.1%"}
ORDER = ["neg0.1", "neg1", "pos1", "pos0.1"]


def val(fac, code, b, J, k):
    v = df[(df.factor == fac) & (df.code == code) & (df.bucket == b) &
           (df.J == J) & (df.k == k)]["mean_dy"]
    return v.iloc[0] if len(v) else np.nan


def md_header(cols):
    return ("| 因子 | 桶 | " + " | ".join(cols) + " |\n"
            "|---|---|" + "---|" * len(cols))


def per_contract(fixed_axis):
    """fixed_axis='fwd' -> cols=hold, J=1 ; 'back' -> cols=lookback, k=20"""
    out = []
    cols = [SEC[t] for t in TIMES]
    for code in CODES:
        out.append(f"\n### {code}\n")
        out.append(md_header(cols))
        for fac in FACS:
            for i, b in enumerate(ORDER):
                fn = f"{fac.upper()}({BAS[fac]})" if i == 0 else ""
                if fixed_axis == "fwd":
                    cells = [val(fac, code, b, 1, t) for t in TIMES]
                else:
                    cells = [val(fac, code, b, t, 20) for t in TIMES]
                out.append(f"| {fn} | {LAB[b]} | " +
                           " | ".join(f"{c:+.2f}" for c in cells) + " |")
    return "\n".join(out)


def strongest():
    # score on the 1% tail (stable, ~10x samples), exclude the 300s edges on both
    # axes: those windows overlap 599/600 -> tiny effective sample, noisy |S| spikes.
    SEARCH = [t for t in TIMES if t <= 240]      # up to 120s only
    out = []
    for fac in FACS:
        best = None
        for J in SEARCH:
            for k in SEARCH:
                s = [abs(val(fac, c, "pos1", J, k) - val(fac, c, "neg1", J, k)) / 2
                     for c in CODES]
                sc = np.nanmean(s)
                if best is None or sc > best[0]:
                    best = (sc, J, k)
        sc, J, k = best
        sign = "动量" if np.nanmean([val(fac, c, "pos1", J, k) - val(fac, c, "neg1", J, k)
                                  for c in CODES]) > 0 else "反转"
        out.append(f"\n### {fac.upper()} ({BAS[fac]}) — 最强窗口: 回看 {SEC[J]} / 持有 {SEC[k]} "
                   f"（1%尾|S|均值={sc:.2f} tick，{sign}；已排除300s重叠噪声边）\n")
        out.append("| 桶 | " + " | ".join(CODES) + " |")
        out.append("|---|" + "---|" * len(CODES))
        for b in ORDER:
            out.append(f"| {LAB[b]} | " +
                       " | ".join(f"{val(fac, c, b, J, k):+.2f}" for c in CODES) + " |")
    return "\n".join(out)


doc = ["# Experiment C v2 — 最新价(VOI/OIR)/中间价(MPB), 统一时间刻度 0.5s~300s\n",
       "值=尾部平均前向收益 tick/笔, 毛, +涨/-跌. 全样本2020-2026.\n",
       "\n## 表1 — 每 contract：列=持有时间(forward)，回看固定=瞬时1tick",
       per_contract("fwd"),
       "\n## 表2 — 每 contract：列=回看时间(look-back)，持有固定=10s",
       per_contract("back"),
       "\n## 表3 — 各因子最强(回看,持有)窗口下的 桶×contract 表（复刻笔记§五格式）",
       strongest()]
open(f"{D}/window_exp_c2_TABLES.md", "w").write("\n".join(doc))
print("\n".join(doc))
print("\nsaved window_exp_c2_TABLES.md")
