# xn/prediction — 盘口信息对价格的预测作用 (Order-Book Information & Price Prediction)

研究问题：**股指期货的盘口（Level-1 报价：买一/卖一价与量、主动成交量、成交额）对未来价格有没有、有多少、在什么期限上有预测作用？**

我们把四篇经典文献的方法逐一搬到自己的数据上（CFFEX IC/IF/IH/IM 连续合约，
DolphinDB `hft_future_ts`，500ms 快照，2020-01 … 2026-05，约 1.3 亿行），
每篇一个独立脚本 + 图 + 学习指南。

---

## 四篇论文，一句话各是什么

| # | 论文 | 一句话 | 指南 | 代码 |
|---|---|---|---|---|
| 1 | **Cont, Kukanov & Stoikov (2014)**, *The Price Impact of Order Book Events*, J. Fin. Econometrics | 同一区间内 ΔP ≈ OFI/(2·深度)：订单流失衡**解释**价格变动（R²≈0.5），但**不预测**下一区间 | [STUDY_GUIDE_OFI.md](STUDY_GUIDE_OFI.md) | `ofi_full_ddb.py` |
| 2 | **Gould & Bonart (2016)**, *Queue Imbalance as a One-Tick-Ahead Price Predictor*, Mkt Microstructure & Liquidity | 买一卖一队列失衡 I=(qb−qa)/(qb+qa) **预测下一次中间价变动的方向**，命中率远超50% | [STUDY_GUIDE_QI.md](STUDY_GUIDE_QI.md) | `qi_ddb.py` |
| 3 | **Shen (2015)**, *Order Imbalance Based Strategy in HFT*, Oxford MSc thesis（用的就是 CFFEX 沪深300期货 2014 年 500ms 数据——和我们同一张"盘口"） | VOI(盘口增量) + OIR(盘口水平) + MPB(成交价基差) 经价差归一后线性**预测未来数秒的平均中间价变动**，样本外成立 | [STUDY_GUIDE_VOI.md](STUDY_GUIDE_VOI.md) | `voi_ddb.py` |
| 4 | **Stoikov (2018)**, *The Micro-Price*, Quantitative Finance | 给定盘口状态，"公允价" = mid + g*(失衡,价差)，用马尔可夫链估计 g*；比 mid / 加权 mid 更好地预测未来 mid | [STUDY_GUIDE_MICROPRICE.md](STUDY_GUIDE_MICROPRICE.md) | `microprice_ddb.py` |

原文链接：CKS OFI ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1712822))；
Gould-Bonart ([arXiv:1512.03492](https://arxiv.org/abs/1512.03492))；
Shen ([Oxford eprints](http://eprints.maths.ox.ac.uk/1895/))；
Stoikov ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2970694))。

---

## 数据与共同方法

- **表**：`dfs://hft_future_ts` / `TickPartitioned`（192.168.1.7:8848，`useread`；只在关沙箱时可达）。
  用到的字段：`m_nBid/AskPrice`, `m_nBid/AskVolume`（盘口一档），`m_iVolume`, `m_iTurnover`
  （增量成交量/额 → 平均成交价），`m_nActBid/AskVolume`（主动成交方向）。
- **共同预处理**（`lob_common.py`）：ts 去重（2024-02 重复行）、丢弃单边/交叉报价、
  交易时段 09:30–11:30 / 13:00–15:00 且**永不跨时段取差分**、价差 >50 tick 或
  |Δmid|>50 tick 视为坏 tick（IH 2024-10-08 半价报价事件）。tick=0.2，一切价格量以 tick 计。
- **逐月分块 + 充分统计量累加**（矩阵 XᵀX、Xᵀy、计数矩阵），全样本合并结果是**精确**的，
  内存里从不放整段历史。
- **训练/测试纪律**（论文3、4）：**2020-01…2024-12 估计参数并冻结，2025-01…2026-05 纯样本外**。
- 已知数据坑（详见 memory / 各脚本注释）：2023-07 整月缺失；2022-12 真实加密采样；
  IM 自 2022-07 才有。

## 总结果一览（全样本；论文3/4为2025-26样本外）

| 方法 | 关键量 | IC | IF | IH | IM |
|---|---|---|---|---|---|
| #1 OFI 同期解释 | R²（10s） | 0.50 | 0.53 | 0.53 | 0.47 |
| #1 OFI 预测下一区间 | R²（10s） | ~0.001 | ~0.001 | ~0.001 | ~0.0005 |
| #2 QI 方向命中率 | sign(I)（全期） | 58.4% | 61.0% | 61.5% | 58.4% |
| #2 QI 命中率（价差=1 tick时） |  | 63.1% | 66.2% | 66.7% | 62.5% |
| #3 VOI 模型B | 样本外 R²（0.5s） | 5.2% | 8.7% | 8.2% | 7.0% |
| #3 VOI 模型B | 样本外 R²（10s） | 1.0% | 1.5% | 1.7% | 1.0% |
| #3 强信号命中率 | \|ŷ\|>0.5tick | 58.0% | 60.5% | 62.5% | 57.8% |
| #4 微观价格 | 0.5s RMSE 相对 mid | −0.6% | −1.2% | +0.2% | −0.8% |

（#2–#4 的精确全样本数字以 `qi_results.csv` / `voi_results.csv` / `mp_rmse.csv` 为准。
注：#4 用 2020–24 冻结参数测 2025–26，g* 幅度轻微过度修正（见指南4 §5）；
邻月估计的试跑版四个合约全部跑赢 mid。）

图：`fig61–65` OFI；`fig71–73` QI；`fig81–86` VOI + 回测；`fig91–93` 微观价格。

**阈值策略回测（论文3闭环，`voi_backtest.py`，图84–85）**：冻结系数、2025-26 样本外、
|ŷ|>q 开仓持有10s、非重叠。信号价值（mid进出）每笔 **+0.3~+1.9 tick、t值17–80——真实存在**；
但同快照吃单即 −0.14~−4.5 tick（最好情形 IF q=2 也差 0.14 tick 才够本），
延迟500ms再亏 0.7–3 tick（预测的变动在第一个快照内就兑现），平昨费率再亏 ~1 tick，
平今费率（3.45bp）直接 −7~−17 tick。q=0.5 时净损益 ≈ 每手每天 −12~−25万元（Sharpe −80~−120）。
**"信号在价差内"由此有了精确数字：taker 永远差 0.15–1 tick，出路只有 maker 报价与执行择时。**

---

## 五条大结论（教学重点）

1. **同期解释 ≠ 预测。** OFI 同期 R²≈0.5，滞后一期 R²≈0.001。市场把已观察到的流量瞬间
   定价掉。任何"高R²"先问：是不是同期回归？
2. **盘口确实预测未来——但只在秒级、且幅度在价差之内。** 四种完全不同的方法
   （方向概率、线性回归、马尔可夫链）给出同一幅图景：可预测成分在 0.5–10 秒内
   指数衰减，60 秒基本归零；期望幅度 0.1–0.5 tick < 1 tick 的过路成本。
   **信号真实、统计显著、逐月稳定，但对 taker 不构成独立 alpha**；
   它的钱在 maker 报价、撤单时机、执行择时里。
3. **价差状态是万能调节旋钮。** 窄价差（IF/IH 常年1 tick）→ 信号强；宽价差（IC/IM）→
   信号弱。论文2跨股票的 tick-size 结论在我们单一市场内部按价差状态完整复现；
   论文3靠"除以价差"把四个合约装进一个模型；论文4的 g* 按价差分层。
4. **这些信号是结构性的，不是 regime 巧合。** QI 命中率、VOI 月度 R² 六年多每个月
   都在线（对比我们 `ticker/` 的开盘反转——逐年翻号的 regime 伪信号）。机制
   （队列竞赛、做市商补货）根植于撮合规则，所以稳。
5. **四篇文献是同一个对象的四个视角。** OFI≈VOI（流量）、QI=OIR（水平）、MPB（成交）、
   g*（把前三者折算成公允价修正）。学界十年的进展 = 把"盘口失衡有信息"这句话
   从解释（2014）→ 方向预测（2016）→ 多因子回归（2015）→ 定价修正（2018）→
   深度学习（2019+, Sirignano-Cont, DeepLOB）逐步形式化。

## 文件地图

```
lob_common.py            共同取数/清洗（时段、去重、坏tick、gid分组）
ofi_ddb.py / ofi_full_ddb.py / ofi_*_plot.py     论文1（先前完成）
qi_ddb.py / qi_plot.py                           论文2
voi_ddb.py / voi_plot.py                         论文3
voi_backtest.py / voi_backtest_plot.py           论文3的阈值策略净费用回测 (bt_*.csv, 图84-85)
microprice_ddb.py / microprice_plot.py           论文4
*_pilot.csv / *_pilot.png                        2024-06/07 两月试跑（验证代码）
qi_bins/results/permonth.csv                     论文2输出
voi_results/coefs/permonth/hitrate.csv           论文3输出
mp_gstar/rmse/bias.csv                           论文4输出
fig71..93_*.png                                  全部图
STUDY_GUIDE_{OFI,QI,VOI,MICROPRICE}.md           四篇学习指南（从零教起）
```

复跑：`PILOT=1 ../.venv/bin/python qi_ddb.py`（两月试跑，~20s）；去掉 `PILOT=1` 跑全样本
（每个脚本 10–25 分钟；需关沙箱连 LAN）。图：`python3 qi_plot.py` 等（`SUF=_pilot` 画试跑版）。

## 下一步可做（按性价比）
- ~~论文3的阈值策略完整损益~~ ✅ 已做（图84-85）：taker 差 0.15–1 tick 才够本，结论定量化。
- 把 `P_micro` 代回 intraminute/ticker 的旧研究当"更准的尺子"（指南4 §6）。
- maker 版回测：挂单在己方最优价、用后续快照判断成交——检验"做市方向"能否翻正。
- 跨合约版：IF 的盘口预测 IC/IM（Cont 2023 cross-impact，方向D lead-lag 的盘口版）。
- HAR-RV 波动率线（RESEARCH_DIRECTIONS.md 方向E）——预测波动比预测方向容易且真能用。
