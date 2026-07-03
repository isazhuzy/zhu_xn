# 研究方向路线图（microstructure / 高频）

> 背景：`future/` + `experiment/` 的"脉冲/放量后反转"结论 = **均值上有微弱结构、频率上≈抛硬币、幅度在价差内 → 可验证、不可盈利**。
> 那条线只用了 `mid + volume`。DolphinDB `hft_future_ts` 表里还有**大量没用的微结构字段**，下面是基于这些字段的全新方向。

## 数据里可用的关键字段（之前没碰）
| 字段 | 含义 | 状态 |
|---|---|---|
| `m_nBidPrice/Volume`, `m_nAskPrice/Volume` | 盘口一档价+量 | ✅ 全满 |
| **`m_nActBidVolume` / `m_nActAskVolume`** | **主动买/主动卖量（成交方向）** | ✅（每 tick 只一边非零） |
| `m_iTurnover` | 成交额 → VWAP/均价 | ✅ |
| `m_nMatchItems` | 成交笔数（**负值，需查证**） | ⚠️ |
| `m_nBidOrder/AskOrder`, `m_nABOrderRate` | 委托档口数/比率 | ✅ |
| `m_nMItemsVolRate` | — | ❌ 全 0 |

**核心转变**：之前研究的是"价格动了之后会怎样"（**滞后/结果变量**）；订单流字段让你研究"**什么在驱动价格**"（**领先/因变量**）——微结构 alpha 的真正所在。

---

## 方向（按 数据齐全度 × 文献证据 × 与已做不重叠 排序）

### 🥇 A · 订单流失衡 OFI → 预测下一步价格  【已启动 → `orderflow/`】
- 用 `ActBidVol−ActAskVol`（交易流）+ 盘口量价变化（CKS-OFI）。
- 文献证明：价格变动 ≈ β·OFI，近似**线性、真有预测力**（对比你的反转~抛硬币）。
- **必读**：Cont, Kukanov & Stoikov (2014) *The Price Impact of Order Book Events*；Cont, Cucuringu & Zhang (2023) *Cross-Impact of Order Flow Imbalance*。

### 🥈 B · 盘口/队列失衡 → 短期方向
- `(BidVol−AskVol)/(BidVol+AskVol)`，对下一个价格变动方向有强预测。
- **必读**：Lipton, Pesavento & Sotiropoulos (2013) *Trade Arrival Dynamics and Quote Imbalance*；Gould & Bonart (2016)。

### 🥉 C · 价格冲击 / 流动性（Kyle's λ, 平方根律）
- 签名成交量 + turnover + 价格变动 → 一笔单子推动价格多少（解释了"边际全在价差内"= 冲击就是成本）。
- **必读**：Kyle (1985)；Tóth et al. (2011) *Anomalous Price Impact and the Critical Nature of Liquidity*；Bouchaud price-impact 综述。

### D · 跨合约领先滞后 (lead-lag) — IC/IF/IH/IM 谁领先谁
- 异步 tick → Hayashi–Yoshida 估计量。统计套利/配对入口。

### E · 已实现波动率预测 (HAR-RV) + 跳跃检测
- 预测波动比预测方向容易、且真有用。**必读**：Corsi (2009) HAR；Andersen-Bollerslev-Diebold；Aït-Sahalia–Jacod 跳跃。

### F · 深度学习 LOB（全特征 → 短期方向）
- **必读**：Sirignano & Cont (2019) *Universal Features of Price Formation*（发现普适的类-OFI 特征）；Zhang-Zohren-Roberts (2019) *DeepLOB*。

### （旧线索）regime 开关
- "大脉冲反转 vs 趋势"逐年翻号；用波动率/价差状态做事前 regime 识别（HMM/change-point），把缺点变信号。

---

## 学习路径（书 + 综述，入门顺序）
1. Cont (2001) *Empirical Properties of Asset Returns: Stylized Facts* — 先建直觉。
2. **Bouchaud, Bonart, Donier, Gould (2018) 《Trades, Quotes and Prices》** — 现代圣经，强烈推荐。
3. Hasbrouck 《Empirical Market Microstructure》（实证）；O'Hara 《Market Microstructure Theory》（理论）。
4. Kyle (1985)、Glosten-Milgrom (1985) — 价差/逆向选择根源。
5. 前沿/ML：Sirignano-Cont (2019)、DeepLOB (2019)、Cont-Kukanov-Stoikov (2014)。

---
*建议起点：A（OFI）—— 数据齐、文献证有效、可复用已有累加器/分年度/跨合约框架，且首次能看到"显著且可能净>0"的样子。*
