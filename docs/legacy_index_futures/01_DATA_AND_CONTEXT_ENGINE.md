# 文档 01 · 数据层与状态引擎（Layer 0–1）

> **本层的定位**：这是全系统唯一一个**目标可以估准**的层。它的目标是二阶矩（波动率）与可观测量（时段、效率、流动性），不是一阶矩（收益方向）。二阶矩的可预测性比一阶矩高两到三个数量级：
> - \(\sigma_t\) 的样本外 \(R^2\) 通常在 40%–70%
> - \(\mu_t\) 的样本外 \(R^2\) 通常在 0.1%–1%
>
> 因此本层的投入产出比远高于后续任何一层。**如果时间不够，宁可砍掉一半检测器，也不要在本层偷工。**下游所有形态检测都建立在 \(\tilde r = r/\hat\sigma\) 之上；如果 \(\hat\sigma\) 是错的，检测器检测到的将是"现在是几点"而不是"市场是什么形态"。

---

## 1. P0 数据取证清单（Stage 1 必做，逐项写入报告）

每一项给出：**做什么 → 期望看到什么 → 若不符如何处理**。

### C1 · Schema 与完整性
读取 parquet，断言：494,035 行、12 字段、无空值、5 个 root、23 个 local_symbol、dtype 与文档一致。检查 `(root, timestamp)` 是否唯一 —— **若有重复，说明同一时点存在多合约报价，与文档"单序列串接"矛盾，必须先解决。**

### C2 · 时区与会话映射（验证 F1）
把 `timestamp` 分别转成 `America/New_York`（ES/NQ/RTY）与 `Asia/Hong_Kong`（HSI/HTI），画**本地时间小时 × 有效 bar 数**的直方图，按月分面。

期望：
- ES/NQ/RTY：本地时间覆盖 18:00 → 次日 17:00，在 17:00–18:00 有维护窗口空洞；**用 UTC 画时会看到 3/8 前后整体移动 1 小时，用本地时间画则不会**。这就是 F1 的判据。
- HSI/HTI：本地时间覆盖 17:00/17:15 → 次日 03:00，**日盘 09:15–16:30 完全空白**。这就是 F2 的判据。

若 F2 不成立（存在日盘数据），立即停止，向用户报告，因为文档 02 的检测器清单需要重写。

### C3 · 会话定义变更检测（验证 F3）
对每个 root，按交易日计算「首根 bar 的本地时间」与「末根 bar 的本地时间」，画时间序列。

期望：HSI/HTI 的会话开始时间在样本中段（约 6 月底至 7 月底之间）从 17:15 跳到 17:00。**用变点检测（简单地：对首 bar 时间做中位数滤波后找跳变点）自动定位切换日期，写入 `configs/base.yaml` 的 `session_defs` 分段表。**

同时检查是否存在半日市（港股节假日前）、美股提前收市（感恩节、圣诞前）。这些日子的会话长度显著偏短，必须打标 `is_half_day` 并在季节性估计中排除。

### C4 · 填充 bar 识别（验证 F4/F5）
定义原始 1min bar 的有效性：

```
is_real_1m = (bar_count > 0) & (volume > 0)
is_flat    = (open == high) & (high == low) & (low == close)
is_filled  = (~is_real_1m) | (is_flat & (volume == 0))
```

统计 `is_filled` 的比例，按 `root × 本地小时` 交叉表输出热力图。

期望：ES 在美股 RTH 接近 0%，在亚洲时段 30%–60%；RTY 全时段高于 ES（因共用网格）；HSI/HTI 夜盘尾段（HKT 01:00–03:00）可能超过 50%。

**关键决策：填充 bar 不删除（保留时间轴完整），但打标；所有基于收益的计算必须跳过它们。**

### C5 · Roll 边界（验证 F6）
按 `root` 找 `local_symbol` 变化点，输出边界表：前合约末 bar 时间/价格、后合约首 bar 时间/价格、时间间隔、价格跳变（bp）。

期望：无重叠（间隔 > 0）；跳变幅度 ES/NQ/RTY 在 100–200 bp 量级（含周末隔夜），HSI/HTI 在 50–150 bp。

**决策（写入 `DECISIONS.md`）：不做后复权。**理由：无重叠报价 ⇒ 无法分离展期价差与跳空。做比例后复权会把一个未知量强行摊到全部历史价格上，扭曲所有比值型与分位型特征。替代方案：合约内计算收益，跨边界的收益与标签一律置 NaN。代价仅为 22 根 bar 的收益 + 22 × h 根 bar 的标签，可以忽略。

**注意**：这意味着**不能画一条连续的价格曲线**。若需要可视化，用「合约内累计收益拼接」的合成序列，并明确标注为"仅用于展示，不用于计算"。

### C6 · 价格与跳动网格
对每个 root，检查价格是否落在最小跳动网格上：`(price / tick) % 1` 应恒为 0。据此**从数据中反推 tick size**，不要硬编码。

期望：ES/NQ 0.25，RTY 0.1，HSI/HTI 1.0。

同时计算相对跳动 `tick / median(close)`，这是成本模型的基础常数：

| root | tick | 中位价 | 相对跳动 (bp) |
|---|---|---|---|
| ES | 0.25 | ~7100 | ~0.35 |
| NQ | 0.25 | ~27000 | ~0.09 |
| RTY | 0.1 | ~2750 | ~0.36 |
| HSI | 1 | ~25200 | ~0.40 |
| HTI | 1 | ~4750 | ~2.11 |

**HTI 的 2.11 bp 是本项目最重要的单一数字之一**：它意味着 HTI 每次进出至少损失 4.2 bp（假设各跨一个 tick），而 HTI 的 5 分钟波动约 6–10 bp。信噪比结构上不成立。

### C7 · 异常值扫描
计算 1min 对数收益（合约内、仅 valid bar），标记 \(|r| > 8 \times\) 滚动 MAD 的点。逐个人工核对（打印前后各 5 根 bar）。区分：真实跳跃（数据发布、突发新闻，应保留）vs 数据错误（单点尖刺后立即回到原位、OHLC 逻辑矛盾，应修正）。

同时断言 OHLC 逻辑：`low <= min(open, close) <= max(open, close) <= high`。

### C8 · 缺口结构
计算相邻 valid bar 的时间间隔分布。区分：
- 1 分钟（正常）
- 会话内小缺口（无成交，2–30 分钟）
- 会话间缺口（隔夜、周末）
- 异常长缺口（> 1 会话，可能是数据缺失日）

列出所有缺失交易日，与官方交易日历比对。**期望 ES 约 128–132 个交易日，HSI 约 122–126 个。**

### C9 · 成交量与笔数结构
画 `volume` 与 `bar_count` 的日内曲线（按本地时间分钟，跨日中位数）。这是后续时段分桶的直接依据。

同时计算**平均单笔规模** `avg_trade_size = volume / bar_count`（在 `bar_count > 0` 时）。这个量在数据发布、机构执行时段会系统性变化，是一个免费的流结构代理变量。

### C10 · 跨市场同步性
在 UTC 09:00–19:00 重叠窗口，计算 ES 与 HSI 的 5min 收益相关系数，以及 lead-lag 交叉相关函数

\[
\rho_k = \mathrm{Corr}\big(r^{ES}_{t-k},\, r^{HSI}_{t}\big),\quad k \in [-10, 10]
\]

期望：峰值在 \(k = 1\) 或 \(2\)（ES 领先 HSI 1–2 分钟），峰值相关约 0.3–0.5。**这个函数的形状直接决定文档 02 中 D06 检测器是否可行。**

### C11 · 有效点差估计
用 Abdi–Ranaldo (CHL) 高低价估计量。令 \(c_t = \log(\text{close}_t)\)，\(\eta_t = \frac{\log h_t + \log l_t}{2}\)，则

\[
\hat S = 2\sqrt{\max\Big(\mathbb{E}\big[(c_t - \eta_t)(c_t - \eta_{t+1})\big],\ 0\Big)}
\]

在 `root × 本地小时` 上分组估计（用 5min bar，只用 valid bar 对）。输出点差曲线 \(\hat S(\text{root}, \tau)\)，单位 bp。

期望：ES 美股 RTH 约 0.3–0.5 bp，亚洲时段 0.8–2 bp；HSI 夜盘 1.5–4 bp，且尾段显著放大；HTI 夜盘 4–12 bp。

**若 CHL 估计出的点差小于 1 个 tick，用 1 个 tick 作为下界。**这是这个项目里成本模型的地基，不要用固定常数敷衍。

### C12 · 有效样本量核算
基于以上，重新计算 \(N_{\text{eff}}\) 与 \(\mathrm{se}(\widehat{SR})\)，更新宪章 1.2 节的数字。计算方式：对 5 个 root 的 5min 收益（重叠时段）算相关矩阵 \(\mathbf{R}\)，取

\[
N_{\text{eff}} = \frac{\big(\sum_i \lambda_i\big)^2}{\sum_i \lambda_i^2},\quad \lambda_i = \text{eig}(\mathbf{R})
\]

**这个数字必须出现在最终报告的第一页。**

---

## 2. 会话日历模块（`calendar_.py`）

### 2.1 设计

会话日历不是可选的辅助工具，它是本系统的**主坐标系**。所有时间相关的特征、所有分组统计、所有回测撮合都以它为准。

核心对象：

```python
@dataclass
class SessionSpec:
    root: str
    tz: str                    # "America/New_York" | "Asia/Hong_Kong"
    valid_from: date           # 分段生效日（处理 F3 的会话变更）
    valid_to: date
    segments: list[tuple]      # [(start_local_time, end_local_time), ...]
    phases: list[PhaseSpec]    # 时段分桶
```

**必须从数据反推、而非硬编码**：C2/C3 的输出直接生成 `SessionSpec` 列表并落盘为 yaml，之后代码只读 yaml。这样会话定义的任何变更都有版本可查。

### 2.2 交易日（`session_id`）定义

- **ES/NQ/RTY**：以 CME 电子盘交易日为准。本地时间 18:00 开始，到次日 17:00 结束，`session_id` 取**结束日**（即 18:00 ET 周日的 bar 属于周一）。
- **HSI/HTI**：夜盘 17:00 开始跨到次日 03:00，按 HKEX 惯例夜盘属于**次一交易日的 T+1 时段**。但由于本数据集只有夜盘，为避免混淆，**直接以夜盘开始日作为 `session_id`**，并在文档中注明。关键是内部一致，不是遵循某个外部约定。

### 2.3 时段分桶（`session_phase`）

这是本项目**唯一一条样本量充足的状态轴**：其他状态轴切一刀就把样本切一刀，而时段每天重复，半年下来每个桶有 120+ 个观测，且这些观测在宏观环境上是分散的。**必须用足。**

**ES / NQ / RTY（本地 ET）**：

| phase | 本地时间 | 经济含义 |
|---|---|---|
| `GLOBEX_OPEN` | 18:00–20:00 | 电子盘开盘，亚洲早盘，流动性阶跃 |
| `ASIA` | 20:00–02:00 | 最薄流动性，趋势延续性弱，噪音占比高 |
| `EU_OPEN` | 02:00–04:00 | 欧洲开盘（LSE/Eurex），流动性二次阶跃 |
| `EU_MORN` | 04:00–08:00 | 欧洲主时段 |
| `US_PRE` | 08:00–09:30 | 美股盘前；**08:30 是 CPI/NFP/PPI/零售等宏观数据密集发布点** |
| `US_OPEN` | 09:30–10:30 | 现货开盘，隔夜信息重定价，日内波动峰值 |
| `US_MORN` | 10:30–12:00 | 主交易时段 |
| `US_LUNCH` | 12:00–14:00 | 流动性低谷，**均值回归性最强** |
| `US_PM` | 14:00–15:30 | 下午；FOMC 日 14:00 有事件 |
| `US_CLOSE` | 15:30–16:15 | MOC 不平衡、被动调仓、日末风险再平衡 |
| `POST` | 16:15–17:00 | 收盘后，流动性骤降 |

**HSI / HTI（本地 HKT，仅夜盘）**：

| phase | 本地时间 | 对应 UTC | 经济含义 |
|---|---|---|---|
| `HK_NIGHT_OPEN` | 17:00/17:15–18:00 | 09:00–10:00 | 夜盘开盘，消化港股日盘收盘后至今的信息 |
| `HK_EU` | 18:00–21:30 | 10:00–13:30 | 欧洲主时段，港股夜盘跟随欧洲风险偏好 |
| `HK_US_PRE` | 21:30–22:30 | 13:30–14:30 | **美股现货开盘**，夜盘波动次峰 |
| `HK_US_MORN` | 22:30–01:00 | 14:30–17:00 | 美股上午，联动最强 |
| `HK_LATE` | 01:00–03:00 | 17:00–19:00 | 夜盘尾段，流动性最薄，点差最宽 |

> **关键认识**：HSI/HTI 的夜盘不是"亚洲市场"，它是**亚洲资产在欧美时段的定价窗口**。它的驱动主要来自欧美风险偏好而非本地基本面。这一点会直接影响 D06（跨市场）检测器的设计，也解释了为什么它与 ES 的相关性远高于"港股 vs 美股"的日频相关性。

### 2.4 `bar_in_session` 与相位坐标

除了离散桶，还要给出连续相位坐标：

\[
\phi_t = \frac{\text{bar\_in\_session}(t)}{\text{session\_length}(\text{session\_id}(t))} \in [0, 1]
\]

用于季节性曲线的平滑估计（避免桶边界的不连续）。半日市的 session_length 不同，用 \(\phi\) 而非绝对序号可以自然对齐。

---

## 3. 清洗与重采样（`clean.py`）

### 3.1 工作频率的选择：5 分钟

**决策：以 5 分钟为主要工作频率，1 分钟仅保留用于（a）realized variance 计算、（b）微观结构特征（如单笔规模、点差估计）。**

理由：
- 1 分钟收益的微观结构噪音占比过高（bid-ask bounce 造成的负自相关），会污染所有形态识别
- 目标持仓 1–8 小时 = 12–96 个 5min bar，形态窗口 20–60 个 bar，在 5min 上是合理的分辨率
- 5min 重采样后 ES 约 27,200 bar，仍然充足
- **HSI/HTI 夜盘填充率高，5min 聚合能显著提高有效 bar 比例**

若后续发现某检测器需要更细粒度（例如 D01 的突破回收判定），允许在该检测器内部临时下钻到 1min，但输出仍对齐到 5min 网格。

### 3.2 重采样规则

```
open      = 首个 valid 1m bar 的 open
high      = max(high over valid 1m bars)
low       = min(low over valid 1m bars)
close     = 末个 valid 1m bar 的 close
volume    = sum(volume)
bar_count = sum(bar_count)
wap       = sum(wap * volume) / sum(volume)   # volume 为 0 时取 close
n_valid_1m= count(valid 1m bars)
is_valid  = n_valid_1m >= 2
```

**注意**：`open`/`close` 用 valid bar 的首尾，不是网格首尾。若整个 5min 窗口无 valid bar，则 `is_valid=False`，价格字段用前值填充（仅为保持时间轴连续），且该 bar 不参与任何计算。

### 3.3 收益构造（`returns.py`）

```
ret[t] = log(close[t]) - log(close[t-1])
```

强制置 NaN 的情形：
1. `is_valid[t] == False` 或 `is_valid[t-1] == False`
2. `local_symbol[t] != local_symbol[t-1]`（roll 边界）
3. `session_id[t] != session_id[t-1]`（跨会话；隔夜收益单独存为 `overnight_ret`，不混入日内序列）

> 隔夜收益不是垃圾，它是一个独立的、有价值的对象（尤其对 ES：美股指数的长期收益绝大部分来自隔夜时段，日内 RTH 收益的长期均值接近零）。但它的分布与日内收益完全不同，**必须分开建模**。本项目中它作为 D05 检测器的输入，不作为日内序列的一部分。

### 3.4 标签构造

对每个 horizon \(h \in \{6, 12, 24, 48, 96\}\)（对应 30min/1h/2h/4h/8h）：

\[
\text{fwd\_ret}_h(t) = \log \frac{\text{close}(t+h)}{\text{close}(t)},
\qquad
\text{fwd\_ret\_z}_h(t) = \frac{\text{fwd\_ret}_h(t)}{\hat\sigma_t \sqrt{h}}
\]

`label_valid_h(t) = False` 若 \([t, t+h]\) 窗口内存在 roll 边界，或跨越了超过 1 个会话缺口，或 valid bar 比例 < 60%。

**标准化标签 `fwd_ret_z` 是主要研究对象**，原始 `fwd_ret` 仅用于回测与成本核算。理由：标准化后跨品种、跨时段、跨波动状态可比，这是池化的前提。

**重叠标签警告**：相邻 bar 的标签窗口高度重叠，重叠度 \(1 - 1/h\)。这使得：
- 有效样本数约为 \(n/h\)，不是 \(n\)
- 普通 t 检验的 t 值会虚高约 \(\sqrt{h}\) 倍
- 必须用 block bootstrap（块长 \(\ge 5h\)）或 Newey-West（lag \(= h\)）修正

---

## 4. 波动率引擎（`vol.py`）—— 本层的核心

### 4.1 三层结构

\[
\hat\sigma_t \;=\; \underbrace{\hat\sigma_{\text{day}(t)}}_{\text{会话级水平}} \;\times\; \underbrace{\hat s(\phi_t)}_{\text{日内季节性}} \;\times\; \underbrace{\hat u_t}_{\text{局部偏离修正}}
\]

三个部分分别用不同方法、不同时间尺度估计，然后相乘。这个乘法分解是关键：它把"今天整体波动多大"（慢变，用长历史估）与"现在是一天中的什么位置"（周期，用跨日平均估）与"刚刚发生了什么"（快变，用短窗口估）解耦。

### 4.2 会话级波动：RV → 跳跃分离 → HAR

**第一步，realized variance。**在 1min 数据上按 5min 子采样计算，用平均子采样（averaged subsampling）降低微观结构噪音：

\[
RV^{(5)}_d = \frac{1}{5}\sum_{j=0}^{4} \sum_{i} \left(r^{(5\text{min},\,\text{offset}=j)}_{d,i}\right)^2
\]

即用 5 个不同起点的 5 分钟网格各算一次 RV，取平均。只计入 valid bar 构成的收益。

**第二步，跳跃分离。**Bipower variation：

\[
BV_d = \frac{\pi}{2}\cdot\frac{n}{n-1}\sum_{i=2}^{n}\big|r_{d,i}\big|\big|r_{d,i-1}\big|
\]

连续部分 \(C_d = \min(RV_d, BV_d)\)，跳跃部分 \(J_d = \max(RV_d - BV_d, 0)\)。

**为什么必须分离**：跳跃部分对未来波动几乎无持续性（一次性信息冲击），连续部分有极强持续性。混在一起会让 HAR 的系数被跳跃日拉偏，显著削弱预测能力。这是一个能带来 5–10 个百分点 \(R^2\) 的细节。

**第三步，HAR-J 回归。**

\[
\log C_{d+1} = \beta_0 + \beta_d \log C_d + \beta_w \log C^{(w)}_d + \beta_m \log C^{(m)}_d + \beta_j \log(1 + J_d) + \beta_l\, r_d^{-} + \varepsilon_{d+1}
\]

其中 \(C^{(w)}_d\) 为过去 5 个会话的均值，\(C^{(m)}_d\) 为过去 22 个会话的均值，\(r_d^- = \min(r_d, 0)\) 为杠杆项（股指的负收益显著放大未来波动，这一项在股指上通常贡献 3–6 个百分点 \(R^2\)）。

**估计方式：扩展窗口滚动 OLS**，最小起始窗口 40 个会话。禁止全样本一次性拟合（会造成软性前视）。

**跨品种池化**：先在 ES/NQ/RTY 上池化估一组系数（加入 root 固定效应），再允许每个 root 的系数向池化值收缩。半年只有约 130 个会话，单品种估 6 个系数已经吃力，池化是必需的。

**门禁 G2 要求扩展窗口样本外 \(R^2 \ge 0.40\)**。若达不到，优先排查：（a）是否混入了填充 bar；（b）是否跨 roll 计算了收益；（c）半日市是否未剔除；（d）HSI/HTI 是否用了错误的会话定义。

**HSI/HTI 的特殊处理**：夜盘会话长度约 585 分钟但 valid 率低，RV 估计噪音大。建议 HSI/HTI 的 HAR 单独估计，且加大平滑（用 \(w=5, m=22\) 之外再加一个 \(m=66\) 项），或直接用 EWMA 替代 HAR（若 HAR 样本外 \(R^2 < 0.30\)）。

### 4.3 日内季节性乘子 —— 最容易做错的一步

**目标**：估计 \(s(\phi)\) 使得 \(r_t / (\sigma_{\text{day}} \cdot s(\phi_t))\) 在日内各时点具有近似相同的尺度。

**方法（稳健版）**：

1. 用会话级波动把收益粗标准化：\(x_{d,i} = r_{d,i} / \hat\sigma_{\text{day}(d)}\)
2. 对每个相位桶 \(\phi\)（建议用 5min 网格上的绝对相位，即"会话内第 k 个 5min bar"，共约 78–280 个位置），取**跨日中位数绝对值**：
   \[
   m(\phi) = \underset{d}{\mathrm{median}}\,\big|x_{d,\phi}\big|
   \]
   用中位数而非均值：单个极端日（如 FOMC）会严重污染均值。
3. 对 \(m(\phi)\) 做平滑：沿相位方向用宽度 3–5 的中心移动中位数，再用局部线性回归（LOWESS，span ≈ 0.1）平滑。**在会话开盘/收盘的 U 型两端不要过度平滑**，那里的真实曲率很大。
4. 归一化：
   \[
   \hat s(\phi) = \frac{m(\phi)}{\sqrt{\frac{1}{K}\sum_{\phi'} m(\phi')^2}}
   \]
   使得 \(\overline{s^2} = 1\)，从而不改变整体波动水平。
5. **单独处理 phase 内的事件时点**：08:30 ET（数据发布）、09:30 ET（开盘）、14:00 ET（FOMC）、15:50–16:00 ET（MOC）在 ES 上会有尖峰。平滑不要把这些尖峰抹平 —— 它们是真实的、可预测的、每天重复的结构。

**关键约束：扩展窗口估计。** \(\hat s\) 在时刻 \(t\) 只能用 \(t\) 之前的会话估计，最小 20 个会话起步，之后每 5 个会话更新一次。

> **这是本项目最隐蔽的前视偏差来源。**季节性剥离看起来只是"技术性预处理"，但用全样本估计会把整个样本期的日内结构泄露到每一天。它的危害程度不亚于用未来收益训练模型。**必须写一个 pytest 断言来守住这一点。**

**验收（G2）**：画剥离前后的日内 \(\mathbb{E}|\tilde r|\) 曲线。剥离后应接近水平，量化指标：\((\max - \min)/\mathrm{mean} < 0.25\)。同时画剥离后 \(\tilde r\) 的分布，峰度应显著低于原始收益（典型：原始 15–40 → 剥离后 5–9）。

### 4.4 局部偏离修正

会话级 + 季节性已经捕捉了大部分，但会话内可能出现波动状态突变（如突发新闻）。加一个快速修正项：

\[
\hat u_t = \left(\frac{\text{EWMA}_{\lambda}\big(\tilde r^2\big)_{t-1}}{1}\right)^{\theta/2},\qquad \lambda \approx 0.97\ (\text{半衰期} \approx 23\ \text{bar}),\ \theta \in [0.3, 0.5]
\]

其中 \(\tilde r = r/(\sigma_{\text{day}} s(\phi))\) 是二次标准化前的残差。\(\theta < 1\) 是刻意的收缩：完全跟随（\(\theta = 1\)）会引入过多噪音。取 \(\theta = 0.4\) 作为默认，并做邻域稳定性检查。

同时对 \(\hat u_t\) 做 clip 到 \([0.5, 2.5]\)，防止极端值。

### 4.5 最终输出与验收

```
sigma_hat[t] = sigma_day[session(t)] * seas[phase(t)] * u[t]      # 5min 收益的条件标准差
```

**无前视断言**（必须写成 pytest）：
```python
def test_sigma_no_lookahead():
    # 用前 60% 数据计算 sigma_hat，与用全量数据计算的前 60% 部分逐点比较
    # 允许的差异只能来自 HAR 的扩展窗口重估，且必须在 t 之前
    assert (sigma_partial == sigma_full[:n60]).all()
```

**诊断图（必须存档）**：
1. 剥离前后日内 \(\mathbb{E}|r|\) 曲线（每个 root 一张）
2. \(\hat\sigma\) 与实际 \(|r|\) 的滚动散点 + 分位校准图（预测 20% 分位的 bar 中，实际是否约 20% 落在该分位内）
3. HAR 样本外 \(R^2\) 的滚动曲线
4. \(\tilde r\) 的 QQ 图与自相关图（自相关应接近 0；若在 lag 1 有显著负值，说明 bid-ask bounce 未被 5min 聚合消除，考虑用 wap 而非 close）

---

## 5. Context 变量（`context.py`）

设计原则（宪章第 6 条）：**低维、可观测、经济含义明确、软调制而非硬切换。**明确放弃 HMM —— 高斯似然对方差的敏感度远高于对均值，HMM 在收益序列上几乎总是收敛到"高波/低波"两态，那是一个 EWMA 就能给出的、更及时更稳定的东西；而 EM 输出的平滑概率含未来信息，滤波概率又需要滚动重估整套参数，成本与收益不成比例。

### 5.1 变量清单

**（1）波动水平 `sigma_pct`**
\(\hat\sigma_{\text{day}}\) 在过去 60 个会话中的经验分位，\(\in [0,1]\)。经济含义：风险偏好、杠杆容量、期权对冲环境。

**（2）波动方向 `vol_ratio`**
\[
\text{vol\_ratio}_t = \frac{\text{EWMA}_{\text{hl}=12}(\tilde r^2)}{\text{EWMA}_{\text{hl}=96}(\tilde r^2)}
\]
> **水平与方向必须分开。**高波但在收缩（恐慌后的平复）与低波但在扩张（酝酿突破）是两种完全不同的环境，混为一谈会互相抵消。这是很多人做 vol regime 时的漏项。

**（3）效率比 `er_n`**
\[
ER_n(t) = \frac{\big|P_t - P_{t-n}\big|}{\sum_{i=0}^{n-1}\big|P_{t-i} - P_{t-i-1}\big|} \in [0,1]
\]
取 \(n = 20\) 与 \(n = 60\)（5min bar，即 100 分钟与 300 分钟）。经济含义：趋势 vs 震荡的直接度量。

> **为什么用 ER 而不是 Hurst 指数**：Hurst 在短窗口上估计方差巨大，对预处理（去趋势方式、聚合方式）极度敏感，且不同估计量（R/S、DFA、小波）在同一段数据上能给出差异 0.15 以上的结果。ER 是同样直觉的一个稳健、无参、有界的实现，且计算成本可忽略。

**（4）方差比 `vr_q`**
\[
VR(q) = \frac{\mathrm{Var}\big(r^{(q)}\big)}{q\cdot\mathrm{Var}\big(r^{(1)}\big)}
\]
取 \(q = 5\)，滚动窗口 120 bar。\(VR < 1\) 表示均值回归，\(> 1\) 表示趋势。与 ER 互补：ER 看路径几何，VR 看二阶矩标度。两者背离时（ER 高但 VR 低）通常意味着单向漂移叠加高频抖动 —— 典型的算法执行痕迹。

**（5）流动性 `liq_z` / `spread_bp` / `avg_trade_size`**
- `liq_z`：`log(volume)` 相对同 phase 历史中位数的 z 分数
- `spread_bp`：C11 估计的点差曲线 + 当前的短窗修正
- `avg_trade_size`：`volume / bar_count`，相对同 phase 历史的比值

**（6）跨市场 `es_ret_z`（HSI/HTI 专用）**
ES 在过去 \(k\) 个 5min bar 的累计标准化收益。由 C10 确定最优 \(k\)。

**（7）会话相位 `session_phase` / `phi`**
如 2.3、2.4 节。

**（8）日历标记**
`is_roll_week`（合约到期周）、`is_month_end`（月末 2 日）、`is_quarter_end`、`is_half_day`。这些是免费的、日历可知的信息，对应展期流、再平衡流。

### 5.2 状态调制函数 `gate_score`

**这是 regime 进入系统的唯一通道。**

设计：把若干 context 变量映射到一个标量 \(g_t \in [0, 1.5]\)，乘在基础仓位上。

\[
z_t = \sum_{j} \alpha_j \cdot \psi_j(c_{j,t}), \qquad g_t = g_{\min} + (g_{\max}-g_{\min})\cdot\varsigma(z_t)
\]

其中 \(\psi_j\) 是把每个 context 变量映射到 \([-1, 1]\) 的单调函数（推荐：经验分位映射后线性拉伸，天然稳健、无参数），\(\varsigma\) 是 logistic 函数，\(\alpha_j\) 是权重。

**参数约束（宪章第 5 条）**：
- \(\alpha_j\) 的**符号必须先验固定**（从 `HYPOTHESES.md` 推导），只允许幅度自由
- \(\alpha_j\) 跨品种共享
- 初版直接用等权（\(\alpha_j = 1/J\)），只有当等权版本通过 G3/G4 后，才允许尝试加权，且加权尝试计入试验计数 \(K\)
- \(g_{\min} = 0.2\)（不完全关闭，保留信息）、\(g_{\max} = 1.5\)

**不同检测器可以使用不同的 gate 组合**，但每个 gate 的构成必须在 `HYPOTHESES.md` 中先验声明。例如：
- 反转型检测器（D01, D07）：\(\psi(\text{低 ER}) + \psi(\text{低 sigma\_pct}) + \psi(\text{vol 收缩}) + \psi(\text{正常流动性})\)
- 趋势型检测器（D02, D03, D04）：\(\psi(\text{高 ER}) + \psi(\text{vol 扩张}) + \psi(\text{高 liq\_z})\)

**G4 验收**：画「条件收益 vs gate_score 分位」的曲线。要求单调、平滑。用 isotonic regression 拟合，\(R^2 \ge 0.6\)，且相邻分位间跳变 < 总幅度的 40%。**锯齿状曲线 = 拟合的是噪音，必须退回无条件版本。**

---

## 6. 成本模型（`spread.py`）

### 6.1 结构

单次往返成本（收益率单位）：

\[
c_t = \underbrace{2\times\Big(\tfrac{1}{2}\hat S(\text{root}, \phi_t) + \kappa_{\text{slip}}\cdot \text{tick}_{\text{bp}}\Big)}_{\text{价差与滑点}} + \underbrace{2\times \text{comm}_{\text{bp}}}_{\text{佣金}}
\]

再乘以宪章规定的保守系数 1.5。

参数：
- \(\hat S(\text{root}, \phi)\)：C11 的时段相关点差曲线，下界为 1 tick
- \(\kappa_{\text{slip}}\)：额外滑点系数，默认 0.5（即半个 tick），在 `liq_z` 低的时段按 \(\kappa_{\text{slip}} \cdot (1 + \max(0, -\text{liq\_z}))\) 放大
- \(\text{comm}_{\text{bp}}\)：ES/NQ/RTY 约 0.03–0.10 bp（按名义），HSI/HTI 约 0.3–0.6 bp

### 6.2 参考量级（Agent 须用 C11 实测校准）

| root | 相对 tick (bp) | 典型点差 (tick) | 往返成本 (bp) | 相对 ES |
|---|---|---|---|---|
| ES | 0.35 | 1 | 0.7–1.0 | 1.0× |
| NQ | 0.09 | 1–2 | 0.4–0.7 | 0.7× |
| RTY | 0.36 | 1–2 | 1.1–1.8 | 1.7× |
| HSI（夜盘） | 0.40 | 2–5 | 2.0–4.5 | 3.5× |
| HTI（夜盘） | 2.11 | 1–3 | 5.5–13 | 12× |

### 6.3 由成本推导目标持仓期

净夏普关于持仓期 \(h\)（以日计）的表达式：

\[
SR_{\text{net}}(h) = \underbrace{\rho(h)\sqrt{\frac{T}{h}}}_{\text{毛夏普}} \;-\; \underbrace{\frac{c\sqrt{T}}{\sigma\, h}}_{\text{成本拖累}}
\]

成本拖累与 \(h\) 成反比，毛夏普只以 \(h^{-1/2}\) 增长（若 \(\rho\) 不变）。因此必然存在内点最优，且

\[
h^{\*} \propto \left(\frac{c}{\rho\,\sigma}\right)^{2}
\]

**对成本是二次敏感的。**由上表：若 ES 最优持仓 2 小时，则

| root | \(c/c_{ES}\) | \(\sigma/\sigma_{ES}\) | \(h^*/h^*_{ES}\) | 建议持仓期 |
|---|---|---|---|---|
| ES | 1.0 | 1.0 | 1.0 | ~2h |
| NQ | 0.7 | 1.4 | 0.25 | ~1h（但受最小分辨率约束，取 1–2h） |
| RTY | 1.7 | 1.4 | 1.5 | ~3h |
| HSI | 3.5 | 1.2 | 8.5 | ~16h（跨越整个夜盘） |
| HTI | 12 | 1.8 | 44 | **不可交易** |

> **这张表是 Tier 分层的数学依据。**实现上不要给五个品种写五套逻辑 —— 信号完全共享，只让**无交易带宽度**按品种自适应（文档 03 第 4 节），持仓期就会自动分化。这是最优雅的实现，也最不容易过拟合。

---

## 7. 本层交付验收清单

- [ ] `reports/01_data_forensics.md`：C1–C12 逐项结论 + 图
- [ ] `configs/session_defs.yaml`：从数据反推的会话定义（含分段）
- [ ] `configs/cost_model.yaml`：从数据估计的点差曲线与成本参数
- [ ] `interim/bars_5m.parquet`：符合契约的 5min bar
- [ ] `processed/context.parquet`：全部 context 变量
- [ ] `processed/labels.parquet`：多 horizon 标签
- [ ] `reports/02_context_engine.md`：波动率引擎诊断 + 4 张必备诊断图
- [ ] `tests/`：无前视断言（sigma、seasonality、HAR 扩展窗口）全绿
- [ ] `STATE.md` / `RESEARCH_LOG.md` 更新，G1/G2 门禁结论明确
