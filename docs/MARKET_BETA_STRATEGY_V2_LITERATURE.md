# 策略机制论文阅读：从风险标签到条件机会

阅读日：2026-10-04。以下使用作者、机构或论文原文，记录“读到什么—如何转化为本项目—迁移边界”。这是研究假说来源，不是对本地回测的替代证明。已读取原文的条目注明章节；仅读到摘要的条目单列，避免假装完成全文精读。

## 1. 时间序列趋势为什么有方向价值

Moskowitz、Ooi、Pedersen（2012），[Time Series Momentum](https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum)，作者版本 [PDF](https://docs.lhpedersen.com/TimeSeriesMomentum.pdf)。读取机构研究页；本次 PDF 获取失败，未把全文标为已读。该研究把自己的过去收益作为信号，与横截面赢家/输家排序分开，发现不同期货上的时间序列动量。这支持“每个合约自己择时”的研究结构，不支持直接照搬其月度期限。

本项目推论：24/72/168/336h 是不同趋势速度，应比较连续规模和持仓期限，不能把过去收益正负直接等同于未来状态。缓慢形成的行情允许慢趋势；下跌转折快时，快趋势可能更及时，也更容易被反复打脸。

Dao、Nguyen、Deremble、Lempérière、Bouchaud、Potters（2016），[Tail protection for long investors: Trend convexity at work](https://arxiv.org/html/1607.02410)。精读 §2.1–2.2 的方差尺度、自协方差与趋势 PnL 推导，及结论。简化线性趋势的累计 PnL 是长尺度价格变化平方与短尺度平方变化和的差；趋势保护是否出现取决于观测期限。

本项目推导：令小时对数收益为 r，线性趋势 s_t=Σ_{j=1}^h a_j r_{t-j}。则其预期价格收益中，除平均漂移外，有 Σ_j a_j Cov(r_{t-j},r_t)。正自相关支持延续，负自相关使同样的仓位变化受损。波动变大本身不能决定该项正负。设计状态必须包含持续性、回撤/恢复、方向变化和共同市场一致性。

## 2. 空头为什么容易在正确的熊市里亏钱

Daniel、Moskowitz（2016），[Momentum crashes](https://spinup-000d1a-wp-offload-media.s3.amazonaws.com/faculty/wp-content/uploads/sites/3/2019/09/Mom_crashes_JFE_final.pdf)。精读恐慌状态结果与 §4 动态权重。论文的横截面动量在市场跌后、高波动、急反弹时易出现崩盘；条件收益与方差共同决定策略规模。

本项目推论：本币趋势空头不是该文的 WML 因子，但也会面临“慢信号仍看空、短期价格已经快速反弹”的错位。先区分下跌持续、加速下跌、恐慌反弹；反弹状态可退出短仓，也可用独立快反弹专家做多。多空均衡应先体现为相同名义上限、相同信号待遇和贡献透明，而非强制每条腿赚钱一半。

## 3. 状态的价值在动作切换，不在标签数量

Ang、Bekaert（2002），[How do Regimes Affect Asset Allocation?](https://business.columbia.edu/sites/default/files-efs/pubfiles/1959/inquire.pdf)。读取摘要、状态转换模型、条件配置与 ex-ante/smoothed 概率讨论。状态概率、收益与协方差可以共同改变配置；前向概率和全历史平滑概率的用途不同。

本项目推论：市场状态有用，意味着同样的趋势分数在“延续、区间、下跌、反弹”里应调用不同专家。训练历史可以估计发射分布及转移，但测试阶段只递推滤波。当前实现是高斯混合发射加经验 Markov 转移，不宣称复现该论文或已拟合完整 HMM 联合似然。

## 4. 从预测收益，走到直接学习动作

Lim、Zohren、Roberts（2019），[Enhancing Time Series Momentum Strategies Using Deep Neural Networks](https://arxiv.org/html/1904.04912)。精读 §III-B 的直接仓位目标、§IV 时序模型、§VI 换手正则。论文把学习仓位与学习方向分开，直接优化策略表现并考虑换手。

本项目推论：先预测均值再套任意阈值可能丢失风险、幅度与动作路径。故加入小型因果卷积网络，用前向 4h 收益和仓位变化惩罚训练连续目标。它是本项目的效用近似，不是原论文 Sharpe-LSTM 的复制；实际回测仍按同一慢调仓规则比较。神经网络损失改善不能代替跨期策略改善。

## 5. 非平稳关系：记忆长度也是策略选择

Zhang、Lu、Zhou（2018），[Adaptive Online Learning in Dynamic Environments](https://arxiv.org/html/1810.10815)。精读动态 regret、专家跟踪及权重递推。动态环境中可比较随时间变化的参照策略，多学习速度专家能够适应不同变化程度；理论依赖相应损失与有界假设。

本项目推论：旧关系不是永久真理。在线专家分别采用 14/42/126 日半衰期，收益完全成熟后才更新；状态条件均值强收缩并保留经济先验。滚动监督采用近 365 日与 126 日衰减，对照扩展训练。当前算法借鉴专家加权，没有实现 Ader 的全部 OGD/regret 保证，更没有金融盈利保证。遗忘太快会追逐近期噪声，遗忘太慢会延续失效关系。

## 6. 加密市场风险与本币策略的边界

Liu、Tsyvinski，[Risks and Returns of Cryptocurrency](https://www.nber.org/papers/w24877)；Liu、Tsyvinski、Wu，[Common Risk Factors in Cryptocurrency](https://www.nber.org/papers/w25882)。本次取得 NBER 摘要与出版信息，PDF 访问受限。前者报告加密资产的时序动量线索；后者讨论市场、规模和动量等共同风险。

本项目推论：12 币独立择时的 PnL 可以来自共同市场和本币剩余项。用事前估计 β_{i,t} 分解，不能把剩余项自动叫 alpha。12 个币同涨同跌时，分散币名并没有分散共同风险；主组合保持清晰的资金权重，逐币与组合结果都报告。

## 择时协方差的完整含义

给定同一期的事前实际暴露 B_t 与随后市场收益 R_{m,t+1}，平均共同市场 PnL 为：

```text
mean(B_t R_m,t+1) = mean(B_t) mean(R_m,t+1) + Cov_population(B_t, R_m,t+1)
```

第一项是平均风险敞口乘市场漂移；第二项回答“是否在市场随后好时暴露更高、随后坏时更低或变负”。市场平均下跌时，净空也可以只赚第一项；趋势策略可能平均净暴露接近零、仍赚第二项。收益要再减费用/资金，并加入本币剩余项。

这与 Cov(B_t, past R_m,t) 不同；后者可能只是跟涨杀跌，前者需要严格向前收益。也与资产协方差 Cov(R_i,R_m) 不同；后者决定回归 beta 与共同风险规模。协方差用各期算术收益有精确恒等式，不能把加总值冒充复利收益。策略的实测 B_t 包含无交易时的自然漂移，可能产生机械项；应同时看目标信号的前向协方差及实际账本项，解释差别。
