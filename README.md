# 加密货币合约多维形态择时研究基地

**方案 A 独立分支实施（2026-10-04）：**[精美离线 HTML 报告](reports/crypto/market_beta_a_v1/MARKET_BETA_A_REPORT.html) · [详细研究报告](reports/crypto/market_beta_a_v1/MARKET_BETA_A_REPORT.md) · [执行宪章](docs/MARKET_BETA_A_IMPLEMENTATION_2026_10_04.md)。`codex/market-beta-a` 实现连续市场信号、慢速多空调仓、官方资金费率核验、真实数量账本与递进模型反证；原形态原型路线及后续其他方法保留。新结果为开发前向/复用探索，没有独立可交易认证。最新进度以 [STATE](STATE.md) 为准；以下为此前研究脉络。

**2026-10-04 转型研究入口：**[从形态原型研究转向可检验的市场 Beta 状态策略](docs/STRATEGY_TRANSFORMATION_2026_10_04.md)。该文重新定义策略目标、主线与卫星路线、对照和停止规则；属于研究设计，尚未产生新的回测结果。以下段落记录此前实验，不代表转型后的策略已经验证。

本项目已有研究围绕 **12 个 USDT 合约的多维历史状态、未来条件分布和探索性多空信号** 展开。此前路线是：可解释量价与中间路径特征 → 方向中性的历史几何身份 → 受限的局部/连续条件分布 → 分开评价历史支持、预测价值与身份稳定性 → 独立信号阶段的市场/币种相对收益。**预测状态 v2 与首轮信号链均已完成**：48 区对未来波动有粗分层，交易方向增量失败；完整本币 X2 的稀疏策略有后验线索，但 2026 已多次使用、阈值和输入敏感，尚无认证的可交易优势。新主研究问题转为市场层条件 Beta 承担，具体对照与停止规则见转型文档。

最新实施研究（2026-09-30）：[完整 X2、状态触发与条件预测原型报告](reports/crypto/REGIME_PROTOTYPE_RESEARCH_2026_09_30.md)；初始[研究诊断](docs/RESEARCH_REVIEW_2026_09_30.md)和[优化指南](docs/NEXT_RESEARCH_OPTIMIZATION_GUIDE.md)已登记完成度。两段约 +12% 的 80/40 后验强候选为**本币 X2、不追加市场摘要**；本轮审计发现利润集中于少数日期。动态低波门槛增加参与却减损净收益，低波下跌六个慢尺度变量前向误差恶化，风险原型未证明方向增量；慢复核降低换手，但全期正收益仅去 2024-12-03 一日即转负。全部仍是开发与已复用探索证据。

## 从这里开始

1. [AGENTS.md](AGENTS.md)：智能体执行约定与下一步。
2. [00_AGENT_CHARTER.md](00_AGENT_CHARTER.md)：目标、研究原则、阶段成果。
3. [01_DATA_AND_CONTEXT_ENGINE.md](01_DATA_AND_CONTEXT_ENGINE.md)：数据契约和候选特征字典。
4. [02_DETECTORS_AND_VALIDATION.md](02_DETECTORS_AND_VALIDATION.md)：多维状态识别与分离度检验。
5. [03_PORTFOLIO_BACKTEST_AND_REPORTING.md](03_PORTFOLIO_BACKTEST_AND_REPORTING.md)：策略族、执行、成本、稳健性。
6. [STATE.md](STATE.md)：当前进度；[HYPOTHESES.md](HYPOTHESES.md)、[DECISIONS.md](DECISIONS.md)、[RESEARCH_LOG.md](RESEARCH_LOG.md) 分别记录假设、决策、实验。

历史强基线见[首轮信号链报告](reports/crypto/signal_chain_v1/07_signal_chain_v1.md)、[设计与后验迭代账](docs/SIGNAL_CHAIN_V1_PLAN.md)及[新增变量登记](docs/SIGNAL_CHAIN_V1_FEATURE_REGISTRY.md)。前一轮[预测状态 v2 报告](reports/crypto/predictive_states_v2/06_predictive_states_v2.md)及[设计](docs/PREDICTIVE_STATES_V2_PLAN.md)是比较起点；[方向中性预测状态报告](reports/crypto/predictive_states_v1/05_predictive_states_stage.md)及[设计](docs/NEXT_PREDICTIVE_STATES_STAGE.md)说明更早路线。历史人工种子见[中期任务书](docs/MIDTERM_PATTERN_STAGE.md)和[136 维原型综合报告](reports/crypto/04_full_pattern_stage.md)。旧项目思想的继承见[研究桥接说明](docs/RESEARCH_BRIDGE.md)。
中期报告的公式、研究取舍与代码实现另见[136 维形态原型研究讲解](docs/04_full_pattern_研究设计与数理实现讲解.md)。

## 数据现状

GitHub 版本保留研究代码、文档、汇总 CSV、图表和小型模型文件；约 1 GB 的本地 parquet 行情、特征及逐行预测不上传。克隆仓库后需按 `configs/` 中的配置指定自己的只读行情路径，再生成派生数据；报告中的汇总值是已完成实验的历史记录，不会因缺少 parquet 而自动重算。版本范围见 [CHANGELOG.md](CHANGELOG.md)。

只读源：`D:\Trading\practical_crypto_strategy\data\parquet\{symbol}\{5m,15m,1h}.parquet`。标的为 ADA、AVAX、BCH、BNB、BTC、DOGE、ETH、LINK、LTC、SOL、TRX、XRP（均为 USDT）。经 [只读盘点脚本](scripts/inspect_crypto_parquet.py) 检查，36 个文件共同覆盖 **2023-01-01 00:00 至 2026-09-24 23:55 UTC**（较粗频率末条各自为 23:45、23:00）。每标的 5m/15m/1h 分别有 392,544 / 130,848 / 32,712 行；总计 6,673,248 行。开盘时间严格连续、无重复，OHLC 基本约束通过。完整逐文件结果见 [inventory](reports/crypto_data_inventory.json)。

字段含 OHLC、基础与计价成交量、成交笔数、主动买入量及预计算衍生列。少量零成交 bar 对应 `vwap`/`taker_buy_ratio` 空值；每文件首条 `log_return` 为空。易读概况见 [数据报告](reports/crypto_data_profile.md)。用户已说明数据来源为**币安**；文件本身仍未证明具体合约类型与费率。数据**没有**资金费率、持仓量、盘口、真实流通量；代理成交活跃度不称作真实换手率。

## 仓库边界

- `src/crypto/` 是加密研究实现；`scripts/build_crypto_features.py` 提供双频因果字段。`scripts/crypto_full_pattern_stage.py`、`scripts/crypto_prototype_research.py` 分别属于旧 136 维与 v0 原型阶段，`scripts/crypto_event_research.py` 是更早的直接预测研究。其余旧 `src/`、`configs/base.yaml` 多数仍是**股指期货流程**，不能直接作为加密运行入口。
- 当前预测状态 v2 的审计、几何、连续概率、路径、校准与推断复现入口均为 `scripts/crypto_predictive_v2_*.py`；CUDA 距离与推断见 `src/crypto/predictive_states_v2.py`，冻结模型及图在 `reports/crypto/predictive_states_v2/`。v1 历史入口为 `scripts/crypto_predictive_states_stage.py`。2026-02—09 已被两轮读取，不能再作为新方法未触碰终段。
- 首轮信号链入口为 `scripts/crypto_signal_chain_v1.py`，仓位与漂移成本账本见 `src/crypto/signal_chain.py`，独立结果检查见 `scripts/crypto_signal_chain_v1_verify.py`。完整家族、逐币预测与持仓、买入持有、市场阶段与原型状态图在 `reports/crypto/signal_chain_v1/`。**本轮 2026 亦为已复用的探索性结果**，不可把后验最好的稀疏规则称独立测试。
- 最新基线审计、状态预测、固定预测仓位消融与低波下跌慢尺度阴性对照分别由 `scripts/crypto_baseline_mechanism_audit.py`、`scripts/crypto_regime_prototype_stage.py`、`scripts/crypto_regime_policy_iteration.py`、`scripts/crypto_calm_down_slow_stage.py` 复现；产物位于 `reports/crypto/baseline_mechanism_audit/`、`reports/crypto/regime_prototypes_v1/`、`reports/crypto/regime_policy_iteration/`、`reports/crypto/calm_down_slow_stage/`，完整解释见最新实施报告。2026 在本阶段继续复用，所有新净值均为探索性。
- `docs/legacy_index_futures/` 保存迁移前的宪章、阶段文档和研究账本；`reports/`、`data/`、`raw_data/` 中旧结果暂留原位，均不得当作加密策略证据。
- 新研究先完成数据与执行契约，再完成特征、状态、策略与验证。每阶段留下可复查的产物，可以根据发现调整顺序；没有旧式 G1–G5 数值门禁。

## 当前可运行的检查

```powershell
python scripts/crypto_market_beta_preflight.py
python scripts/crypto_market_beta_contract_api.py
python scripts/crypto_market_beta_a.py
python scripts/crypto_market_beta_verify.py
python scripts/crypto_market_beta_diagnostics.py
python scripts/crypto_market_beta_shadow.py
python scripts/crypto_market_beta_report.py
python scripts/inspect_crypto_parquet.py > reports/crypto_data_inventory.json
python scripts/crypto_state_pilot.py
python scripts/build_crypto_features.py
python scripts/audit_crypto_features.py
python scripts/crypto_event_research.py
python scripts/summarize_crypto_event.py
python scripts/crypto_prototype_research.py
python scripts/crypto_full_pattern_stage.py
python scripts/crypto_full_pattern_diagnostics.py
python scripts/summarize_crypto_full_stage.py
python scripts/crypto_predictive_states_stage.py
python scripts/crypto_predictive_states_registry.py
python scripts/crypto_predictive_states_diagnostics.py
python scripts/crypto_predictive_graph_stage.py
python scripts/crypto_predictive_v2_audit.py
python scripts/crypto_predictive_v2_geometry.py
python scripts/crypto_predictive_v2_conditional.py
python scripts/crypto_predictive_v2_resolution.py
python scripts/crypto_predictive_v2_hybrid.py
python scripts/crypto_predictive_v2_path_levels.py
python scripts/crypto_predictive_v2_middle_path.py
python scripts/crypto_predictive_v2_probability_diagnostics.py
python scripts/crypto_predictive_v2_summary.py
python scripts/crypto_predictive_v2_medoids.py
python scripts/crypto_predictive_v2_inference_check.py
python scripts/crypto_baseline_mechanism_audit.py
python scripts/crypto_regime_prototype_stage.py
python scripts/crypto_regime_policy_iteration.py
python scripts/crypto_calm_down_slow_stage.py
python -m pytest -q
```

新预测状态脚本请在 `conda activate universal` 后运行，实测使用 CUDA 12.8。`pytest` 包括旧工程与新的特征/事件时序测试。原始 parquet 始终只读；旧策略研究脚本的费用是情景假设，新预测状态阶段不以费用或换手评价形态。
