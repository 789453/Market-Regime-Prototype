# 加密 K 线源初步概况（2026-09-28）

数据源只读：`D:\Trading\practical_crypto_strategy\data\parquet`。由 `scripts/inspect_crypto_parquet.py` 逐文件读取；机器可读明细见 `crypto_data_inventory.json`。

| 频率 | 每标的行数 | 12 标的总行数 | 首条 UTC | 末条 UTC | 时间缺口/重复 |
|---|---:|---:|---|---|---:|
| 5m | 392,544 | 4,710,528 | 2023-01-01 00:00 | 2026-09-24 23:55 | 0 / 0 |
| 15m | 130,848 | 1,570,176 | 2023-01-01 00:00 | 2026-09-24 23:45 | 0 / 0 |
| 1h | 32,712 | 392,544 | 2023-01-01 00:00 | 2026-09-24 23:00 | 0 / 0 |

12 标的为 ADA、AVAX、BCH、BNB、BTC、DOGE、ETH、LINK、LTC、SOL、TRX、XRP（USDT）。全部 36 文件均通过开盘时间对齐、`date` 对齐、收盘毫秒时间、OHLC 关系、价格为正、成交量/笔数非负的基础检查。`log_return` 每文件首条为空。零量 bar 并非缺失时间戳：5m 共 307 条（BTC 43、其余各 24），15m 共 53 条（BTC 9、其余各 4），1h 每标的 1 条，共 12 条；`vwap` 和 `taker_buy_ratio` 对应空值。这些 bar 的经济解释及能否成交仍待核查。

模式为 `open_time` 毫秒整数、`date` UTC timestamp、OHLC、`volume`、`quote_volume`、`trade_count`、主动买入/卖出量及预计算 `log_return/range_pct/body_pct/vwap` 等 19 列。成交量列的数据类型在标的间并不完全相同，数据适配器应按语义转换，而非要求相同物理 dtype。

**尚未核验**：原始市场/合约身份、5m 聚合与 15m/1h 的数值一致性、衍生列计算公式、零量时段的来源、历史纳入规则、资金费率、手续费、bid-ask 与实际可交易容量。时间连续性不能替代这些检查。此报告仅确认基础结构，不评价策略收益。
