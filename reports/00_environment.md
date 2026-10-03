# Stage 0 · 环境与骨架验收

生成时间：2026-09-04T13:52:18.646761+00:00

## 结论

G0 **通过**。严格加载器返回 494,035 行、5 个品种、23 个合约；字段、类型、UTC 时区、非空与 `(root, timestamp)` 唯一性均通过。

## 环境

- 指定解释器：`D:/Total_Tools/miniforge3/python.exe`
- Python：3.12.12
- pandas：2.3.3
- 原始数据：`D:/Trading/HongKongQuant/time_selecting_strategy/raw_data/six_month_index_futures_minute(1).parquet`（只读输入，代码不修改）
- Git：当前目录不是 Git 仓库，manifest 中 `git_hash` 记为 `null`

## 数据摘要

| root | 行数 | 合约数 |
|---|---:|---:|
| ES | 136,087 | 3 |
| HSI | 42,890 | 7 |
| HTI | 42,884 | 7 |
| NQ | 136,087 | 3 |
| RTY | 136,087 | 3 |


- 字段数：12
- 空值数：0
- 重复 `(root, timestamp)`：0
- 时间范围：2026-03-01 23:00:00+00:00 至 2026-09-02 13:36:00+00:00

## G0 门禁

- [x] `src.io_layer.load()` 返回 494,035 行
- [x] 5 个 root
- [x] 23 个 `local_symbol`
- [x] 12 个字段及 Arrow dtype 严格匹配
- [x] timestamp 为 tz-aware UTC
- [x] 零空值、唯一主键
- [x] 原始文件与配置 hash 已写入 manifest
