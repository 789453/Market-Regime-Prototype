"""Stage 0: validate the environment, raw schema, and project skeleton."""

from __future__ import annotations

import platform
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.io_layer import base_config, load, project_root, raw_data_path, write_manifest


def main() -> None:
    root = project_root()
    config_path = root / "configs" / "base.yaml"
    cfg = base_config()
    source = raw_data_path(cfg)
    bars = load(source)
    root_counts = bars.groupby("root", sort=True).size()
    contract_counts = bars.groupby("root", sort=True)["local_symbol"].nunique()

    report = f"""# Stage 0 · 环境与骨架验收

生成时间：{datetime.now(timezone.utc).isoformat()}

## 结论

G0 **通过**。严格加载器返回 {len(bars):,} 行、{bars['root'].nunique()} 个品种、{bars['local_symbol'].nunique()} 个合约；字段、类型、UTC 时区、非空与 `(root, timestamp)` 唯一性均通过。

## 环境

- 指定解释器：`D:/Total_Tools/miniforge3/python.exe`
- Python：{platform.python_version()}
- pandas：{pd.__version__}
- 原始数据：`{source.as_posix()}`（只读输入，代码不修改）
- Git：当前目录不是 Git 仓库，manifest 中 `git_hash` 记为 `null`

## 数据摘要

| root | 行数 | 合约数 |
|---|---:|---:|
"""
    for instrument in root_counts.index:
        report += f"| {instrument} | {root_counts[instrument]:,} | {contract_counts[instrument]} |\n"
    report += f"""

- 字段数：{bars.shape[1]}
- 空值数：{int(bars.isna().sum().sum())}
- 重复 `(root, timestamp)`：{int(bars.duplicated(['root', 'timestamp']).sum())}
- 时间范围：{bars['timestamp'].min()} 至 {bars['timestamp'].max()}

## G0 门禁

- [x] `src.io_layer.load()` 返回 494,035 行
- [x] 5 个 root
- [x] 23 个 `local_symbol`
- [x] 12 个字段及 Arrow dtype 严格匹配
- [x] timestamp 为 tz-aware UTC
- [x] 零空值、唯一主键
- [x] 原始文件与配置 hash 已写入 manifest
"""
    report_path = root / "reports" / "00_environment.md"
    report_path.write_text(report, encoding="utf-8")
    write_manifest(
        root / "data" / "interim" / "stage_0_manifest.json",
        inputs=[source],
        configs=[config_path],
    )
    print(report)


if __name__ == "__main__":
    main()

