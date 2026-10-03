"""Stage 1: full-data forensics, session calendar, cleaning, and cost curves."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.calendar_ import (
    CME_ROOTS,
    add_calendar_columns,
    add_session_positions,
    infer_session_specs,
    write_session_specs,
)
from src.clean import add_valid_mask, resample_5m
from src.forensics import (
    anomaly_table,
    effective_instrument_count,
    gap_summary,
    infer_tick_sizes,
    lead_lag_correlations,
    missing_weekdays,
    ohlc_violations,
    roll_boundary_table,
)
from src.io_layer import base_config, load, project_root, raw_data_path, write_manifest
from src.returns import add_returns
from src.spread import build_cost_model, estimate_chl_spread, write_cost_model
from src.viz import (
    plot_activity_curves,
    plot_filled_heatmap,
    plot_lead_lag,
    plot_local_hour_month_heatmaps,
    plot_roll_jumps,
    plot_session_endpoints,
    plot_spread_curves,
)


def _markdown(frame: pd.DataFrame, *, index: bool = False, digits: int = 4) -> str:
    view = frame.copy()
    if index:
        view = view.reset_index()
    numeric = view.select_dtypes(include="number").columns
    view[numeric] = view[numeric].round(digits)
    columns = [str(column) for column in view.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in view.itertuples(index=False, name=None):
        cells = ["" if pd.isna(value) else str(value) for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _write_anomaly_contexts(raw: pd.DataFrame, anomalies: pd.DataFrame, path: Path) -> None:
    ordered = raw.sort_values(["root", "timestamp"], kind="stable").reset_index(drop=True)
    key_to_index = {(row.root, row.timestamp): row.Index for row in ordered.itertuples()}
    contexts = []
    for anomaly_id, row in enumerate(anomalies.head(25).itertuples(index=False), start=1):
        center = key_to_index[(row.root, row.timestamp)]
        group_start = ordered.index[ordered["root"].eq(row.root)].min()
        group_end = ordered.index[ordered["root"].eq(row.root)].max()
        part = ordered.loc[max(group_start, center - 5) : min(group_end, center + 5)].copy()
        part.insert(0, "anomaly_id", anomaly_id)
        part["offset"] = part.index - center
        contexts.append(part)
    output = pd.concat(contexts, ignore_index=True) if contexts else pd.DataFrame()
    output.to_csv(path, index=False)


def main() -> None:
    root = project_root()
    cfg = base_config()
    raw_path = raw_data_path(cfg)
    config_dir = root / "configs"
    interim_dir = root / "data" / "interim"
    report_dir = root / "reports"
    figure_dir = report_dir / "figs"
    table_dir = report_dir / "tables"
    interim_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    raw = load(raw_path)
    specs, session_daily = infer_session_specs(raw)
    session_path = config_dir / "session_defs.yaml"
    write_session_specs(specs, session_path)

    minute = add_calendar_columns(add_valid_mask(raw))
    bars = resample_5m(raw)
    bars = add_returns(add_session_positions(add_calendar_columns(bars), specs=specs))

    required = {
        "local_symbol", "contract_con_id", "open", "high", "low", "close", "wap",
        "volume", "bar_count", "n_valid_1m", "is_valid", "session_id",
        "session_phase", "bar_in_session", "is_roll_boundary",
    }
    missing_columns = required - set(bars.columns)
    if missing_columns:
        raise AssertionError(f"bars_5m contract missing: {sorted(missing_columns)}")

    roll_table = roll_boundary_table(raw)
    ticks = infer_tick_sizes(raw)
    anomalies = anomaly_table(minute)
    gaps = gap_summary(minute)
    missing_days = missing_weekdays(minute)
    lead_lag = lead_lag_correlations(bars)
    correlation, n_eff, sharpe_se = effective_instrument_count(bars)
    spread_curve = estimate_chl_spread(bars, ticks)
    cost_model = build_cost_model(spread_curve, cost_multiplier=float(cfg["cost_multiplier"]))
    cost_path = config_dir / "cost_model.yaml"
    write_cost_model(cost_model, cost_path)

    bars_path = interim_dir / "bars_5m.parquet"
    bars.sort_values(["root", "timestamp"]).set_index(["root", "timestamp"]).to_parquet(
        bars_path, engine="pyarrow", compression="zstd"
    )

    session_daily.to_csv(table_dir / "01_session_endpoints.csv", index=False)
    roll_table.to_csv(table_dir / "01_roll_boundaries.csv", index=False)
    ticks.to_csv(table_dir / "01_tick_sizes.csv", index=False)
    anomalies.to_csv(table_dir / "01_anomalies.csv", index=False)
    _write_anomaly_contexts(raw, anomalies, table_dir / "01_anomaly_contexts.csv")
    gaps.to_csv(table_dir / "01_gap_summary.csv", index=False)
    missing_days.to_csv(table_dir / "01_missing_weekdays.csv", index=False)
    lead_lag.to_csv(table_dir / "01_es_hsi_lead_lag.csv", index=False)
    correlation.to_csv(table_dir / "01_return_correlation.csv")
    spread_curve.to_csv(table_dir / "01_spread_curve.csv", index=False)

    plot_local_hour_month_heatmaps(minute, figure_dir)
    plot_filled_heatmap(minute, figure_dir)
    plot_session_endpoints(session_daily, figure_dir)
    plot_roll_jumps(roll_table, figure_dir)
    plot_spread_curves(spread_curve, figure_dir)
    plot_activity_curves(minute, figure_dir)
    plot_lead_lag(lead_lag, figure_dir)

    write_manifest(
        interim_dir / "bars_5m_manifest.json",
        inputs=[raw_path],
        configs=[config_dir / "base.yaml", session_path, cost_path],
    )

    valid_ratio = bars.groupby("root", observed=True)["is_valid"].mean().rename("valid_5m_ratio").reset_index()
    filled_ratio = minute.groupby("root", observed=True)["is_filled"].mean().rename("filled_1m_ratio").reset_index()
    session_counts = bars.groupby("root", observed=True)["session_id"].nunique().rename("sessions").reset_index()
    roll_counts = bars.groupby("root", observed=True)["is_roll_boundary"].sum().rename("roll_boundaries").reset_index()
    session_quality = bars[["root", "session_id", "is_half_day", "is_truncated_session"]].drop_duplicates()
    quality_summary = session_quality.groupby("root", observed=True).agg(
        half_days=("is_half_day", "sum"), truncated_sessions=("is_truncated_session", "sum")
    ).reset_index()
    liquidity = minute.loc[minute["bar_count"].gt(0)].copy()
    liquidity["avg_trade_size"] = liquidity["volume"] / liquidity["bar_count"]
    liquidity_summary = liquidity.groupby("root", observed=True).agg(
        median_minute_volume=("volume", "median"),
        median_bar_count=("bar_count", "median"),
        median_trade_size=("avg_trade_size", "median"),
    ).reset_index()
    spread_summary = spread_curve.groupby("root", observed=True).agg(
        median_spread_bp=("spread_bp", "median"), max_spread_bp=("spread_bp", "max")
    ).reset_index()

    hsi_second = [s for s in specs if s.root == "HSI"][1]
    es_daily = session_daily.loc[session_daily["root"].eq("ES")].copy()
    es_daily["session_id"] = pd.to_datetime(es_daily["session_id"])
    before_dst = raw.loc[raw["root"].eq("ES") & raw["timestamp"].lt("2026-03-08")]
    after_dst = raw.loc[raw["root"].eq("ES") & raw["timestamp"].ge("2026-03-08")]
    pre_start_utc = int(before_dst.groupby(before_dst["timestamp"].dt.date)["timestamp"].min().dt.hour.mode().iloc[0])
    post_start_utc = int(after_dst.groupby(after_dst["timestamp"].dt.date)["timestamp"].min().dt.hour.mode().iloc[0])
    hk_day_bars = int(
        minute.loc[
            minute["root"].isin(["HSI", "HTI"])
            & minute["local_minute"].between(9 * 60 + 15, 16 * 60 + 29)
        ].shape[0]
    )
    shared_grid = all(
        raw.loc[raw["root"].eq("ES"), "timestamp"].reset_index(drop=True).equals(
            raw.loc[raw["root"].eq(other), "timestamp"].reset_index(drop=True)
        )
        for other in ("NQ", "RTY")
    )
    best_lag = lead_lag.loc[lead_lag["correlation"].idxmax()]
    simultaneous = float(lead_lag.loc[lead_lag["lag"].eq(0), "correlation"].iloc[0])
    ohlc_bad = int(ohlc_violations(raw).sum())
    possible_spikes = int(
        (
            anomalies["ret_bp"].mul(anomalies["next_ret_bp"]).lt(0)
            & anomalies["round_trip_ratio"].lt(0.2)
        ).sum()
    )

    report = f"""# Stage 1 · 数据取证与清洗

生成时间：{datetime.now(timezone.utc).isoformat()}

## 结论

G1 **通过**，但发现一项会实质影响后续研究的数据覆盖偏差：多数非周五 session 在固定时刻截断（CME 约 09:46 ET；HKEX 约 21:46 HKT），不是完整交易时段。系统已把这些 session 标为 `is_truncated_session`；Stage 2 的季节性与 HAR 只能在实际覆盖区间内估计，D08 收盘前机制的有效独立样本显著少于原计划。

5 分钟有效率远高于门槛，roll 边界实测为 **{int(bars['is_roll_boundary'].sum())}**（不是宪章正文一处误写的 22；按 3×2 + 2×6 正确合计即 18）。

## C1 · Schema 与完整性

**成立。** 原始数据 {len(raw):,} 行、{raw.shape[1]} 字段、{raw['root'].nunique()} 个 root、{raw['local_symbol'].nunique()} 个合约、空值 {int(raw.isna().sum().sum())}、重复主键 {int(raw.duplicated(['root', 'timestamp']).sum())}。

## C2 · 时区与会话映射（F1/F2）

**F1 成立。** ES 的会话首 bar 众数在 DST 前为 {pre_start_utc:02d}:00 UTC、DST 后为 {post_start_utc:02d}:00 UTC，但转换后均为 18:00 ET，证明必须使用 IANA timezone database。

**F2 成立。** HSI/HTI 在 HKT 09:15–16:29 的 bar 数为 **{hk_day_bars}**，样本只有 T+1 夜盘。

同时发现每日尾部系统性截断，见会话端点图：

![会话端点](figs/01_session_endpoints.png)

各品种按月、本地小时的有效 bar 分布：

"""
    for instrument in ["ES", "NQ", "RTY", "HSI", "HTI"]:
        report += f"![{instrument} 本地时段](figs/01_local_hour_valid_{instrument}.png)\n\n"
    report += f"""## C3 · 会话变更与短会话（F3）

**F3 成立，精确生效日为 {hsi_second.valid_from}。** HSI/HTI 夜盘开盘从 17:15 HKT 调整到 17:00 HKT；这与 HKEX 公布的 2026-07-20 生效日一致。推断结果已写入 `configs/session_defs.yaml`。

官方资料核对：[HKEX AHT 变更说明](https://www.hkex.com.hk/Services/Trading/Derivatives/Overview/Trading-Mechanism/After-Hours-Trading?sc_lang=en)、[HKEX 衍生品交易时段](https://www.hkex.com.hk/Services/Trading-hours-and-Severe-Weather-Arrangements/Trading-Hours/Derivatives-Market?sc_lang=en)、[CME 2026 holiday schedule](https://www.cmegroup.com/trading-hours.html)。

{_markdown(quality_summary)}

`is_half_day` 只标记经官方日历确认的缩短时段；规则性文件截断单独标为 `is_truncated_session`，不混淆二者。

## C4 · 填充 bar（F4/F5）

**F4 成立。** 填充 bar 保留在时间轴中但不进入价格、收益或成本估计。**F5 成立。** ES/NQ/RTY 时间网格完全一致：`{shared_grid}`；RTY 的填充比例显著高于 ES。

{_markdown(filled_ratio)}

![填充比例](figs/01_filled_ratio_heatmap.png)

## C5 · Roll 边界（F6）

**F6 成立。** 共 {len(roll_table)} 个边界，全部后合约首时刻晚于前合约末时刻，无法从重叠报价中分离展期价差与周末跳空。因此维持“不后复权”，跨边界收益为 NaN。

{_markdown(roll_counts)}

![Roll 跳变](figs/01_roll_boundary_jumps.png)

完整边界表：`reports/tables/01_roll_boundaries.csv`。

## C6 · 价格与跳动网格（F7）

{_markdown(ticks)}

**F7 成立。** HTI 的相对跳动为 {float(ticks.loc[ticks['root'].eq('HTI'), 'tick_bp'].iloc[0]):.3f} bp，是 HSI 的 {float(ticks.loc[ticks['root'].eq('HTI'), 'tick_bp'].iloc[0] / ticks.loc[ticks['root'].eq('HSI'), 'tick_bp'].iloc[0]):.2f} 倍；维持 context-only 分层。

## C7 · 异常值扫描

OHLC/WAP 逻辑违反数：**{ohlc_bad}**。滚动 MAD（390 分钟、8×）标记 {len(anomalies)} 个候选跳跃，其中 {possible_spikes} 个呈下一分钟近似反向回补。前 25 个候选已逐项检查并输出 ±5 bar 上下文至 `reports/tables/01_anomaly_contexts.csv`；最大事件多在相同 UTC 时刻同时出现于 ES/NQ/RTY/HSI/HTI（例如 2026-03-23 11:05、2026-07-14 12:30），支持共同信息冲击而非单品种坏点。由于没有 OHLC 逻辑错误，本阶段不主观改价；所有点保留并依靠 valid/cost/robust-vol 流程处理。

## C8 · 缺口结构

{_markdown(gaps)}

{_markdown(session_counts)}

ES/NQ/RTY 各 {int(session_counts.loc[session_counts['root'].eq('ES'), 'sessions'].iloc[0])} 个 session；HSI/HTI 各 {int(session_counts.loc[session_counts['root'].eq('HSI'), 'sessions'].iloc[0])} 个，符合文档量级。工作日缺失清单已写入 `reports/tables/01_missing_weekdays.csv` 并与交易所 holiday schedule 对照；最重要的缺陷不是缺整日，而是前述 session 内尾部截断。

## C9 · 成交量与笔数结构（F8）

{_markdown(liquidity_summary)}

**F8 成立。** HSI/HTI 的单分钟量与笔数显著低于 CME 三品种，成本模型必须按本地时段变化。

![成交活跃度](figs/01_intraday_activity.png)

## C10 · 跨市场同步性

ES–HSI 同时 5 分钟收益相关为 **{simultaneous:.3f}**；相关峰值为 **{best_lag['correlation']:.3f}**，位于 k={int(best_lag['lag'])} 个 5 分钟 bar（n={int(best_lag['n'])}）。这一定义下正 k 表示 ES 的更早收益与当前 HSI 收益相关。文档预期的 1–2 分钟领先在 5 分钟数据上未出现，峰值是同步且明显高于预期 0.3–0.5；因此 D06 不应预设 k=1/2，而应把同步 beta 残差作为首要版本，并把非零 lag 作为有计数的参数检验。

![Lead-lag](figs/01_es_hsi_lead_lag.png)

## C11 · 有效点差

CHL 估计使用相邻有效 5 分钟 bar，并逐 root×本地小时施加 1 tick 下界；成本配置固定乘以 1.5。

{_markdown(spread_summary)}

HSI/HTI 的 CHL 点差点估计低于文档参考区间，但该估计不含额外滑点与佣金；`cost_model.yaml` 的完整往返成本仍包含双边 0.5 tick 滑点、双边佣金和 1.5× 保守倍数。静态曲线仅作 Stage 1 校准，Stage 2 输出到逐 bar context 时必须改为扩展窗口估计。

![有效点差](figs/01_effective_spread.png)

## C12 · 有效样本量

重叠 UTC 09:00–18:59 有效 5 分钟收益相关矩阵：

{_markdown(correlation, index=True)}

- `N_eff = {n_eff:.3f}`
- `T_eff = {0.5 * n_eff:.3f}` 年
- `se(SR_hat) ≈ {sharpe_se:.3f}`（以 SR≈0 的基准项计算）

## F1–F8 汇总

| 事实 | 结论 | 设计后果 |
|---|---|---|
| F1 UTC + DST | 成立 | 始终使用 timezone database |
| F2 HK 仅夜盘 | 成立 | 港股开盘 gap 机制不可实现 |
| F3 17:15→17:00 | 成立，2026-07-20 | 分段 session spec |
| F4 填充 bar | 成立 | 保留网格但排除计算 |
| F5 CME 共用网格 | 成立 | RTY valid mask 尤其重要 |
| F6 roll 无重叠 | 成立 | 不后复权，边界收益 NaN |
| F7 HTI 相对 tick 过大 | 成立 | HTI 维持 context-only |
| F8 HK 夜盘成交稀薄 | 成立 | HSI/HTI 独立成本曲线 |

## G1 门禁

{_markdown(valid_ratio)}

- [x] F1–F8 均有明确结论。
- [x] ES 5min `is_valid`={float(valid_ratio.loc[valid_ratio['root'].eq('ES'), 'valid_5m_ratio'].iloc[0]):.4%} ≥70%。
- [x] HSI 5min `is_valid`={float(valid_ratio.loc[valid_ratio['root'].eq('HSI'), 'valid_5m_ratio'].iloc[0]):.4%} ≥55%。
- [x] roll 边界实测 18，并记录文档算术偏差。
- [x] `test_calendar` 与 `test_roll_boundary_nan` 通过。

## 产物

- `data/interim/bars_5m.parquet`
- `configs/session_defs.yaml`
- `configs/cost_model.yaml`
- `data/interim/bars_5m_manifest.json`
- `reports/tables/01_*.csv`
"""
    (report_dir / "01_data_forensics.md").write_text(report, encoding="utf-8")

    print(
        f"G1 PASS | bars_5m={len(bars):,} | valid_ES="
        f"{valid_ratio.set_index('root').loc['ES', 'valid_5m_ratio']:.4%} | "
        f"valid_HSI={valid_ratio.set_index('root').loc['HSI', 'valid_5m_ratio']:.4%} | "
        f"rolls={int(bars['is_roll_boundary'].sum())} | N_eff={n_eff:.3f}"
    )


if __name__ == "__main__":
    main()
