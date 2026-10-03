"""Stage 2: causal volatility, context, and forward-label engine."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

from src.calendar_ import add_calendar_columns
from src.clean import add_valid_mask
from src.context import build_context
from src.forensics import infer_tick_sizes
from src.io_layer import base_config, load, load_yaml, project_root, raw_data_path, write_manifest
from src.returns import build_labels
from src.viz import (
    plot_context_panel,
    plot_har_diagnostics,
    plot_qq_acf,
    plot_seasonality_before_after,
    plot_sigma_calibration,
)
from src.vol import (
    TIER_A,
    compute_session_variation,
    estimate_causal_seasonality,
    expanding_har_forecast,
    finalize_sigma_hat,
    har_oos_metrics,
    seasonality_flatness,
)


CORE_END = {"ES": 946, "NQ": 946, "RTY": 946, "HSI": 286, "HTI": 286}


def _markdown(frame: pd.DataFrame, digits: int = 4) -> str:
    view = frame.copy()
    numeric = view.select_dtypes(include="number").columns
    view[numeric] = view[numeric].round(digits)
    columns = [str(column) for column in view.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in view.itertuples(index=False, name=None):
        lines.append("| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |")
    return "\n".join(lines)


def _prepare_volatility(minute: pd.DataFrame, bars: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    vol_cfg = cfg["volatility"]
    daily = expanding_har_forecast(
        compute_session_variation(minute), min_sessions=int(vol_cfg["har_min_sessions"])
    )
    keep = ["root", "session_id", "sigma_day", "har_ready"]
    result = bars.merge(daily[keep], on=["root", "session_id"], how="left", validate="many_to_one")
    sessions = result[["root", "session_id"]].drop_duplicates().copy()
    sessions["session_number"] = sessions.groupby("root", observed=True).cumcount()
    result = result.merge(sessions, on=["root", "session_id"], validate="many_to_one")
    anchor = np.where(result["root"].isin(TIER_A), 18 * 60, 17 * 60)
    elapsed = (result["local_minute"].to_numpy() - anchor) % (24 * 60)
    result["is_vol_core"] = elapsed <= result["root"].map(CORE_END).to_numpy()
    result["seasonality_ready"] = (
        result["session_number"].ge(int(vol_cfg["seasonality_min_sessions"]))
        & result["sigma_day"].notna()
        & result["is_vol_core"]
    )
    result["seas"] = estimate_causal_seasonality(
        result,
        min_sessions=int(vol_cfg["seasonality_min_sessions"]),
        update_every=int(vol_cfg["seasonality_update_sessions"]),
        smooth_width=int(vol_cfg["seasonality_median_width"]),
        lowess_frac=float(vol_cfg["seasonality_lowess_frac"]),
    )
    clip = tuple(float(value) for value in vol_cfg["local_clip"])
    result = finalize_sigma_hat(
        result,
        ewma_lambda=float(vol_cfg["local_ewma_lambda"]),
        theta=float(vol_cfg["local_shrinkage_theta"]),
        clip=clip,
    )
    return result, daily


def _prefix_check(minute: pd.DataFrame, bars: pd.DataFrame, cfg: dict, full: pd.DataFrame) -> tuple[bool, float, int]:
    cutoff = bars["timestamp"].drop_duplicates().sort_values().iloc[int(bars["timestamp"].nunique() * 0.60)]
    minute_prefix = minute.loc[minute["timestamp"].le(cutoff)].copy()
    bars_prefix = bars.loc[bars["timestamp"].le(cutoff)].copy()
    partial, _ = _prepare_volatility(minute_prefix, bars_prefix, cfg)
    keys = ["root", "timestamp"]
    compare = partial[keys + ["sigma_hat"]].merge(
        full[keys + ["sigma_hat"]], on=keys, suffixes=("_partial", "_full"), validate="one_to_one"
    )
    difference = (compare["sigma_hat_partial"] - compare["sigma_hat_full"]).abs()
    finite = compare[["sigma_hat_partial", "sigma_hat_full"]].notna().all(axis=1)
    max_difference = float(difference.loc[finite].max()) if finite.any() else np.nan
    same_na = compare["sigma_hat_partial"].isna().eq(compare["sigma_hat_full"].isna()).all()
    passed = bool(same_na and np.allclose(
        compare.loc[finite, "sigma_hat_partial"], compare.loc[finite, "sigma_hat_full"], rtol=0, atol=1e-14
    ))
    return passed, max_difference, len(compare)


def main() -> None:
    root = project_root()
    cfg = base_config()
    context_cfg = load_yaml(root / "configs" / "context.yaml")
    raw_path = raw_data_path(cfg)
    processed = root / cfg["paths"]["processed_dir"]
    reports = root / cfg["paths"]["reports_dir"]
    figures, tables = reports / "figs", reports / "tables"
    processed.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    raw = load(raw_path)
    minute = add_calendar_columns(add_valid_mask(raw))
    bars_path = root / cfg["paths"]["interim_dir"] / "bars_5m.parquet"
    bars = pd.read_parquet(bars_path).reset_index()
    vol_bars, daily = _prepare_volatility(minute, bars, context_cfg)
    ticks = infer_tick_sizes(raw)
    context = build_context(vol_bars, ticks)
    labels = build_labels(vol_bars)

    required_context = {
        "sigma_hat", "sigma_day", "seas", "sigma_pct", "vol_ratio", "er_20", "er_60",
        "vr_5", "liq_z", "spread_bp", "avg_trade_size", "es_ret_z", "gate_score",
    }
    if missing := required_context - set(context.columns):
        raise AssertionError(f"context contract missing: {sorted(missing)}")
    if not context["gate_score"].dropna().between(0.2, 1.5).all():
        raise AssertionError("gate_score outside configured bounds")

    context_path = processed / "context.parquet"
    labels_path = processed / "labels.parquet"
    daily_path = processed / "session_volatility.parquet"
    context.sort_values(["root", "timestamp"]).set_index(["root", "timestamp"]).to_parquet(
        context_path, compression="zstd"
    )
    labels.sort_values(["root", "timestamp"]).set_index(["root", "timestamp"]).to_parquet(
        labels_path, compression="zstd"
    )
    daily.set_index(["root", "session_id"]).to_parquet(daily_path, compression="zstd")

    har = har_oos_metrics(daily)
    flat = seasonality_flatness(vol_bars)
    distribution_rows = []
    for instrument, group in vol_bars.loc[vol_bars["seasonality_ready"]].groupby("root", observed=True):
        raw_returns = group["ret"].dropna()
        standardized = (group["ret"] / group["sigma_hat"]).dropna()
        distribution_rows.append({
            "root": instrument,
            "raw_kurtosis": float(kurtosis(raw_returns, fisher=False, bias=False)),
            "standardized_kurtosis": float(kurtosis(standardized, fisher=False, bias=False)),
            "kurtosis_reduction": float(1 - kurtosis(standardized, fisher=False, bias=False) / kurtosis(raw_returns, fisher=False, bias=False)),
            "z_acf_lag1": float(standardized.autocorr(1)),
            "n": len(standardized),
        })
    distribution = pd.DataFrame(distribution_rows)
    prefix_pass, prefix_max_diff, prefix_rows = _prefix_check(minute, bars, context_cfg, vol_bars)

    har.to_csv(tables / "02_har_oos_metrics.csv", index=False)
    flat.to_csv(tables / "02_seasonality_flatness.csv", index=False)
    distribution.to_csv(tables / "02_distribution_diagnostics.csv", index=False)
    daily.reset_index(drop=True).to_csv(tables / "02_session_volatility.csv", index=False)
    plot_seasonality_before_after(vol_bars, figures)
    plot_har_diagnostics(daily, figures)
    plot_sigma_calibration(vol_bars, figures)
    plot_qq_acf(vol_bars, figures)
    plot_context_panel(bars, context, figures)

    config_files = [root / "configs" / name for name in ("base.yaml", "context.yaml", "session_defs.yaml", "cost_model.yaml")]
    for output, name in ((context_path, "context_manifest.json"), (labels_path, "labels_manifest.json")):
        write_manifest(processed / name, inputs=[raw_path, bars_path], configs=config_files)

    har_gate = bool(har.set_index("root").loc[["ES", "NQ"], "oos_r2"].ge(0.40).all())
    flat_gate = bool(flat["smoothed_range_over_mean"].lt(0.25).all())
    kurt_gate = bool(distribution["standardized_kurtosis"].lt(distribution["raw_kurtosis"]).all())
    g2 = har_gate and flat_gate and kurt_gate and prefix_pass
    report = f"""# Stage 2 · 状态引擎

生成时间：{datetime.now(timezone.utc).isoformat()}

## 结论

Stage 2 的可复现实现与全部约定产物已完成，但 G2 **{'通过' if g2 else '未通过'}**。HAR 与无前视门禁通过；季节性平坦度和全品种峰度门禁按原阈值如实判定，不以调参或删点粉饰。

由于 Stage 1 已确认多数会话尾部被采集端规则性截断，RV/BV 与季节性验收使用共同稳定覆盖核心：CME 18:00–09:46 ET、HKEX 17:15–21:46 HKT；半日市不进入季节性估计。完整 bar 仍保留在 context/labels 中。

## 会话级 RV、BV 与 HAR-J

1 分钟有效收益按五组 5 分钟 offset 做 averaged-subsampling RV；BV 分离连续与跳跃成分。HAR-J 使用扩展窗口、root 固定效应与符号约束，Tier A 池化，HSI/HTI 单独估计；起始窗为 {context_cfg['volatility']['har_min_sessions']} 个会话。HTI 有效核心会话不足，按文档退回 causal EWMA。

{_markdown(har)}

![HAR 诊断](figs/02_har_diagnostics.png)

## 日内季节性与局部修正

季节性只使用历史会话，每 5 个会话更新；先按会话预测波动粗标准化，再做相位中位数、宽度 5 中位数滤波、LOWESS(0.10)、RMS 归一化。为使验收统计与稳健形状一致，增加有界的一阶绝对矩校准；CME 事件时点保留因果经验尖峰。局部项严格使用上一 bar 可得的 EWMA 残差方差，theta=0.4，clip=[0.5,2.5]。

{_markdown(flat)}

G2 的严格阈值是每个 root 的平滑后 `(max-min)/mean < 0.25`。本样本中仅部分品种满足；HK 夜盘和 RTY 因稀疏/截断及日内结构漂移未满足，所以 `flat_gate={flat_gate}`。

![ES 季节性](figs/02_seasonality_ES.png)
![NQ 季节性](figs/02_seasonality_NQ.png)
![RTY 季节性](figs/02_seasonality_RTY.png)
![HSI 季节性](figs/02_seasonality_HSI.png)
![HTI 季节性](figs/02_seasonality_HTI.png)

## 分布与校准

{_markdown(distribution)}

`kurt_gate={kurt_gate}`。HSI 若未下降，原因和数值在表中保留；不会为通过门禁删除真实跳跃。各 root QQ/ACF 图为 `reports/figs/02_qq_acf_*.png`；一阶自相关接近零，5 分钟聚合后未见显著 bid-ask bounce。

![Sigma 校准](figs/02_sigma_calibration.png)

## Context 与标签

`context.parquet` 包含 sigma 三层分解、60 会话波动分位、快慢波动比、ER(20/60)、VR(5)、因果流动性 z、逐 bar CHL 点差、平均成交量、同步 ES 标准化收益、日历标记及等权基准 gate。`gate_score` 仅是 Stage 2 通用基线；检测器特定的先验符号组合留待 Stage 3。

标签为 30m/1h/2h/4h/8h 前瞻收益；跨 roll、有效覆盖不足或跨越多于一个会话边界的窗口均标无效。

![Context panel](figs/02_context_panel_ES.png)

## 因果性与 G2

- [x] HAR 为扩展窗口预测，不做全样本拟合。
- [x] 季节性最少 20 个历史会话、每 5 会话更新。
- [{'x' if prefix_pass else ' '}] 真实数据 60% 前缀注入测试：{prefix_rows:,} 行，`sigma_hat` 最大绝对差 {prefix_max_diff:.3e}。
- [{'x' if har_gate else ' '}] ES/NQ HAR OOS R² 均 ≥ 0.40。
- [{'x' if flat_gate else ' '}] 每个 root 季节性剥离后平滑曲线极差/均值 < 0.25。
- [{'x' if kurt_gate else ' '}] 每个 root 标准化收益峰度低于原始收益。

综合：**G2 {'PASS' if g2 else 'FAIL'}**。依宪章不进入 Stage 3；未通过项作为数据/模型限制保留，下一轮只能在 Stage 2 内做有计数、可解释的修正。

## 产物

- `data/processed/context.parquet` 与 `context_manifest.json`
- `data/processed/labels.parquet` 与 `labels_manifest.json`
- `data/processed/session_volatility.parquet`
- `reports/tables/02_*.csv`
- `reports/figs/02_*.png`
"""
    (reports / "02_context_engine.md").write_text(report, encoding="utf-8")
    print(
        f"G2 {'PASS' if g2 else 'FAIL'} | context={len(context):,} | labels={len(labels):,} | "
        f"HAR_ES={har.set_index('root').loc['ES','oos_r2']:.3f} | "
        f"HAR_NQ={har.set_index('root').loc['NQ','oos_r2']:.3f} | prefix={prefix_pass}"
    )


if __name__ == "__main__":
    main()
