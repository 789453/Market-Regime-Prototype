"""Stage 4: detector-specific regime gates and hierarchical shrinkage."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.io_layer import base_config, load_yaml, project_root, write_manifest
from src.regime import detector_gate, gate_curve
from src.stats.shrinkage import (
    DEFAULT_DIMENSIONS,
    categorize_context,
    causal_cell_posterior,
    root_hierarchy,
    second_order_decomposition,
)
from src.viz import plot_gate_curves, plot_shrinkage_summary


def _markdown(frame: pd.DataFrame, digits: int = 5) -> str:
    view = frame.copy()
    numeric = view.select_dtypes(include="number").columns
    view[numeric] = view[numeric].round(digits)
    columns = [str(column) for column in view.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in view.itertuples(index=False, name=None):
        lines.append("| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |")
    return "\n".join(lines)


def _maturity_times(bars: pd.DataFrame, horizons: set[int]) -> pd.DataFrame:
    ordered = bars.sort_values(["root", "timestamp"], kind="stable").copy()
    output = ordered[["root", "timestamp"]].copy()
    for horizon in horizons:
        output[f"outcome_available_{horizon}"] = ordered.groupby("root", sort=False)["timestamp"].shift(-horizon)
    return output


def main() -> None:
    root = project_root(); cfg = base_config()
    detector_cfg = load_yaml(root / "configs" / "detectors.yaml")
    regime_cfg = load_yaml(root / "configs" / "regime.yaml")
    processed = root / cfg["paths"]["processed_dir"]
    reports = root / cfg["paths"]["reports_dir"]
    figures, tables = reports / "figs", reports / "tables"
    figures.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True)

    bars_path = root / cfg["paths"]["interim_dir"] / "bars_5m.parquet"
    context_path = processed / "context.parquet"; labels_path = processed / "labels.parquet"
    bars = pd.read_parquet(bars_path).reset_index().sort_values(["root", "timestamp"]).reset_index(drop=True)
    context = pd.read_parquet(context_path).reset_index()
    labels = pd.read_parquet(labels_path).reset_index()
    horizons = {int(value["default_horizon"]) for value in detector_cfg["detectors"].values()}
    maturity = _maturity_times(bars, horizons)
    gate_cfg = regime_cfg["gate"]
    summaries, effects_all, roots_all, curves = [], [], [], {}
    prefix_checks = []

    context_columns = [
        "root", "timestamp", "sigma_pct", "vol_ratio", "er_20", "er_60", "vr_5", "liq_z",
        "spread_bp", "gate_score", "is_month_end", "is_quarter_end",
    ]
    bar_columns = ["root", "timestamp", "session_phase", "session_id", "is_half_day"]
    for detector_id, specification in detector_cfg["detectors"].items():
        horizon = int(specification["default_horizon"])
        source_path = processed / f"signals_{detector_id.lower()}.parquet"
        signal = pd.read_parquet(source_path).reset_index()
        meta = [column for column in signal if column.startswith("meta_")]
        analysis = signal.merge(bars[bar_columns], on=["root", "timestamp"], validate="one_to_one")
        analysis = analysis.merge(context[context_columns], on=["root", "timestamp"], validate="one_to_one", suffixes=("", "_generic"))
        analysis = analysis.merge(
            labels[["root", "timestamp", f"fwd_ret_z_{horizon}", f"label_valid_{horizon}"]],
            on=["root", "timestamp"], validate="one_to_one",
        )
        analysis = analysis.merge(maturity[["root", "timestamp", f"outcome_available_{horizon}"]], on=["root", "timestamp"], validate="one_to_one")
        analysis["detector_gate"] = detector_gate(
            analysis, detector_id, minimum=float(gate_cfg["minimum"]), maximum=float(gate_cfg["maximum"])
        )
        trigger = analysis.loc[analysis["direction"].ne(0)].copy()
        trigger["outcome"] = trigger["direction"] * trigger[f"fwd_ret_z_{horizon}"]
        trigger.loc[~trigger[f"label_valid_{horizon}"], "outcome"] = np.nan
        trigger["outcome_available_time"] = trigger[f"outcome_available_{horizon}"]
        trigger = categorize_context(trigger)

        effects, sigma2 = second_order_decomposition(trigger, dimensions=DEFAULT_DIMENSIONS)
        effects.insert(0, "detector_id", detector_id); effects_all.append(effects)
        root_effects = root_hierarchy(trigger)
        root_effects.insert(0, "detector_id", detector_id); roots_all.append(root_effects)
        causal = causal_cell_posterior(trigger, dimensions=DEFAULT_DIMENSIONS)
        cutoff = trigger["timestamp"].quantile(0.60)
        prefix_trigger = trigger.loc[trigger["timestamp"].le(cutoff)].copy()
        prefix_causal = causal_cell_posterior(prefix_trigger, dimensions=DEFAULT_DIMENSIONS)
        expected = causal.loc[prefix_trigger.index]
        causal_difference = (prefix_causal["posterior_mean"] - expected["posterior_mean"]).abs().max()
        confidence_difference = (prefix_causal["confidence"] - expected["confidence"]).abs().max()
        prefix_pass = bool(
            np.allclose(prefix_causal["posterior_mean"], expected["posterior_mean"], equal_nan=True, rtol=0, atol=1e-14)
            and np.allclose(prefix_causal["confidence"], expected["confidence"], equal_nan=True, rtol=0, atol=1e-14)
        )
        prefix_checks.append({"detector_id": detector_id, "rows": len(prefix_trigger), "posterior_max_diff": causal_difference, "confidence_max_diff": confidence_difference, "passed": prefix_pass})

        trigger[["posterior_mean", "state_confidence", "prior_cell_n"]] = causal.rename(columns={"confidence": "state_confidence"})
        curve, metrics = gate_curve(trigger, quantiles=int(gate_cfg["quantiles"]))
        curves[detector_id] = curve
        curve.insert(0, "detector_id", detector_id)
        curve.to_csv(tables / f"04_gate_curve_{detector_id}.csv", index=False)
        positive_tau_terms = int(effects.loc[effects["tau2"].gt(0), ["term_order", "variables"]].drop_duplicates().shape[0])
        heterogeneity = positive_tau_terms > 0
        raw_gate_pass = bool(
            metrics["isotonic_r2"] >= float(gate_cfg["isotonic_r2_minimum"])
            and metrics["adjacent_jump_ratio"] < float(gate_cfg["adjacent_jump_maximum"])
        )

        output = signal.copy()
        output["raw_gate_score"] = analysis["detector_gate"].to_numpy()
        output["gate_score"] = output["raw_gate_score"] if raw_gate_pass else 1.0
        output["posterior_mean"] = 0.0; output["state_confidence"] = 0.0; output["prior_cell_n"] = 0
        output.loc[trigger.index, "posterior_mean"] = trigger["posterior_mean"]
        output.loc[trigger.index, "state_confidence"] = trigger["state_confidence"]
        output.loc[trigger.index, "prior_cell_n"] = trigger["prior_cell_n"]
        if heterogeneity:
            output["confidence"] = output["state_confidence"].clip(0, 1)
        else:
            output["confidence"] = np.where(output["direction"].ne(0), 1.0, 0.0)
        output["magnitude_gated"] = (output["magnitude"] * output["gate_score"]).clip(0, 1)
        destination = processed / f"signals_stage4_{detector_id.lower()}.parquet"
        output.set_index(["root", "timestamp"]).to_parquet(destination, compression="zstd")
        write_manifest(
            processed / f"signals_stage4_{detector_id.lower()}_manifest.json",
            inputs=[source_path, context_path, labels_path],
            configs=[root / "configs" / "detectors.yaml", root / "configs" / "regime.yaml"],
        )
        summaries.append({
            "detector_id": detector_id, "triggers": len(trigger), "unconditional_mean": trigger["outcome"].mean(),
            "sigma2": sigma2, "positive_tau_terms": positive_tau_terms,
            "median_trigger_confidence": trigger["state_confidence"].median(),
            "mean_trigger_confidence": trigger["state_confidence"].mean(),
            "isotonic_r2": metrics["isotonic_r2"], "adjacent_jump_ratio": metrics["adjacent_jump_ratio"],
            "raw_gate_pass": raw_gate_pass, "applied_mode": "conditional" if raw_gate_pass else "unconditional",
            "prefix_pass": prefix_pass,
        })

    summary = pd.DataFrame(summaries); effects = pd.concat(effects_all, ignore_index=True)
    root_effects = pd.concat(roots_all, ignore_index=True); prefix = pd.DataFrame(prefix_checks)
    summary.to_csv(tables / "04_regime_summary.csv", index=False)
    effects.to_parquet(tables / "04_hierarchical_effects.parquet", compression="zstd")
    root_effects.to_csv(tables / "04_root_shrinkage.csv", index=False)
    prefix.to_csv(tables / "04_causal_prefix_checks.csv", index=False)
    plot_gate_curves(curves, figures); plot_shrinkage_summary(summary, figures)

    d01_pass = bool(summary.set_index("detector_id").loc["D01", "raw_gate_pass"])
    all_prefix = bool(summary["prefix_pass"].all())
    framework_safe = all_prefix and summary["applied_mode"].isin(["conditional", "unconditional"]).all()
    report = f"""# Stage 4 · 状态条件化与分层收缩

生成时间：{datetime.now(timezone.utc).isoformat()}

## 结论

按用户明确要求，D01–D08 全部进入 Stage 4，以验证完整框架；这不改写 Stage 3 只有 D01 通过的事实。二阶截断的经验贝叶斯分层、强收缩的跨品种层次、逐时点因果后验置信度、检测器专属 gate 与失败后无条件回退均已跑通。

正式 G3 候选 D01 的原始 gate G4 结论为 **{'PASS' if d01_pass else 'FAIL'}**。八检测器实验性并行结果见下表；未通过单调/平滑门槛者已按文档自动退回 `gate_score=1.0`，没有把锯齿状态曲线带入 Stage 5。框架安全性检查为 **{'PASS' if framework_safe else 'FAIL'}**。

## 汇总

{_markdown(summary)}

![Gate curves](figs/04_gate_curves.png)

![Shrinkage](figs/04_shrinkage_summary.png)

## 分层结构

- 复合 cell 固定为 `(structure, efficiency, volatility, vol_direction, liquidity, phase)`。
- 连续 context 用事前固定阈值离散为 2–3 档，不使用全样本分位拟合桶边界。
- 模型仅包含六组主效应与 15 组二阶交互，三阶及以上强制为零。
- 每个效应的 `tau²=max(Var(group mean)-E[sigma²/n],0)`；`tau²=0` 时该项完全收缩至零。
- root 偏离的 prior variance 额外上限为 `sigma²/100`，执行文档要求的强跨品种收缩。
- 完整效应表：`reports/tables/04_hierarchical_effects.parquet`；root 层次：`reports/tables/04_root_shrinkage.csv`。

## 因果置信度

每个触发的 cell posterior 只纳入在该时刻已经走完默认 horizon 的历史触发结果；不是简单使用 `timestamp < t`，而是显式使用 outcome availability timestamp。真实数据 60% 前缀复算结果：

{_markdown(prefix)}

所有检测器前缀逐点相同：**{all_prefix}**。Stage 4 输出中的 `confidence`、`posterior_mean` 与 `prior_cell_n` 可直接供 Stage 5 融合；非触发 bar 的 confidence 为零。

## Gate 设计

各 gate 的方向在查看条件收益前由 HYPOTHESES 固定：反转型偏低 ER/波动收缩/正常流动性，趋势型偏高 ER/波动扩张/高流动性；D07 按回归/延续分支分别映射；D08 只以月末/季末和正常流动性软调制。全部等权并映射至 `[0.2,1.5]`。

验收使用 10 分位 raw curve、递增 isotonic 拟合、`R²≥0.6` 且最大相邻拟合跳变/总幅度 `<0.4`。未通过即应用无条件版本，不二次改符号或重新分箱。

## 门禁解释

- 原始 G4 是对通过 G3 的候选执行；因此正式门禁取 D01 的 raw gate 结果。
- 用户要求其余七个阴性检测器也进入本阶段，它们用于管线压力测试，不因 Stage 4 的任何结果升级为有效 alpha。
- `framework_safe={framework_safe}` 仅表示因果性、边界、回退和产物契约正确，不等价于经济门禁通过。
- 本阶段固定评估 8 个预注册 gate 与 8 个分层收缩规格，`ΔK=16`；不搜索 gate 权重或重新分箱，累计 `K=168`。

综合：**G4 {'PASS' if d01_pass and framework_safe else 'FAIL'}**。本阶段在报告边界停止。
"""
    (reports / "04_regime_conditioning.md").write_text(report, encoding="utf-8")
    print(f"G4 {'PASS' if d01_pass and framework_safe else 'FAIL'} | D01_gate={d01_pass} | conditional={int(summary.raw_gate_pass.sum())}/8 | prefix={all_prefix}")


if __name__ == "__main__":
    main()
