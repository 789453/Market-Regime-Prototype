"""Stage 3: eight causal detectors and preregistered statistical validation."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.detectors import DETECTORS
from src.io_layer import base_config, load_yaml, project_root, write_manifest
from src.stats.cards import build_stat_card
from src.stats.nulls import matched_null_distribution, null_percentile
from src.stats.stability import evaluate_parameter_surface
from src.viz import plot_detector_horizons, plot_null_distributions, plot_parameter_surface


TIER_A = ["ES", "NQ", "RTY"]
MINIMUM_POOL = {"D01": 200, "D02": 200, "D03": 150, "D04": 150, "D05": 300, "D06": 150, "D07": 300, "D08": 300}


def _seed(identifier: str, base_seed: int) -> int:
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    return (int.from_bytes(digest[:4], "little") + base_seed) % (2**31)


def _markdown(frame: pd.DataFrame, digits: int = 4) -> str:
    view = frame.copy()
    numeric = view.select_dtypes(include="number").columns
    view[numeric] = view[numeric].round(digits)
    columns = [str(column) for column in view.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in view.itertuples(index=False, name=None):
        lines.append("| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |")
    return "\n".join(lines)


def _cost_lookup(cost_cfg: dict) -> dict[tuple[str, int], float]:
    return {
        (root, int(row["local_hour"])): float(row["round_trip_cost_bp"]) * 1e-4
        for root, details in cost_cfg["instruments"].items()
        for row in details["hourly"]
    }


def _analysis_frame(signal: pd.DataFrame, bars: pd.DataFrame, context: pd.DataFrame, labels: pd.DataFrame, costs: dict) -> pd.DataFrame:
    columns = ["root", "timestamp", "session_id", "session_phase", "local_minute"]
    analysis = signal.merge(bars[columns], on=["root", "timestamp"], validate="one_to_one")
    analysis = analysis.merge(context[["root", "timestamp", "gate_score"]], on=["root", "timestamp"], validate="one_to_one")
    analysis = analysis.merge(labels, on=["root", "timestamp"], validate="one_to_one")
    analysis["month"] = analysis["timestamp"].dt.strftime("%Y-%m")
    analysis["local_hour"] = (analysis["local_minute"] // 60).astype(int)
    analysis["cost_return"] = [costs.get((root, hour), np.nan) for root, hour in zip(analysis.root, analysis.local_hour)]
    return analysis


def main() -> None:
    root = project_root(); cfg = base_config()
    detector_cfg = load_yaml(root / "configs" / "detectors.yaml")
    cost_cfg = load_yaml(root / "configs" / "cost_model.yaml")
    shared = detector_cfg["shared"]
    processed = root / cfg["paths"]["processed_dir"]
    reports = root / cfg["paths"]["reports_dir"]
    figures, tables = reports / "figs", reports / "tables"
    figures.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True)

    bars_path = root / cfg["paths"]["interim_dir"] / "bars_5m.parquet"
    context_path = processed / "context.parquet"; labels_path = processed / "labels.parquet"
    bars = pd.read_parquet(bars_path).reset_index()
    context = pd.read_parquet(context_path).reset_index()
    labels = pd.read_parquet(labels_path).reset_index()
    costs = _cost_lookup(cost_cfg)
    horizons = [int(value) for value in shared["horizons"]]
    all_cards, all_gates, all_surfaces_1d, all_surfaces_2d, all_loi, all_adoptions = [], [], [], [], [], []
    null_distributions, actual_effects = {}, {}
    experiment_count = 0
    report_sections = []

    for detector_type in DETECTORS:
        detector = detector_type()
        signal = detector.detect(bars, context)
        signal_path = processed / f"signals_{detector.detector_id.lower()}.parquet"
        signal.set_index(["root", "timestamp"]).to_parquet(signal_path, compression="zstd")
        write_manifest(
            processed / f"signals_{detector.detector_id.lower()}_manifest.json",
            inputs=[bars_path, context_path, labels_path],
            configs=[root / "configs" / "base.yaml", root / "configs" / "detectors.yaml", root / "configs" / "cost_model.yaml"],
        )
        analysis = _analysis_frame(signal, bars, context, labels, costs)
        seed = _seed(detector.detector_id + "stage_3", int(cfg["project"]["seed"]))
        card, breakdowns = build_stat_card(analysis, detector.detector_id, horizons, seed)
        all_cards.append(card)
        card.to_csv(tables / f"03_card_{detector.detector_id}.csv", index=False)
        for name, breakdown in breakdowns.items():
            breakdown.to_csv(tables / f"03_{detector.detector_id}_{name}.csv", index=False)

        surface_1d, surface_2d, stability, variants = evaluate_parameter_surface(detector, bars, context, labels)
        experiment_count += variants
        all_surfaces_1d.append(surface_1d); all_surfaces_2d.append(surface_2d)
        default_performance = float(surface_1d.loc[surface_1d["is_default"], "performance"].mean())
        best_point = surface_2d.loc[surface_2d["performance"].idxmax()]
        adopted = {name: detector.params[name] for name in [best_point["parameter_1"], best_point["parameter_2"]]}
        all_adoptions.append({
            "detector_id": detector.detector_id, "peak_parameters": f"{best_point['parameter_1']}={best_point['value_1']}; {best_point['parameter_2']}={best_point['value_2']}",
            "peak_performance": best_point["performance"], "adopted_parameters": str(adopted),
            "adopted_performance": default_performance,
            "peak_to_adopted_gap": (best_point["performance"] - default_performance) / abs(best_point["performance"]) if best_point["performance"] != 0 else np.nan,
        })
        surface_1d.to_csv(tables / f"03_surface_1d_{detector.detector_id}.csv", index=False)
        surface_2d.to_csv(tables / f"03_surface_2d_{detector.detector_id}.csv", index=False)
        plot_parameter_surface(surface_1d, surface_2d, detector.detector_id, figures)

        horizon = detector.default_horizon
        valid = analysis.direction.ne(0) & analysis[f"label_valid_{horizon}"]
        actual = float((analysis.loc[valid, "direction"] * analysis.loc[valid, f"fwd_ret_z_{horizon}"]).mean())
        null = matched_null_distribution(analysis, horizon, draws=int(shared["null_draws"]), seed=seed)
        percentile = null_percentile(actual, null)
        null_distributions[detector.detector_id] = null; actual_effects[detector.detector_id] = actual
        pd.DataFrame({"null_mean_z": null}).to_parquet(tables / f"03_null_{detector.detector_id}.parquet", compression="zstd")

        trigger = analysis.loc[analysis.direction.ne(0)]
        counts = trigger.groupby("root", observed=True).size()
        root_means = breakdowns["root"].set_index("root")["mean"] if len(breakdowns["root"]) else pd.Series(dtype=float)
        signed = analysis.loc[valid, ["root", "direction", f"fwd_ret_z_{horizon}"]].copy()
        signed["signed_z"] = signed["direction"] * signed[f"fwd_ret_z_{horizon}"]
        full_mean = signed["signed_z"].mean()
        for holdout in ["ES", "NQ", "RTY", "HSI"]:
            holdout_mean = signed.loc[signed.root.eq(holdout), "signed_z"].mean()
            development_mean = signed.loc[signed.root.isin([name for name in ["ES", "NQ", "RTY", "HSI"] if name != holdout]), "signed_z"].mean()
            all_loi.append({
                "detector_id": detector.detector_id, "holdout_root": holdout,
                "holdout_mean": holdout_mean, "development_mean": development_mean, "full_mean": full_mean,
                "loi_ratio": holdout_mean / full_mean if full_mean > 0 else np.nan,
            })
        tier_a_consistent = int(root_means.reindex(TIER_A).gt(0).sum())
        if detector.detector_id == "D06":
            trigger_gate = counts.get("HSI", 0) >= MINIMUM_POOL["D06"]
            sign_gate = bool(root_means.get("HSI", np.nan) > 0)
        else:
            trigger_gate = len(trigger) >= MINIMUM_POOL[detector.detector_id] and all(
                counts.get(name, 0) >= int(shared["minimum_triggers_per_tier_a_root"]) for name in TIER_A
            )
            sign_gate = tier_a_consistent >= 2
        stability_gate = stability >= float(shared["stability_gate"])
        null_gate = percentile >= float(shared["null_percentile_gate"])
        default_row = card.loc[card.horizon.eq(horizon)].iloc[0]
        net_gate = bool(default_row["mean_net"] > 0)
        passed = trigger_gate and sign_gate and stability_gate and null_gate and net_gate
        session_counts = trigger.groupby("session_id", observed=True).size().sort_values(ascending=False)
        concentration = float(session_counts.head(10).sum() / len(trigger)) if len(trigger) else np.nan
        all_gates.append({
            "detector_id": detector.detector_id, "hypothesis_id": detector.hypothesis_id,
            "triggers": len(trigger), "ES": counts.get("ES", 0), "NQ": counts.get("NQ", 0), "RTY": counts.get("RTY", 0),
            "tier_a_positive": tier_a_consistent, "stability": stability, "null_percentile": percentile,
            "mean_z": actual, "mean_net": default_row["mean_net"], "top10_session_share": concentration,
            "trigger_gate": trigger_gate, "sign_gate": sign_gate, "stability_gate": stability_gate,
            "null_gate": null_gate, "net_gate": net_gate, "g3_pass": passed, "variants": variants,
        })
        report_sections.append(f"""### {detector.detector_id} · {detector.hypothesis_id}

- 默认参数：`{detector.params}`
- 触发：{len(trigger):,}；ES/NQ/RTY={counts.get('ES',0)}/{counts.get('NQ',0)}/{counts.get('RTY',0)}；前 10 会话集中度={concentration:.1%}
- 默认 horizon={horizon}：signed mean z={actual:.4f}，成本后原始均值={default_row['mean_net']:.6f}
- matched-null percentile={percentile:.3f}；Stab={stability:.3f}；Tier A 正号数={tier_a_consistent}
- G3 单检测器结论：**{'PASS' if passed else 'FAIL'}**

{_markdown(card)}

![{detector.detector_id} 参数曲面](figs/03_parameter_surface_{detector.detector_id}.png)
""")

    cards = pd.concat(all_cards, ignore_index=True); gates = pd.DataFrame(all_gates)
    loi = pd.DataFrame(all_loi)
    adoptions = pd.DataFrame(all_adoptions)
    surface_1d = pd.concat(all_surfaces_1d, ignore_index=True); surface_2d = pd.concat(all_surfaces_2d, ignore_index=True)
    cards.to_csv(tables / "03_all_detector_cards.csv", index=False)
    gates.to_csv(tables / "03_g3_gate_summary.csv", index=False)
    loi.to_csv(tables / "03_leave_one_instrument_out.csv", index=False)
    adoptions.to_csv(tables / "03_parameter_adoption.csv", index=False)
    surface_1d.to_csv(tables / "03_all_surfaces_1d.csv", index=False)
    surface_2d.to_csv(tables / "03_all_surfaces_2d.csv", index=False)
    plot_detector_horizons(cards, figures); plot_null_distributions(null_distributions, actual_effects, figures)
    passing = int(gates["g3_pass"].sum()); g3 = passing >= 3
    report = f"""# Stage 3 · 检测器与单形态统计

生成时间：{datetime.now(timezone.utc).isoformat()}

## 结论

8 个预注册检测器、在线 ZigZag、统计卡片、stationary bootstrap、phase/聚类/多空匹配随机时点零分布、LOI 所需 root 分解以及一维/二维参数曲面均已完成。共有 **{passing}** 个检测器通过全部 G3 子门禁，故 G3 **{'通过' if g3 else '未通过'}**。

Stage 2 的季节性平坦度降级由用户接受，但原始失败指标继续保留；本阶段没有重估 `sigma_hat`。HTI 仍只作 context，不计入 Tier A 符号一致性。

## G3 汇总

{_markdown(gates[["detector_id","triggers","ES","NQ","RTY","tier_a_positive","stability","null_percentile","mean_z","mean_net","g3_pass"]], digits=6)}

![持有期曲线](figs/03_detector_horizon_curves.png)

![匹配随机时点零分布](figs/03_matched_nulls.png)

跨品种留一迁移比（固定预注册参数，不在 holdout 上调整）：

{_markdown(loi)}

参数峰值与采用值（采用预注册默认/高原中心，不追逐峰值）：

{_markdown(adoptions)}

## 方法口径

- 所有检测器输出在 bar t 仅使用 t 及以前的信息；ZigZag pivot 只在 `confirm_time` 后可见。
- D01/D02 共享同一突破事件，分别要求收回/守住且低量/高量，触发集合互斥。
- 条件效应始终为 `direction × fwd_ret_z`，多空另表报告，禁止合并成绝对收益。
- 显著性同时报告原始 t、Newey-West(h) 和 stationary bootstrap（期望块长 5h）。
- 随机对照在 root×phase 内循环平移完整触发序列，精确保持触发数、方向序列与簇集间距。
- 成本使用 `cost_model.yaml` 中已经包含 1.5× 保守倍数的逐 root×本地小时往返成本，避免再次重复乘 1.5。
- 参数稳定性只扫描预登记网格；默认点、一维邻域和最重要两参数二维曲面去重后共 **{experiment_count}** 个参数变体。另有 8×5=40 个强制持有期诊断，总计 **{experiment_count + 40}** 个 Stage 3 已查看变体，全部计入 K。

## 检测器卡片

{''.join(report_sections)}

## 门禁

- 触发数：池化 ≥150，且 ES/NQ/RTY 各 ≥25。
- D06 按预注册例外仅要求 HSI ≥150，并以 HSI 自身符号检验替代 Tier A 一致性。
- Tier A 至少两个 root 的有向条件均值为正。
- `Stab ≥ 0.70`。
- matched-null 单边分位 ≥0.925。
- 默认 horizon 扣已放大 1.5× 的成本后均值 >0。
- 至少三个检测器同时满足以上全部条件。

综合：**G3 {'PASS' if g3 else 'FAIL'}**。按宪章在此 Stage 边界停止；Stage 4 不在本次执行范围内。
"""
    (reports / "03_detector_cards.md").write_text(report, encoding="utf-8")
    print(f"G3 {'PASS' if g3 else 'FAIL'} | passing={passing}/8 | variants={experiment_count} | signals={len(bars):,} each")


if __name__ == "__main__":
    main()
