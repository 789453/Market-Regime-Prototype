"""Post-selection diagnostics only; no parameters or test choices are changed here."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCOPE = ROOT / "reports/crypto/predictive_states_v1"


def paired_week(values: pd.DataFrame, a: str, b: str, *, seed: int = 20260929) -> tuple[float, float, float]:
    daily = values.groupby("date")[[a, b]].mean().sort_index()
    difference = (daily[a] - daily[b]).to_numpy()
    rng = np.random.default_rng(seed)
    n = len(difference)
    starts = rng.integers(0, n, size=(2000, int(np.ceil(n / 7))))
    positions = ((starts[:, :, None] + np.arange(7)) % n).reshape(2000, -1)[:, :n]
    draws = difference[positions].mean(axis=1)
    return float(difference.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def residual_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    market = frame.groupby("available_at", sort=False).return4.transform("mean")
    frame = frame.copy()
    frame["market_residual_bp"] = (frame.return4 - market) * 10000
    frame["date"] = frame.available_at.dt.floor("D")
    days = pd.date_range(frame.date.min(), frame.date.max(), freq="D", tz="UTC")
    states = sorted(frame.loc[frame.accepted, "state"].unique())
    rng = np.random.default_rng(20260929)
    n = len(days)
    starts = rng.integers(0, n, size=(3000, int(np.ceil(n / 7))))
    positions = ((starts[:, :, None] + np.arange(7)) % n).reshape(3000, -1)[:, :n]
    alpha = .05 / max(len(states), 1)
    result = []
    for state in states:
        part = frame.loc[frame.accepted & frame.state.eq(state)]
        summed = part.groupby("date").market_residual_bp.sum().reindex(days, fill_value=0).to_numpy()
        count = part.groupby("date").size().reindex(days, fill_value=0).to_numpy()
        numerator = summed[positions].sum(axis=1)
        denominator = count[positions].sum(axis=1)
        draws = np.divide(numerator, denominator, out=np.full(len(numerator), np.nan), where=denominator > 0)
        low, high = np.nanquantile(draws, [alpha / 2, 1 - alpha / 2])
        share = part.symbol.value_counts(normalize=True)
        result.append({"state": int(state), "rows": len(part), "days": part.date.nunique(),
                       "mean_market_residual_bp": float(part.market_residual_bp.mean()),
                       "bonferroni_week_low_bp": float(low), "bonferroni_week_high_bp": float(high),
                       "largest_symbol_share": float(share.iloc[0]),
                       "mean_raw_4h_bp": float(part.return4.mean() * 10000)})
    return pd.DataFrame(result)


def calibration(frame: pd.DataFrame, name: str, task: str, target: str,
                event_categories: list[int]) -> pd.DataFrame:
    probability = frame[[f"{name}_{task}_p{c}" for c in event_categories]].sum(axis=1)
    observed = frame[target].isin(event_categories).astype(float)
    bucket = pd.qcut(probability, 10, duplicates="drop")
    out = pd.DataFrame({"probability": probability, "observed": observed, "bucket": bucket}).groupby("bucket", observed=True).agg(
        mean_probability=("probability", "mean"), observed_fraction=("observed", "mean"), rows=("observed", "size")).reset_index(drop=True)
    out.insert(0, "target", task + "_" + "+".join(map(str, event_categories)))
    out.insert(0, "model", name)
    return out


def main() -> None:
    manifest = json.loads((SCOPE / "manifest.json").read_text(encoding="utf-8"))
    selected = manifest["winner_validation_only"]
    daily = pd.read_csv(SCOPE / "validation_daily_loss.csv", parse_dates=["date"])
    rep = selected.removesuffix("_split").removesuffix("_root")
    root = daily.loc[daily.model == rep + "_root", ["date", "overall"]].rename(columns={"overall": "root"})
    split = daily.loc[daily.model == rep + "_split", ["date", "overall"]].rename(columns={"overall": "split"})
    paired = root.merge(split, on="date")
    validation_gain = paired_week(paired, "split", "root")

    prediction = pd.read_parquet(SCOPE / "test_predictions.parquet")
    prediction["date"] = prediction.available_at.dt.floor("D")
    prediction["month"] = prediction.available_at.dt.strftime("%Y-%m")
    monthly = prediction.groupby("month").agg(rows=("state_logloss", "size"),
        state_logloss=("state_logloss", "mean"), direct_logloss=("direct_logloss", "mean"),
        background_logloss=("background_logloss", "mean"), coverage=("accepted", "mean")).reset_index()
    monthly.to_csv(SCOPE / "test_monthly_scores.csv", index=False)
    comparisons = []
    for task in ("4h_return", "24h_return", "4h_vol"):
        for competitor in ("direct", "background"):
            difference = paired_week(prediction, f"state_{task}_logloss", f"{competitor}_{task}_logloss")
            comparisons.append({"target": task, "state_minus": competitor,
                                "daily_mean": difference[0], "week_low": difference[1], "week_high": difference[2]})
    comparisons.append({"target": "validation_split_vs_root", "state_minus": "root",
                        "daily_mean": validation_gain[0], "week_low": validation_gain[1], "week_high": validation_gain[2]})
    pd.DataFrame(comparisons).to_csv(SCOPE / "paired_task_intervals.csv", index=False)

    residual = residual_intervals(prediction)
    residual.to_csv(SCOPE / "test_market_residual_intervals.csv", index=False)
    calibration_cards = []
    for name in ("state", "direct", "background"):
        calibration_cards.append(calibration(prediction, name, "4h_return", "target_4h_bin", [0, 1]))
        calibration_cards.append(calibration(prediction, name, "4h_vol", "target_high_rv4", [1]))
    pd.concat(calibration_cards, ignore_index=True).to_csv(SCOPE / "test_calibration.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    interval = residual.sort_values("mean_market_residual_bp")
    x = np.arange(len(interval))
    axes[0].errorbar(x, interval.mean_market_residual_bp,
                     yerr=np.vstack((interval.mean_market_residual_bp - interval.bonferroni_week_low_bp,
                                     interval.bonferroni_week_high_bp - interval.mean_market_residual_bp)),
                     fmt="o", capsize=2)
    axes[0].axhline(0, color="gray", lw=.8)
    axes[0].set_xticks(x, interval.state.astype(str), rotation=45)
    axes[0].set_ylabel("4h return less simultaneous 12-coin return, bp")
    axes[0].set_title("Test local effect; Bonferroni 7-day intervals")
    score = pd.read_csv(SCOPE / "all_scores.csv")
    test = score.loc[score.period == "test"]
    columns = ["logloss_4h_return", "logloss_24h_return", "logloss_4h_vol"]
    for _, row in test.iterrows():
        axes[1].plot(range(3), [row[c] for c in columns], marker="o", label=row.model)
    axes[1].set_xticks(range(3), ["4h return", "24h return", "4h volatility"])
    axes[1].set_ylabel("Distribution log loss")
    axes[1].set_title("Held-out test: same targets")
    axes[1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(SCOPE / "test_distribution_and_residual.png", dpi=160)
    plt.close(figure)
    print(pd.DataFrame(comparisons).to_string(index=False))
    print(residual.to_string(index=False))


if __name__ == "__main__":
    main()
