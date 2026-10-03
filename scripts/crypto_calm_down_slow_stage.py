"""Controlled slow-scale feature test for calm, declining market history.

The frozen 100-coordinate X2 input and symbol indicators are the control.
Only six known-at-decision slow coordinates change in the candidate.  Every
fold's conditional model is trained on earlier calm-down observations only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.crypto_regime_prototype_stage import CFG, SYMBOLS, lgb_fit, load_data

OUT = ROOT / "reports/crypto/calm_down_slow_stage"
SOURCE = ROOT / "reports/crypto/regime_prototypes_v1/forward_forecasts.parquet"
NAMES = ("relative_7d", "relative_30d", "signed_efficiency_7d",
         "market_trend_7d", "market_trend_30d", "rv_ratio_7d_30d")


def slow_coordinates(data: dict) -> np.ndarray:
    close = data["close"]
    n, coins = close.shape
    rv = np.maximum(data["rv"], 1e-9)
    market = close.mean(axis=1)
    scale = np.sqrt(rv)
    out = np.full((n, coins, 6), np.nan, dtype=np.float32)
    market_rv = np.maximum(data["market_rv"], 1e-9)
    for horizon, slot in ((168, 0), (720, 1)):
        mchange = np.full(n, np.nan)
        mchange[horizon:] = market[horizon:] - market[:-horizon]
        rel = np.full((n, coins), np.nan)
        rel[horizon:] = close[horizon:] - close[:-horizon] - data["beta"][horizon:] * mchange[horizon:, None]
        out[:, :, slot] = np.clip(rel / (scale * np.sqrt(horizon / 24)), -5, 5)
        mslot = 3 if horizon == 168 else 4
        out[:, :, mslot] = np.broadcast_to(
            np.clip(mchange / np.sqrt(market_rv * horizon / 24), -5, 5)[:, None], (n, coins))
    for j in range(coins):
        returns = np.diff(close[:, j], prepend=close[0, j])
        travel = pd.Series(np.abs(returns)).rolling(168, min_periods=168).sum().to_numpy()
        out[:, j, 2] = np.clip((close[:, j] - np.roll(close[:, j], 168)) /
                               np.maximum(travel, 1e-8), -1, 1)
        r7 = pd.Series(rv[:, j]).rolling(168, min_periods=168).mean().to_numpy()
        r30 = pd.Series(rv[:, j]).rolling(720, min_periods=720).mean().to_numpy()
        out[:, j, 5] = np.clip(np.log((r7 + 1e-9) / (r30 + 1e-9)), -3, 3)
    out[:720] = np.nan
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    slow = slow_coordinates(data)
    n, coins, width = data["x"].shape
    time = data["time"]
    base = np.concatenate((data["x"], np.broadcast_to(np.eye(coins), (n, coins, coins))), axis=2)
    expanded = np.concatenate((base, slow), axis=2)
    saved = pd.read_parquet(SOURCE)
    folds = [pd.Timestamp(s) for s in CFG["fold_edges"]]
    scores, predictions = [], []
    for fold, (start, end) in enumerate(zip(folds[:-1], folds[1:])):
        train = ((time >= pd.Timestamp(CFG["training_start"])) &
                 (time + pd.Timedelta(hours=CFG["longest_label_purge_hours"]) < start) &
                 np.isfinite(slow).all(axis=(1, 2)) & np.isfinite(data["signature"]).all(axis=(1, 2)))
        eval_ = ((time >= start) & (time < end) & np.isfinite(slow).all(axis=(1, 2)))
        tr, ev = np.flatnonzero(train), np.flatnonzero(eval_)
        if len(tr) < 100 or len(ev) < 100:
            raise AssertionError("invalid slow-scale fold")
        qlo = np.quantile(data["market_rv"][tr], .33)
        condition_train = ((data["market_rv"][tr] < qlo) & (data["market_trend"][tr] < 0))
        condition_eval = ((data["market_rv"][ev] < qlo) & (data["market_trend"][ev] < 0))
        train_clocks = tr[condition_train]
        eval_clocks = ev[condition_eval]
        if len(train_clocks) < 200 or len(eval_clocks) < 30:
            raise AssertionError("insufficient calm-down conditional history")
        y = data["signature"][train_clocks, :, 1].reshape(-1)
        base_model = lgb_fit(base[train_clocks].reshape(-1, width + coins), y, 21000 + fold)
        slow_model = lgb_fit(expanded[train_clocks].reshape(-1, width + coins + 6), y, 21000 + fold)
        compact_base = lgb_fit(base[train_clocks].reshape(-1, width + coins), y,
                               22000 + fold, trees=50, leaves=5)
        compact_slow = lgb_fit(expanded[train_clocks].reshape(-1, width + coins + 6), y,
                               22000 + fold, trees=50, leaves=5)
        base_pred = base_model.predict(base[eval_clocks].reshape(-1, width + coins))
        slow_pred = slow_model.predict(expanded[eval_clocks].reshape(-1, width + coins + 6))
        compact_base_pred = compact_base.predict(base[eval_clocks].reshape(-1, width + coins))
        compact_slow_pred = compact_slow.predict(expanded[eval_clocks].reshape(-1, width + coins + 6))
        actual = data["signature"][eval_clocks, :, 1].reshape(-1)
        scale = data["scale24"][eval_clocks].reshape(-1)
        stamp = pd.DatetimeIndex(np.repeat(time[eval_clocks].to_numpy(), coins))
        ref = saved[(saved.fold == fold) & (saved.regime == "calm_down")].sort_values(["time", "symbol"])
        if len(ref) != len(actual) or not np.array_equal(pd.DatetimeIndex(ref.time).asi8, stamp.asi8):
            raise AssertionError("conditional prediction rows do not match prior forward panel")
        comparator = ref.direct_alpha24.to_numpy() / scale
        calm_both = ref.calm_alpha24.to_numpy() / scale
        for name, pred in (("all_state_direct", comparator), ("both_calms_direct", calm_both),
                           ("calm_down_x2", base_pred),
                           ("calm_down_x2_slow", slow_pred),
                           ("calm_down_x2_compact", compact_base_pred),
                           ("calm_down_x2_slow_compact", compact_slow_pred)):
            scores.append({"fold": fold, "start": str(start), "model": name,
                           "clocks": len(eval_clocks), "coin_rows": len(actual),
                           "mse_normalized_alpha24": float(np.mean((actual - pred) ** 2)),
                           "raw_alpha24_corr": float(np.corrcoef(actual * scale, pred * scale)[0, 1])})
        predictions.append(pd.DataFrame({"time": stamp, "symbol": np.tile(SYMBOLS, len(eval_clocks)),
            "fold": fold, "actual_normalized_alpha24": actual, "base_prediction": base_pred,
            "slow_prediction": slow_pred, "compact_base_prediction": compact_base_pred,
            "compact_slow_prediction": compact_slow_pred, "reference_prediction": comparator,
            "scale24": scale}))
        print("calm-down fold", fold, "train clocks", len(train_clocks), "eval clocks", len(eval_clocks), flush=True)
    scores = pd.DataFrame(scores)
    scores.to_csv(OUT / "fold_scores.csv", index=False)
    forward = pd.concat(predictions, ignore_index=True)
    forward.to_parquet(OUT / "forward_predictions.parquet", index=False)
    uncertainty = []
    for label, base_col, slow_col in (("standard", "base_prediction", "slow_prediction"),
                                      ("compact", "compact_base_prediction", "compact_slow_prediction")):
        delta = ((forward.actual_normalized_alpha24 - forward[base_col]) ** 2 -
                 (forward.actual_normalized_alpha24 - forward[slow_col]) ** 2)
        day = pd.DataFrame({"day": forward.time.dt.floor("D"), "gain": delta}).groupby("day").gain.mean().to_numpy()
        rng = np.random.default_rng(20260930)
        starts = rng.integers(0, len(day), (1000, int(np.ceil(len(day) / 14))))
        draws = day[((starts[:, :, None] + np.arange(14)) % len(day)).reshape(1000, -1)[:, :len(day)]].mean(axis=1)
        lo, hi = np.quantile(draws, [.025, .975])
        uncertainty.append({"capacity": label, "days": len(day), "daily_mean_mse_gain": day.mean(),
                            "block14_low": lo, "block14_high": hi,
                            "positive_fold_count": int((scores[scores.model == ("calm_down_x2" if label == "standard" else "calm_down_x2_compact")]
                                .mse_normalized_alpha24.to_numpy() -
                                scores[scores.model == ("calm_down_x2_slow" if label == "standard" else "calm_down_x2_slow_compact")]
                                .mse_normalized_alpha24.to_numpy() > 0).sum())})
    pd.DataFrame(uncertainty).to_csv(OUT / "slow_increment_uncertainty.csv", index=False)
    registry = [
        ("relative_7d", "coin 168h log return minus known beta times mean-market 168h log return, divided by sqrt(known rv*7)", "168h", "dimensionless", "signed relative trend", "X2 trend/path", "missing until 168h history; whole slow panel starts after 720h"),
        ("relative_30d", "coin 720h log return minus known beta times mean-market 720h log return, divided by sqrt(known rv*30)", "720h", "dimensionless", "signed relative trend", "X2 trend/path and 7d relative", "missing until 720h history"),
        ("signed_efficiency_7d", "coin 168h net log return / sum absolute completed hourly log returns", "168h", "ratio [-1,1]", "persistent signed path", "relative_7d and X2 trend", "missing until 168h history"),
        ("market_trend_7d", "mean-market 168h log return / sqrt(mean known rv*7)", "168h", "dimensionless", "common signed trend", "market_trend_30d", "missing until 168h history"),
        ("market_trend_30d", "mean-market 720h log return / sqrt(mean known rv*30)", "720h", "dimensionless", "common signed trend", "market_trend_7d", "missing until 720h history"),
        ("rv_ratio_7d_30d", "log(mean known 24h rv over 168h / mean known 24h rv over 720h)", "168h/720h", "log ratio", "risk expansion positive", "X2 risk/path", "missing until 720h history"),
    ]
    pd.DataFrame(registry, columns=["feature", "formula", "window", "unit", "direction_expectation",
                                    "possible_duplicate", "missing_behavior"]).assign(
        available_at="HH:00 UTC after completed hourly bar; all inputs are past or completed").to_csv(
            OUT / "feature_registry.csv", index=False)
    compare = scores.pivot(index="fold", columns="model", values="mse_normalized_alpha24")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    gains = pd.DataFrame({"standard": compare.calm_down_x2 - compare.calm_down_x2_slow,
                          "compact": compare.calm_down_x2_compact - compare.calm_down_x2_slow_compact})
    gains.plot.bar(ax=axes[0], color=["#267b8e", "#ba713e"])
    axes[0].axhline(0, color="black", linewidth=.8)
    axes[0].set(title="Calm-down slow input: positive MSE gain", xlabel="Forward fold", ylabel="Base minus slow MSE")
    scores[scores.model.str.startswith("calm_down")].pivot(
        index="fold", columns="model", values="raw_alpha24_corr").plot(ax=axes[1], marker="o")
    axes[1].axhline(0, color="black", linewidth=.8)
    axes[1].set(title="Calm-down direction correlation by fold", xlabel="Forward fold")
    fig.tight_layout(); fig.savefig(OUT / "calm_down_slow_test.png", dpi=170); plt.close(fig)
    print("saved slow-scale conditional comparison", OUT)


if __name__ == "__main__":
    main()
