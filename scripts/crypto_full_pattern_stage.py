"""Integrated 112-field prototype, baseline and executable lifecycle research.

2023-25 are development only. Every 90-day test fold refits from the preceding
365 days; 2026 data are excluded before feature/outcome loading. This script
reports candidates and model-design comparisons, not certified alpha.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.crypto.full_patterns import (
    BASE_COLUMNS, GROUP_COLUMNS, GROUPS, MODEL_COLUMNS, SEEDS,
    EmpiricalMap, add_path, anchor_score, combine_distances, cosine_scores,
    dynamic_weights, fit_prototype, group_distances,
)
from crypto_event_research import load_symbol


CFG = yaml.safe_load((ROOT / "configs/crypto_research.yaml").read_text(encoding="utf-8"))
FEATURE_DIR = ROOT / "data/crypto/features"
REPORT = ROOT / "reports/crypto"
FIG = REPORT / "figs"
QUANTILES = (.99, .995, .9975)
METHODS = ("semantic_seed", "full_equal", "full_redundancy", "full_dynamic", "semantic_gated_full", "full_cosine")
FOLD_STARTS = list(pd.date_range("2024-01-01", "2026-01-01", freq="90D", tz="UTC"))
if FOLD_STARTS[-1] < pd.Timestamp("2026-01-01", tz="UTC"):
    FOLD_STARTS.append(pd.Timestamp("2026-01-01", tz="UTC"))
FOLD_END = pd.Timestamp("2026-01-01", tz="UTC")


def load_panel() -> pd.DataFrame:
    parts = []
    for symbol in CFG["data"]["symbols"]:
        labels = load_symbol(symbol)[["symbol", "available_at", "execution_at", "label_available_at", "execution_price", "future_logret", "future_up_var", "future_down_var"]]
        fields = pq.read_table(
            FEATURE_DIR / f"{symbol}.parquet", columns=["available_at", *BASE_COLUMNS],
            filters=[("available_at", "<", pd.Timestamp("2026-01-01", tz="UTC"))],
        ).to_pandas()
        merged = labels.merge(fields, on="available_at", validate="one_to_one", how="inner")
        parts.append(merged)
        print(f"loaded {symbol}: {len(merged):,} rows", flush=True)
    panel = add_path(pd.concat(parts, ignore_index=True))
    panel["next_execution_logret"] = panel.groupby("symbol", sort=False).execution_price.transform(
        lambda x: np.log(x.shift(-1) / x)
    )
    panel["symbol_code"] = pd.Categorical(panel.symbol, categories=CFG["data"]["symbols"]).codes.astype(np.int8)
    return panel.reset_index(drop=True)


def thin_indices(frame: pd.DataFrame, mask: np.ndarray, *, gap_hours: int = 4) -> np.ndarray:
    positions = np.flatnonzero(mask)
    if not len(positions):
        return positions
    sym = frame.symbol_code.to_numpy()[positions]
    times = frame.available_at.astype("int64").to_numpy()[positions]
    last = np.full(len(CFG["data"]["symbols"]), np.iinfo(np.int64).min // 2, dtype=np.int64)
    keep = np.zeros(len(positions), dtype=bool)
    minimum = gap_hours * 3_600_000_000_000
    for j, (s, t) in enumerate(zip(sym, times)):
        if t - last[s] >= minimum:
            keep[j] = True
            last[s] = t
    return positions[keep]


def evidence(signed_returns: np.ndarray, dates: pd.Series) -> tuple[float, float, float, int]:
    if not len(signed_returns):
        return 0.0, 0.0, 0.0, 0
    outcomes = pd.Series(signed_returns)
    by_day = outcomes.groupby(dates.dt.floor("D").reset_index(drop=True)).mean().to_numpy()
    sem = float(np.std(by_day, ddof=1) / np.sqrt(len(by_day))) if len(by_day) > 1 else float("inf")
    mean = float(np.mean(signed_returns))
    event_time = dates.astype("int64").to_numpy()
    midpoint = np.median(event_time)
    recent = float(np.mean(signed_returns[event_time >= midpoint]))
    # Low-capacity soft reliability; negative training evidence cannot be inverted.
    reliability = max(0.0, mean) / (abs(mean) + sem + .0005)
    reliability *= min(1.0, len(signed_returns) / 150)
    if recent <= 0:
        reliability *= .2
    return mean, sem, reliability, len(by_day)


def positions_from_candidates(scores: np.ndarray, enter: np.ndarray, exit_: np.ndarray,
                              reliability: np.ndarray, directions: np.ndarray,
                              frame: pd.DataFrame, *, hysteresis: bool) -> tuple[np.ndarray, np.ndarray]:
    """Resolve opposite prototypes and create four-hour or hysteretic exposure."""
    scaled = np.maximum(scores - enter, 0) / np.maximum(1 - enter, .02)
    scaled *= reliability[None, :]
    scaled = np.minimum(scaled, 1)
    long = np.where(directions[None, :] > 0, scaled, 0)
    short = np.where(directions[None, :] < 0, scaled, 0)
    long_best = long.max(axis=1)
    short_best = short.max(axis=1)
    long_idx = long.argmax(axis=1)
    short_idx = short.argmax(axis=1)
    candidate = np.where(long_best > short_best * 1.25, 1, np.where(short_best > long_best * 1.25, -1, 0)).astype(np.int8)
    selected = np.where(candidate > 0, long_idx, np.where(candidate < 0, short_idx, -1)).astype(np.int8)
    conflicting = (long_best > 0) & (short_best > 0) & (candidate == 0)
    positions = np.zeros(len(frame), dtype=np.float32)
    symbols = frame.symbol_code.to_numpy()
    for symbol in np.unique(symbols):
        idx = np.flatnonzero(symbols == symbol)
        current, age, active_proto = 0, 0, -1
        for row in idx:
            signal = int(candidate[row])
            if not hysteresis:
                if age >= 16:
                    current, age, active_proto = 0, 0, -1
                if current == 0 and signal and selected[row] >= 0:
                    current, active_proto, age = signal, int(selected[row]), 0
                positions[row] = current
                if current:
                    age += 1
                continue
            if current:
                age += 1
                expired = age >= 32
                faded = age >= 4 and scores[row, active_proto] < exit_[active_proto]
                opposed = signal == -current and (long_best[row] + short_best[row]) > 0
                if expired or faded or opposed:
                    current, age, active_proto = 0, 0, -1
            if current == 0 and signal and selected[row] >= 0:
                current, active_proto, age = signal, int(selected[row]), 0
            positions[row] = current
    return positions, conflicting


def fold_shift_null(frame: pd.DataFrame, positions: np.ndarray, direction: int, *, seed: int) -> np.ndarray:
    """Shift the complete cross-symbol event clock by common whole days."""
    if len(positions) == 0:
        return np.full(200, np.nan)
    rng = np.random.default_rng(seed)
    sym = frame.symbol_code.to_numpy()
    outcome = frame.future_logret.to_numpy(dtype=np.float64)
    within = frame.groupby("symbol_code", sort=False).cumcount().to_numpy()
    sizes = np.bincount(sym, minlength=len(CFG["data"]["symbols"]))
    n_days = max(2, int(min(sizes) // 96))
    shift_days = rng.integers(1, n_days, size=200)
    sums = np.zeros(200)
    count = 0
    for s in np.unique(sym[positions]):
        mask = positions[sym[positions] == s]
        values = outcome[sym == s]
        local = within[mask]
        shifted = values[(local[None, :] + shift_days[:, None] * 96) % len(values)]
        sums += np.nansum(shifted * direction, axis=1)
        count += len(mask)
    return sums / count


def run_fold(panel: pd.DataFrame, fold: int, start: pd.Timestamp, stop: pd.Timestamp):
    eligible = panel.loc[(panel.available_at >= start - pd.Timedelta(days=365))
                         & (panel.label_available_at < start)]
    # Random, outcome-independent thinning covers every 15m clock phase and
    # symbol; taking every eighth row aliases the same hours repeatedly.
    rng = np.random.default_rng(20260928 + fold)
    take = np.sort(rng.choice(len(eligible), size=max(1, len(eligible) // 8), replace=False))
    train = eligible.iloc[take].copy()
    test = panel.loc[(panel.available_at >= start) & (panel.available_at < stop)].copy()
    if train.empty or test.empty:
        return [], [], [], {}, []
    ref = EmpiricalMap(train, MODEL_COLUMNS)
    x_train, miss_train = ref.transform(train)
    x_test, miss_test = ref.transform(test)
    redund_sample = x_train[::max(1, len(x_train) // 5000)]
    direction = np.array([p.direction for p in SEEDS])
    train_outcome = train.future_logret.to_numpy(dtype=np.float64)
    date_array = train.available_at.dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    score_train = {method: np.empty((len(train), len(SEEDS)), dtype=np.float32) for method in METHODS}
    score_test = {method: np.empty((len(test), len(SEEDS)), dtype=np.float32) for method in METHODS}
    geometry = []
    feature_signature = []
    for k, seed in enumerate(SEEDS):
        proto = fit_prototype(x_train, seed, redundancy_sample=redund_sample)
        d_train_equal = group_distances(x_train, miss_train, proto, use_redundancy=False)
        d_test_equal = group_distances(x_test, miss_test, proto, use_redundancy=False)
        d_train_redund = group_distances(x_train, miss_train, proto, use_redundancy=True)
        d_test_redund = group_distances(x_test, miss_test, proto, use_redundancy=True)
        w_dynamic = dynamic_weights(d_train_redund, seed.direction * train_outcome, date_array, proto.group_prior)
        score_train["semantic_seed"][:, k] = anchor_score(x_train, seed)
        score_test["semantic_seed"][:, k] = anchor_score(x_test, seed)
        equal = np.ones(len(GROUPS), dtype=np.float32)
        score_train["full_equal"][:, k] = combine_distances(d_train_equal, equal)
        score_test["full_equal"][:, k] = combine_distances(d_test_equal, equal)
        score_train["full_redundancy"][:, k] = combine_distances(d_train_redund, proto.group_prior)
        score_test["full_redundancy"][:, k] = combine_distances(d_test_redund, proto.group_prior)
        score_train["full_dynamic"][:, k] = combine_distances(d_train_redund, w_dynamic)
        score_test["full_dynamic"][:, k] = combine_distances(d_test_redund, w_dynamic)
        score_train["full_cosine"][:, k] = cosine_scores(x_train, proto.center)
        score_test["full_cosine"][:, k] = cosine_scores(x_test, proto.center)
        for g, group in enumerate(GROUPS):
            names = GROUP_COLUMNS[group]
            geometry.append({"fold": fold, "start": start, "prototype": seed.name, "group": group,
                             "prior_weight": float(proto.group_prior[g]), "dynamic_weight": float(w_dynamic[g]),
                             "mean_contrast": float(np.mean(proto.contrast[[MODEL_COLUMNS.index(n) for n in names]])),
                             "mean_redundancy": float(np.mean(proto.redundancy[[MODEL_COLUMNS.index(n) for n in names]])),
                             "anchor_count": proto.anchor_count})
        for j, name in enumerate(MODEL_COLUMNS):
            feature_signature.append({"fold": fold, "start": start, "prototype": seed.name,
                                      "group": next(g for g, names in GROUP_COLUMNS.items() if name in names),
                                      "feature": name, "center_percentile": float(proto.center[j]),
                                      "half_width": float(proto.width[j]),
                                      "contrast_weight": float(proto.contrast[j]),
                                      "redundancy_weight": float(proto.redundancy[j]),
                                      "effective_feature_weight": float(proto.contrast[j] * proto.redundancy[j])})
    # Mechanism evidence is a core condition; the 136-field geometry only
    # refines matches. This prevents weak semantic matches from being rescued
    # by many neutral dimensions in a high-dimensional center.
    score_train["semantic_gated_full"] = score_train["full_dynamic"] * np.exp(4 * score_train["semantic_seed"])
    score_test["semantic_gated_full"] = score_test["full_dynamic"] * np.exp(4 * score_test["semantic_seed"])
    del x_train, x_test, miss_train, miss_test

    rows, saved_events, train_cards = [], [], []
    null_draws = defaultdict(list)
    position_output = {method: {} for method in METHODS}
    for method in METHODS:
        enter = np.quantile(score_train[method], .995, axis=0)
        exit_ = np.quantile(score_train[method], .99, axis=0)
        reliability = np.zeros(len(SEEDS), dtype=np.float32)
        for k, seed in enumerate(SEEDS):
            train_idx = thin_indices(train, score_train[method][:, k] >= enter[k])
            signed = seed.direction * train_outcome[train_idx]
            mean, sem, rel, days = evidence(signed, train.available_at.iloc[train_idx])
            reliability[k] = rel
            train_cards.append({"fold": fold, "start": start, "method": method, "prototype": seed.name,
                                "train_events": len(train_idx), "train_days": days, "train_signed_4h_bp": mean * 10000,
                                "train_day_sem_bp": sem * 10000, "reliability": rel, "entry_threshold": float(enter[k]),
                                "exit_threshold": float(exit_[k])})
            for quantile in QUANTILES:
                threshold = float(np.quantile(score_train[method][:, k], quantile))
                idx = thin_indices(test, score_test[method][:, k] >= threshold)
                ret = seed.direction * test.future_logret.to_numpy()[idx] * 10000
                rows.append({"fold": fold, "start": start, "stop": stop, "year": start.year,
                             "method": method, "prototype": seed.name, "family": seed.family,
                             "direction": seed.direction, "quantile": quantile, "threshold": threshold,
                             "events": len(idx), "days": test.available_at.iloc[idx].dt.floor("D").nunique(),
                             "mean_signed_4h_bp": float(np.mean(ret)) if len(idx) else np.nan,
                             "median_signed_4h_bp": float(np.median(ret)) if len(idx) else np.nan,
                             "mean_up_semivar": float(test.future_up_var.iloc[idx].mean()) if len(idx) else np.nan,
                             "mean_down_semivar": float(test.future_down_var.iloc[idx].mean()) if len(idx) else np.nan})
                if quantile == .995:
                    e = test.iloc[idx][["symbol", "available_at", "future_logret"]].copy()
                    e["method"], e["prototype"], e["direction"], e["fold"] = method, seed.name, seed.direction, fold
                    e["similarity"] = score_test[method][idx, k]
                    saved_events.append(e)
                    null_draws[(method, seed.name)].append((float(np.nansum(ret / 10000)), len(idx),
                                                              fold_shift_null(test, idx, seed.direction, seed=fold * 1000 + k)))
        pos, conflict = positions_from_candidates(score_test[method], enter, exit_, reliability, direction, test, hysteresis=False)
        position_output[method]["fixed4h"] = pos
        position_output[method]["conflict"] = conflict
        if method == "semantic_gated_full":
            # Library-capacity diagnostic: the selected prototype identities
            # are fixed from each fold's training evidence, never test labels.
            for top_n in (1, 2):
                selected_reliability = np.zeros_like(reliability)
                best = np.argsort(reliability)[-top_n:]
                selected_reliability[best] = reliability[best]
                selected_pos, _ = positions_from_candidates(
                    score_test[method], enter, exit_, selected_reliability, direction, test, hysteresis=False)
                position_output[method][f"train_top{top_n}_fixed4h"] = selected_pos
        if method == "full_dynamic":
            pos_h, _ = positions_from_candidates(score_test[method], enter, exit_, reliability, direction, test, hysteresis=True)
            position_output[method]["hysteresis"] = pos_h
            sparse_enter = np.quantile(score_train[method], .9975, axis=0)
            sparse_rel = np.zeros(len(SEEDS), dtype=np.float32)
            for k, seed in enumerate(SEEDS):
                sparse_idx = thin_indices(train, score_train[method][:, k] >= sparse_enter[k])
                _, _, sparse_rel[k], _ = evidence(seed.direction * train_outcome[sparse_idx], train.available_at.iloc[sparse_idx])
            sparse_pos, _ = positions_from_candidates(score_test[method], sparse_enter, enter, sparse_rel, direction, test, hysteresis=False)
            position_output[method]["sparse_fixed4h"] = sparse_pos
    positions = test[["symbol", "available_at", "execution_price", "next_execution_logret"]].copy()
    positions["fold"] = fold
    for method, variants in position_output.items():
        positions[f"{method}_fixed4h"] = variants["fixed4h"]
        if "hysteresis" in variants:
            positions[f"{method}_hysteresis"] = variants["hysteresis"]
        if "sparse_fixed4h" in variants:
            positions[f"{method}_sparse_fixed4h"] = variants["sparse_fixed4h"]
        for top_n in (1, 2):
            key = f"train_top{top_n}_fixed4h"
            if key in variants:
                positions[f"{method}_{key}"] = variants[key]
        positions[f"{method}_conflict"] = variants["conflict"]
    # A separate, bounded sizing diagnostic: reduce exposure when current
    # realized volatility exceeds the past training-window median. It never
    # increases leverage and has no access to the test-period return label.
    historical_rv = float(np.nanmedian(train.rv_slow_15m.to_numpy(dtype=float)))
    current_rv = test.rv_slow_15m.to_numpy(dtype=float)
    factor = np.sqrt(historical_rv / np.maximum(current_rv, 1e-12))
    factor = np.clip(np.nan_to_num(factor, nan=0.0, posinf=0.0), .25, 1.0)
    positions["full_dynamic_vol_scaled_fixed4h"] = position_output["full_dynamic"]["fixed4h"] * factor
    print(f"fold {fold}: {start.date()} -> {stop.date()}, train={len(train):,}, test={len(test):,}", flush=True)
    return rows, saved_events, train_cards, null_draws, [positions, pd.DataFrame(geometry), pd.DataFrame(feature_signature)]


def daily_strategy(position_frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.concat(position_frames, ignore_index=True).sort_values(["symbol", "available_at"]).reset_index(drop=True)
    panel = panel.loc[(panel.available_at >= "2024-01-01") & (panel.available_at < "2026-01-01")]
    dates = panel.available_at.dt.floor("D")
    strategies = [c for c in panel if c.endswith("_fixed4h") or c.endswith("_hysteresis")]
    rows = []
    summaries = []
    for strategy in strategies:
        pos = panel[strategy].to_numpy(dtype=float)
        previous = panel.groupby("symbol", sort=False)[strategy].shift(1).fillna(0).to_numpy(dtype=float)
        turnover = np.abs(pos - previous)
        gross = np.nan_to_num(pos * panel.next_execution_logret.to_numpy(dtype=float), nan=0)
        for cost in (0, 4, 10):
            net = gross - turnover * cost / 10000
            by_date = pd.DataFrame({"date": dates, "symbol": panel.symbol, "net": net, "gross": gross,
                                    "turnover": turnover, "active": np.abs(pos)}).groupby(["date", "symbol"], sort=False).sum()
            daily = by_date.groupby("date").mean()
            for date, row in daily.iterrows():
                rows.append({"date": date, "strategy": strategy, "cost_per_side_bp": cost,
                             "daily_logreturn": row.net, "daily_gross_logreturn": row.gross,
                             "mean_symbol_turnover": row.turnover, "mean_active_bars": row.active})
            summaries.append({"strategy": strategy, "cost_per_side_bp": cost,
                              "mean_daily_bp": float(daily.net.mean() * 10000),
                              "cumulative_return": float(np.exp(daily.net.sum()) - 1),
                              "mean_daily_turnover": float(daily.turnover.mean()),
                              "active_fraction": float(np.mean(np.abs(pos) > 0))})
    # External benchmark: 12 initial equal cash allocations, no rebalancing.
    first = panel.groupby("symbol", sort=False).execution_price.first()
    end_prices = panel.groupby([dates, "symbol"], sort=False).execution_price.last().unstack("symbol").ffill()
    wealth = (end_prices / first).mean(axis=1)
    start_wealth = 1.0
    prior = wealth.shift(1).fillna(start_wealth)
    bh_daily = np.log(wealth / prior)
    for date, value in bh_daily.items():
        rows.append({"date": date, "strategy": "buyhold", "cost_per_side_bp": 0,
                     "daily_logreturn": value, "daily_gross_logreturn": value,
                     "mean_symbol_turnover": 0.0, "mean_active_bars": 96.0})
    summaries.append({"strategy": "buyhold", "cost_per_side_bp": 0,
                      "mean_daily_bp": float(bh_daily.mean() * 10000),
                      "cumulative_return": float(wealth.iloc[-1] - 1),
                      "mean_daily_turnover": 0.0, "active_fraction": 1.0})
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def block_ci(events: pd.DataFrame, *, seed: int = 20260928) -> tuple[float, float]:
    if events.empty:
        return np.nan, np.nan
    outcome = events.direction * events.future_logret * 10000
    day = events.available_at.dt.floor("D")
    dates = pd.date_range("2024-01-01", "2025-12-31", freq="D", tz="UTC")
    summed = outcome.groupby(day).sum().reindex(dates, fill_value=0).to_numpy()
    count = outcome.groupby(day).size().reindex(dates, fill_value=0).to_numpy()
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(dates), size=(1000, int(np.ceil(len(dates) / 7))))
    indices = ((starts[:, :, None] + np.arange(7)) % len(dates)).reshape(1000, -1)[:, :len(dates)]
    totals = summed[indices].sum(axis=1)
    counts = count[indices].sum(axis=1)
    means = np.divide(totals, counts, out=np.full(len(counts), np.nan), where=counts > 0)
    return tuple(np.nanquantile(means, [.025, .975]).astype(float))


def plot_results(summary: pd.DataFrame, strategy: pd.DataFrame, geometry: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    top = summary.loc[summary["quantile"] == .995]
    pivot = top.pivot(index="prototype", columns="method", values="mean_signed_4h_bp").reindex([p.name for p in SEEDS])
    fig, ax = plt.subplots(figsize=(10, 5))
    picture = ax.imshow(pivot.to_numpy(), cmap="RdBu_r", vmin=-15, vmax=15, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("2024–25 development folds: signed next-4h mean at 99.5% train threshold")
    fig.colorbar(picture, ax=ax, label="bp per sparse event")
    fig.tight_layout()
    fig.savefig(FIG / "full_pattern_method_heatmap.png", dpi=160)
    plt.close(fig)

    cards = pd.read_csv(REPORT / "full_pattern_fold_cards.csv", parse_dates=["start", "stop"])
    target = cards.loc[(cards.prototype == "release_long") & (cards["quantile"] == .995)]
    fig, ax = plt.subplots(figsize=(9, 4))
    for method, rows in target.groupby("method"):
        ax.plot(rows.start, rows.mean_signed_4h_bp, marker="o", label=method)
    ax.axhline(0, color="gray", linewidth=.8)
    ax.set_xlabel("Forward test fold year and quarter")
    ax.set_ylabel("Signed next-4h mean, bp")
    ax.set_title("Compression-release long: fold stability by representation")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "full_pattern_release_folds.png", dpi=160)
    plt.close(fig)

    weights = geometry.loc[geometry.prototype == "release_long"].groupby("group")[["prior_weight", "dynamic_weight"]].mean()
    weights.plot(kind="bar", figsize=(9, 4), title="Release-long group weights across training folds")
    plt.tight_layout()
    plt.savefig(FIG / "full_pattern_dynamic_weights.png", dpi=160)
    plt.close()

    show = strategy.loc[(strategy.cost_per_side_bp == 0) & strategy.strategy.isin(("full_dynamic_fixed4h", "full_dynamic_hysteresis", "buyhold"))]
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, rows in show.groupby("strategy"):
        ax.plot(rows.date, np.exp(rows.daily_logreturn.cumsum()), label=name)
    ax.set_title("Development-period wealth paths, zero-cost diagnostic")
    ax.set_ylabel("Wealth, start = 1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "full_pattern_strategy_paths.png", dpi=160)
    plt.close(fig)


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    cards, events, training, geometries, signatures, position_frames = [], [], [], [], [], []
    all_null = defaultdict(list)
    for fold, (start, stop) in enumerate(zip(FOLD_STARTS[:-1], FOLD_STARTS[1:])):
        rows, saved, train_cards, nulls, artifacts = run_fold(panel, fold, start, min(stop, FOLD_END))
        cards.extend(rows)
        events.extend(saved)
        training.extend(train_cards)
        for key, values in nulls.items():
            all_null[key].extend(values)
        if artifacts:
            position_frames.append(artifacts[0])
            geometries.append(artifacts[1])
            signatures.append(artifacts[2])
    card_frame = pd.DataFrame(cards)
    event_frame = pd.concat(events, ignore_index=True)
    training_frame = pd.DataFrame(training)
    geometry_frame = pd.concat(geometries, ignore_index=True)
    signature_frame = pd.concat(signatures, ignore_index=True)
    daily, strategy_summary = daily_strategy(position_frames)
    pd.concat(position_frames, ignore_index=True).to_parquet(REPORT / "full_pattern_positions.parquet", index=False, compression="zstd")
    summaries = []
    for (method, prototype, quantile), rows in card_frame.groupby(["method", "prototype", "quantile"]):
        part = event_frame.loc[(event_frame.method == method) & (event_frame.prototype == prototype)] if quantile == .995 else pd.DataFrame()
        count = rows.events.sum()
        weighted = np.average(rows.mean_signed_4h_bp, weights=rows.events) if count else np.nan
        low, high = block_ci(part) if quantile == .995 else (np.nan, np.nan)
        null = all_null.get((method, prototype), []) if quantile == .995 else []
        if null:
            null_sum = sum(n * values for _, n, values in null)
            null_n = sum(n for _, n, _ in null)
            null_means = null_sum / null_n * 10000
            null_mean = float(np.nanmean(null_means))
            null_percentile = float(np.mean(null_means <= weighted))
        else:
            null_mean, null_percentile = np.nan, np.nan
        summaries.append({"method": method, "prototype": prototype, "quantile": quantile,
                          "events": int(count), "mean_signed_4h_bp": weighted,
                          "block7_low_bp": low, "block7_high_bp": high,
                          "shift_null_mean_bp": null_mean, "shift_null_percentile": null_percentile,
                          "shift_excess_bp": weighted - null_mean})
    summary = pd.DataFrame(summaries)
    card_frame.to_csv(REPORT / "full_pattern_fold_cards.csv", index=False)
    summary.to_csv(REPORT / "full_pattern_method_summary.csv", index=False)
    training_frame.to_csv(REPORT / "full_pattern_training_evidence.csv", index=False)
    geometry_frame.to_csv(REPORT / "full_pattern_geometry.csv", index=False)
    signature_frame.to_csv(REPORT / "full_pattern_feature_signatures.csv", index=False)
    event_frame.to_parquet(REPORT / "full_pattern_events.parquet", index=False, compression="zstd")
    daily.to_csv(REPORT / "full_pattern_daily.csv", index=False)
    strategy_summary.to_csv(REPORT / "full_pattern_strategy_summary.csv", index=False)
    plot_results(summary, daily, geometry_frame)
    manifest = {"model_columns": len(MODEL_COLUMNS), "base_columns": len(BASE_COLUMNS),
                "path_columns": len(MODEL_COLUMNS) - len(BASE_COLUMNS), "groups": {g: len(v) for g, v in GROUP_COLUMNS.items()},
                "methods": METHODS, "prototypes": [vars(s) for s in SEEDS], "folds": len(FOLD_STARTS) - 1,
                "source_cutoff": "2026-01-01T00:00:00Z", "training": "trailing 365d sampled every eighth 15m bar; labels available before fold start",
                "test": "next ~90d through 2025-12-31; all symbols same UTC fold",
                "note": "development comparison only; no certified prototype or trade recommendation"}
    (REPORT / "full_pattern_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(summary.loc[summary["quantile"] == .995, ["method", "prototype", "events", "mean_signed_4h_bp", "shift_excess_bp"]].to_string(index=False))
    print(strategy_summary.to_string(index=False))


if __name__ == "__main__":
    main()
