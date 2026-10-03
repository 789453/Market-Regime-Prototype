"""Data-born predictive states: discovery, validation selection, then one test evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.crypto.predictive_states import (TASK_SIZES, distribution, fit_states, log_loss,
                                          make_representation, outcome_labels, predict_states, CUDA_ENABLED)

CFG = yaml.safe_load((ROOT / "configs/crypto_predictive_states.yaml").read_text(encoding="utf-8"))
SCOPE = ROOT / "reports/crypto/predictive_states_v1"
SYMBOLS = yaml.safe_load((ROOT / "configs/crypto_research.yaml").read_text(encoding="utf-8"))["data"]["symbols"]


def load_symbol(symbol: str, columns: list[str]) -> pd.DataFrame:
    cutoff = pd.Timestamp(CFG["test_end"])
    fields = pq.read_table(ROOT / CFG["feature_dir"] / f"{symbol}.parquet",
                           columns=["available_at", *columns], filters=[("available_at", "<", cutoff)]).to_pandas()
    raw = pq.read_table(Path(CFG["source"]) / symbol / "5m.parquet",
                        columns=["open_time", "open"],
                        filters=[("open_time", "<", int(cutoff.timestamp() * 1000))]).to_pandas()
    t = raw.open_time.to_numpy(dtype=np.int64)
    price = np.log(raw.open.to_numpy(dtype=np.float64))
    entry_ms = fields.available_at.astype("int64").to_numpy() // 1_000_000 + CFG["execution_delay_minutes"] * 60_000
    entry = np.searchsorted(t, entry_ms)
    aligned = (entry < len(t)) & (t[np.minimum(entry, len(t) - 1)] == entry_ms)
    if not aligned.all():
        raise AssertionError(f"missing next execution open for {symbol}")
    for h, bars in ((4, 48), (24, 288)):
        valid = entry + bars < len(price)
        ret = np.full(len(entry), np.nan)
        ret[valid] = price[entry[valid] + bars] - price[entry[valid]]
        fields[f"return{h}"] = ret
    future_return = np.diff(price, prepend=price[0]) ** 2
    prefix = np.r_[0, np.cumsum(future_return)]
    valid4 = entry + 48 < len(price)
    rv4 = np.full(len(entry), np.nan)
    rv4[valid4] = prefix[entry[valid4] + 49] - prefix[entry[valid4] + 1]
    fields["future_rv4"] = rv4
    fields["label_available_at"] = pd.to_datetime(entry_ms, unit="ms", utc=True) + pd.Timedelta(hours=24)
    fields["symbol"] = symbol
    fields["symbol_code"] = SYMBOLS.index(symbol)
    return fields


def load_panel() -> pd.DataFrame:
    columns = list(dict.fromkeys(name for group in CFG["representation_columns"].values() for name in group))
    if "rv_slow_15m" not in columns:
        columns.append("rv_slow_15m")
    panel = pd.concat([load_symbol(s, columns) for s in SYMBOLS], ignore_index=True)
    panel["bar_index"] = panel.groupby("symbol_code", sort=False).cumcount()
    return panel


def period_indices(panel: pd.DataFrame) -> dict[str, np.ndarray]:
    time = panel.available_at
    known = panel.label_available_at
    fit_end = pd.Timestamp(CFG["discovery_fit_end"])
    discover_end = pd.Timestamp(CFG["discovery_end"])
    validate_end = pd.Timestamp(CFG["validation_end"])
    test_end = pd.Timestamp(CFG["test_end"])
    finite = np.isfinite(panel[["return4", "return24", "future_rv4", "rv_slow_15m"]]).all(axis=1).to_numpy(copy=True)
    finite &= (panel.rv_slow_15m.to_numpy() > 0) & (panel.bar_index.to_numpy() >= 16)
    def period(start: pd.Timestamp, stop: pd.Timestamp) -> np.ndarray:
        return np.flatnonzero(finite & (time >= start).to_numpy() & (time < stop).to_numpy()
                              & (known < stop).to_numpy())
    return {"fit": period(pd.Timestamp("2023-01-01", tz="UTC"), fit_end),
            "inner": period(fit_end, discover_end),
            "discovery": period(pd.Timestamp("2023-01-01", tz="UTC"), discover_end),
            "validation": period(discover_end, validate_end),
            "test": period(validate_end, test_end)}


def score_row(name: str, period: str, y: np.ndarray, p: list[np.ndarray],
              times: pd.Series, accepted: np.ndarray | None = None, n_states: int = 0) -> dict:
    losses = log_loss(y, p)
    by_day = pd.DataFrame({"day": times.dt.floor("D"), "loss": losses.mean(axis=1)}).groupby("day").loss.mean()
    return {"model": name, "period": period, "rows": len(y), "days": len(by_day),
            "logloss": float(losses.mean()), "logloss_day_equal": float(by_day.mean()),
            "logloss_4h_return": float(losses[:, 0].mean()),
            "logloss_24h_return": float(losses[:, 1].mean()),
            "logloss_4h_vol": float(losses[:, 2].mean()),
            "coverage": float(accepted.mean()) if accepted is not None else 1.0,
            "states": n_states}


def background_predict(model, symbols: np.ndarray) -> list[np.ndarray]:
    out = [np.empty((len(symbols), c), dtype=np.float32) for c in TASK_SIZES]
    for symbol in np.unique(symbols):
        mask = symbols == symbol
        for j in range(3):
            out[j][mask] = model.background[int(symbol)][j]
    return out


def direct_fit(x: np.ndarray, y: np.ndarray, index: np.ndarray) -> list[lgb.LGBMClassifier]:
    models = []
    for task in range(3):
        model = lgb.LGBMClassifier(n_estimators=80, learning_rate=.045, num_leaves=15,
                                   max_depth=6, min_child_samples=500, reg_lambda=10,
                                   colsample_bytree=.8, subsample=.8, subsample_freq=1,
                                   n_jobs=6, verbosity=-1, random_state=CFG["random_seed"] + task)
        model.fit(x[index], y[index, task])
        models.append(model)
    return models


def direct_predict(models: list[lgb.LGBMClassifier], x: np.ndarray) -> list[np.ndarray]:
    return [model.predict_proba(x).astype(np.float32) for model in models]


def paired_block_interval(a: np.ndarray, b: np.ndarray, dates: pd.Series,
                          *, seed: int = 20260929) -> tuple[float, float, float]:
    values = pd.DataFrame({"date": dates.dt.floor("D").to_numpy(), "difference": a - b}).groupby("date").difference.mean().to_numpy()
    rng = np.random.default_rng(seed)
    n = len(values)
    starts = rng.integers(0, n, size=(1000, int(np.ceil(n / 7))))
    blocks = ((starts[:, :, None] + np.arange(7)) % n).reshape(1000, -1)[:, :n]
    draws = values[blocks].mean(axis=1)
    return float(values.mean()), *[float(x) for x in np.quantile(draws, [.025, .975])]


def state_cards(panel: pd.DataFrame, index: np.ndarray, state: np.ndarray, accepted: np.ndarray,
                period: str) -> pd.DataFrame:
    source = panel.iloc[index]
    rows = []
    for leaf in np.unique(state):
        mask = (state == leaf) & accepted
        data = source.loc[mask]
        if data.empty:
            continue
        market = source.groupby("available_at").return4.transform("mean").to_numpy()
        rows.append({"period": period, "leaf": int(leaf), "rows": len(data),
                     "days": int(data.available_at.dt.floor("D").nunique()), "symbols": data.symbol.nunique(),
                     "mean_4h_bp": float(data.return4.mean() * 10000),
                     "mean_24h_bp": float(data.return24.mean() * 10000),
                     "mean_4h_residual_bp": float(np.mean(source.return4.to_numpy()[mask] - market[mask]) * 10000),
                     "negative_4h_fraction": float((data.return4 < 0).mean()),
                     "future_rv4_mean": float(data.future_rv4.mean())})
    return pd.DataFrame(rows)


def state_transitions(panel: pd.DataFrame, index: np.ndarray, state: np.ndarray,
                      accepted: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = panel.iloc[index][["symbol_code", "available_at", "return4"]].copy()
    frame["state"] = np.where(accepted, state, -1)
    frame["previous"] = frame.groupby("symbol_code", sort=False).state.shift(1).fillna(-2).astype(int)
    transition = frame.groupby(["previous", "state"]).size().rename("rows").reset_index()
    new_run = (frame.state != frame.previous).astype(int)
    frame["run"] = new_run.groupby(frame.symbol_code).cumsum()
    frame["age_15m"] = frame.groupby(["symbol_code", "run"]).cumcount() + 1
    frame["age_band"] = pd.cut(frame.age_15m, bins=[0, 1, 4, 16, 96, np.inf], labels=["first", "2-4", "5-16", "17-96", "97+"])
    ages = frame.loc[frame.state >= 0].groupby(["state", "age_band"], observed=True).agg(
        rows=("return4", "size"), mean_4h_bp=("return4", lambda x: float(x.mean() * 10000))).reset_index()
    return transition, ages


def main() -> None:
    SCOPE.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    print(f"loaded {len(panel):,} causal snapshots", flush=True)
    periods = period_indices(panel)
    labels = outcome_labels(panel.return4.to_numpy(), panel.return24.to_numpy(), panel.future_rv4.to_numpy(),
                            panel.rv_slow_15m.to_numpy(), tuple(CFG["return_bin_edges"]))
    symbol = panel.symbol_code.to_numpy(dtype=np.int8)
    rng = np.random.default_rng(CFG["random_seed"])
    sampled = {}
    for name in ("fit", "inner", "discovery"):
        idx = periods[name]
        sampled[name] = np.sort(rng.choice(idx, size=max(1, len(idx) // CFG["outcome_sample_stride"]), replace=False))
    print({key: len(value) for key, value in periods.items()}, flush=True)
    all_scores = []
    state_models = {}
    direct_models = {}
    validation_predictions = {}
    representations = make_representation(panel, CFG)
    for rep in ("X0_current", "X1_endpoints", "X2_ordered_path"):
        x = representations[rep]
        print(f"fitting {rep}: {x.shape[1]} coordinates", flush=True)
        model = fit_states(x, labels, symbol, panel.available_at, sampled["fit"], sampled["inner"],
                           sampled["discovery"], CFG)
        state_models[rep] = model
        v = periods["validation"]
        for split in (False, True):
            pred, ids, distance, accepted = predict_states(model, x[v], symbol[v], split=split)
            name = f"{rep}_{'split' if split else 'root'}"
            all_scores.append(score_row(name, "validation", labels[v], pred,
                                        panel.available_at.iloc[v], accepted, len(np.unique(ids))))
            validation_predictions[name] = (pred, ids, accepted)
        direct = direct_fit(x, labels, sampled["discovery"])
        direct_models[rep] = direct
        dp = direct_predict(direct, x[v])
        all_scores.append(score_row(f"{rep}_direct", "validation", labels[v], dp, panel.available_at.iloc[v]))
        validation_predictions[f"{rep}_direct"] = (dp, None, None)
        print(f"  validation: {[(r['model'], round(r['logloss'], 4)) for r in all_scores[-3:]]}", flush=True)
    # Identity ablation: pooled vs per-symbol training percentile reference, same X2 rule.
    pooled_x = make_representation(panel, CFG, pooled=True)["X2_ordered_path"]
    pooled_model = fit_states(pooled_x, labels, symbol, panel.available_at, sampled["fit"],
                              sampled["inner"], sampled["discovery"], CFG)
    v = periods["validation"]
    pp, pi, _, pa = predict_states(pooled_model, pooled_x[v], symbol[v], split=True)
    all_scores.append(score_row("X2_pooled_split", "validation", labels[v], pp,
                                panel.available_at.iloc[v], pa, len(np.unique(pi))))
    validation_predictions["X2_pooled_split"] = (pp, pi, pa)
    state_models["X2_pooled"] = pooled_model
    del pooled_x
    background = background_predict(state_models["X2_ordered_path"], symbol[v])
    all_scores.append(score_row("symbol_background", "validation", labels[v], background, panel.available_at.iloc[v]))
    validation_predictions["symbol_background"] = (background, None, None)
    score_frame = pd.DataFrame(all_scores)
    score_frame.to_csv(SCOPE / "validation_scores.csv", index=False)
    validation_daily = []
    dates = panel.available_at.iloc[v].dt.floor("D").to_numpy()
    for name, (p, _, _) in validation_predictions.items():
        loss = log_loss(labels[v], p)
        by_day = pd.DataFrame({"date": dates, "overall": loss.mean(axis=1),
                               "return4": loss[:, 0], "return24": loss[:, 1], "vol4": loss[:, 2]}).groupby("date").mean().reset_index()
        by_day["model"] = name
        validation_daily.append(by_day)
    pd.concat(validation_daily, ignore_index=True).to_csv(SCOPE / "validation_daily_loss.csv", index=False)
    candidates = score_frame.loc[score_frame.model.str.endswith(("_root", "_split")) & (score_frame.coverage >= .65)]
    if candidates.empty:
        candidates = score_frame.loc[score_frame.model.str.endswith(("_root", "_split"))]
    winner = str(candidates.sort_values(["logloss_day_equal", "model"]).model.iloc[0])
    representation = winner.removesuffix("_root").removesuffix("_split")
    split = winner.endswith("_split")
    selected_model = state_models[representation]
    selected_x = (make_representation(panel, CFG, pooled=True)["X2_ordered_path"] if representation == "X2_pooled"
                  else representations[representation])
    test = periods["test"]
    prediction, state, distance, accepted = predict_states(selected_model, selected_x[test], symbol[test], split=split)
    direct_rep = "X2_ordered_path" if representation == "X2_pooled" else representation
    direct = direct_predict(direct_models[direct_rep], representations[direct_rep][test])
    base = background_predict(selected_model, symbol[test])
    all_scores.extend((score_row(winner, "test", labels[test], prediction, panel.available_at.iloc[test], accepted, len(np.unique(state))),
                       score_row(f"{direct_rep}_direct", "test", labels[test], direct, panel.available_at.iloc[test]),
                       score_row("symbol_background", "test", labels[test], base, panel.available_at.iloc[test])))
    score_frame = pd.DataFrame(all_scores)
    score_frame.to_csv(SCOPE / "all_scores.csv", index=False)
    test_loss = log_loss(labels[test], prediction).mean(axis=1)
    direct_loss = log_loss(labels[test], direct).mean(axis=1)
    base_loss = log_loss(labels[test], base).mean(axis=1)
    task_losses = {"state": log_loss(labels[test], prediction),
                   "direct": log_loss(labels[test], direct),
                   "background": log_loss(labels[test], base)}
    uncertainty = {"state_minus_direct": paired_block_interval(test_loss, direct_loss, panel.available_at.iloc[test]),
                   "state_minus_background": paired_block_interval(test_loss, base_loss, panel.available_at.iloc[test])}
    cards = []
    for period in ("discovery", "validation", "test"):
        idx = periods[period]
        if period == "validation" and winner in validation_predictions:
            _, s, a = validation_predictions[winner]
        else:
            _, s, _, a = predict_states(selected_model, selected_x[idx], symbol[idx], split=split)
        cards.append(state_cards(panel, idx, s, a, period))
    card_frame = pd.concat(cards, ignore_index=True)
    card_frame.to_csv(SCOPE / "state_cards.csv", index=False)
    transition, age = state_transitions(panel, test, state, accepted)
    transition.to_csv(SCOPE / "test_transitions.csv", index=False)
    age.to_csv(SCOPE / "test_state_age.csv", index=False)
    pd.DataFrame(selected_model.medoids).to_csv(SCOPE / "real_medoids.csv", index=False)
    pd.DataFrame(selected_model.split_evidence).to_csv(SCOPE / "split_evidence.csv", index=False)
    predictions = panel.iloc[test][["symbol", "available_at", "return4", "return24", "future_rv4"]].copy()
    predictions["state"] = state
    predictions["accepted"] = accepted
    predictions["geometry_distance"] = distance
    predictions["p_4h_down"] = prediction[0][:, :2].sum(axis=1)
    predictions["p_4h_up"] = prediction[0][:, 3:].sum(axis=1)
    predictions["p_24h_down"] = prediction[1][:, :2].sum(axis=1)
    predictions["p_high_rv4"] = prediction[2][:, 1]
    predictions["state_logloss"] = test_loss
    predictions["direct_logloss"] = direct_loss
    predictions["background_logloss"] = base_loss
    predictions["past_rv24"] = panel.rv_slow_15m.iloc[test].to_numpy()
    predictions["target_4h_bin"] = labels[test, 0]
    predictions["target_24h_bin"] = labels[test, 1]
    predictions["target_high_rv4"] = labels[test, 2]
    for name, value in task_losses.items():
        for task, column in enumerate(("4h_return", "24h_return", "4h_vol")):
            predictions[f"{name}_{column}_logloss"] = value[:, task]
    for name, value in (("state", prediction), ("direct", direct), ("background", base)):
        for task, column in enumerate(("4h_return", "24h_return", "4h_vol")):
            for category in range(TASK_SIZES[task]):
                predictions[f"{name}_{column}_p{category}"] = value[task][:, category]
    predictions.to_parquet(SCOPE / "test_predictions.parquet", index=False, compression="zstd")
    joblib.dump({"representation": representation, "split": split, "model": selected_model,
                 "direct": direct_models[direct_rep]}, SCOPE / "selected_models.joblib", compress=3)
    manifest = {"version": CFG["version"], "config": CFG, "rows_by_period": {k: len(v) for k, v in periods.items()},
                "winner_validation_only": winner, "test_paired_7d_logloss_difference": uncertainty,
                "geometry_cuda_enabled": CUDA_ENABLED,
                "test_executed_after_selection": True,
                "caution": "2025 was viewed in legacy work; this is a new-algorithm forward study, not pristine historical holdout"}
    (SCOPE / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(10, 5))
    view = score_frame.loc[score_frame.period == "validation"].sort_values("logloss")
    ax.barh(view.model, view.logloss, color="#3f6b93")
    ax.invert_yaxis()
    ax.set_xlabel("Mean distribution log loss; lower is better")
    ax.set_title("Predictive state discovery: validation before test")
    fig.tight_layout()
    fig.savefig(SCOPE / "validation_method_comparison.png", dpi=150)
    plt.close(fig)
    print(f"selected on validation: {winner}", flush=True)
    print(score_frame.loc[score_frame.period == "test"].to_string(index=False), flush=True)
    print("paired 7d differences", uncertainty, flush=True)


if __name__ == "__main__":
    main()
