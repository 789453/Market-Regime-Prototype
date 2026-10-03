"""Reproducible geometry, horizon, symbol and event-path diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/crypto"
FIG = REPORT / "figs"
CFG = yaml.safe_load((ROOT / "configs/crypto_research.yaml").read_text(encoding="utf-8"))


def horizon_outcomes(events: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for symbol, rows in events.groupby("symbol", sort=False):
        raw = pq.read_table(Path(CFG["data"]["root"]) / symbol / "5m.parquet",
                            columns=["open_time", "open"], filters=[("open_time", "<", 1767225600000)]).to_pandas()
        t = raw.open_time.to_numpy(dtype=np.int64)
        price = np.log(raw.open.to_numpy(dtype=np.float64))
        sub = rows.copy()
        entry_time = sub.available_at.astype("int64").to_numpy() // 1_000_000 + 300_000
        entry = np.searchsorted(t, entry_time)
        if not np.all((entry < len(t)) & (t[np.minimum(entry, len(t) - 1)] == entry_time)):
            raise AssertionError(f"event execution is not a known 5m open: {symbol}")
        for horizon, bars in ((4, 48), (12, 144), (24, 288)):
            valid = entry + bars < len(price)
            outcome = np.full(len(sub), np.nan)
            outcome[valid] = price[entry[valid] + bars] - price[entry[valid]]
            if horizon == 4 and not np.allclose(outcome[valid], sub.future_logret.to_numpy()[valid], atol=1e-10):
                raise AssertionError("4h event label mismatches next 5m execution open")
            sub[f"signed_{horizon}h_bp"] = outcome * sub.direction.to_numpy() * 10000
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def block_interval(values: pd.DataFrame, column: str) -> tuple[float, float]:
    period = values[["available_at", column]].dropna()
    if period.empty:
        return np.nan, np.nan
    days = pd.date_range("2024-01-01", "2025-12-31", freq="D", tz="UTC")
    dates = period.available_at.dt.floor("D")
    summed = period.groupby(dates)[column].sum().reindex(days, fill_value=0).to_numpy()
    count = period.groupby(dates)[column].size().reindex(days, fill_value=0).to_numpy()
    rng = np.random.default_rng(20260928)
    starts = rng.integers(0, len(days), size=(1000, int(np.ceil(len(days) / 7))))
    pick = ((starts[:, :, None] + np.arange(7)) % len(days)).reshape(1000, -1)[:, :len(days)]
    means = np.divide(summed[pick].sum(axis=1), count[pick].sum(axis=1),
                      out=np.full(1000, np.nan), where=count[pick].sum(axis=1) > 0)
    return tuple(np.nanquantile(means, [.025, .975]).astype(float))


def plot_event_path(events: pd.DataFrame, prototype: str) -> None:
    selected = events.loc[(events.method == "semantic_gated_full") & (events.prototype == prototype)
                          & (events.symbol == "BTCUSDT") & (events.available_at.dt.year == 2024)].sort_values("available_at")
    if selected.empty:
        return
    # The first qualifying event is chosen without looking at its outcome.
    event_time = selected.available_at.iloc[0]
    source = Path(CFG["data"]["root"]) / "BTCUSDT" / "5m.parquet"
    raw = pq.read_table(source, columns=["open_time", "open"],
                        filters=[("open_time", ">=", int((event_time - pd.Timedelta(hours=8)).timestamp() * 1000)),
                                 ("open_time", "<", int((event_time + pd.Timedelta(hours=8)).timestamp() * 1000))]).to_pandas()
    raw["time"] = pd.to_datetime(raw.open_time, unit="ms", utc=True)
    source_features = ROOT / "data/crypto/features/BTCUSDT.parquet"
    fields = pq.read_table(source_features, columns=["available_at", "rv_fast_slow_15m", "momentum_z_med_15m", "flow_imb_med_15m"],
                           filters=[("available_at", ">=", event_time - pd.Timedelta(hours=8)),
                                    ("available_at", "<", event_time + pd.Timedelta(hours=8))]).to_pandas()
    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(raw.time, raw.open, color="#222222")
    axes[0].set_ylabel("BTC 5m open")
    for ax, col, label in zip(axes[1:], ("rv_fast_slow_15m", "momentum_z_med_15m", "flow_imb_med_15m"),
                              ("RV fast/slow", "15m trend z", "15m taker flow")):
        ax.plot(fields.available_at, fields[col])
        ax.set_ylabel(label)
        ax.axhline(0, color="gray", lw=.5)
    for ax in axes:
        ax.axvline(event_time, color="#d45525", linestyle="--", lw=1)
        ax.grid(alpha=.2)
    fig.suptitle(f"First BTC {prototype} match in 2024: {event_time} (selection ignores outcome)")
    fig.tight_layout()
    fig.savefig(FIG / f"full_pattern_first_{prototype}_path.png", dpi=160)
    plt.close(fig)


def plot_signature(signature: pd.DataFrame) -> None:
    grouped = signature.groupby(["prototype", "group", "feature"], as_index=False).agg(
        center=("center_percentile", "mean"), center_sd=("center_percentile", "std"),
        weight=("effective_feature_weight", "mean"))
    grouped["salience"] = grouped.weight * np.abs(grouped.center - .5)
    grouped.to_csv(REPORT / "full_pattern_feature_summary.csv", index=False)
    samples = ("release_long", "rejection_long", "range_short")
    fig, axes = plt.subplots(1, 3, figsize=(17, 7))
    for ax, prototype in zip(axes, samples):
        top = grouped.loc[grouped.prototype == prototype].nlargest(14, "salience").sort_values("salience")
        ax.barh(top.feature, top.center - .5, color=np.where(top.center >= .5, "#ca6d3e", "#477ca9"))
        ax.set_xlim(-.5, .5)
        ax.axvline(0, color="gray", lw=.7)
        ax.set_title(prototype)
        ax.set_xlabel("prototype percentile minus training median")
    fig.suptitle("Full-field prototype signatures: most contrasted coordinates across folds")
    fig.tight_layout()
    fig.savefig(FIG / "full_pattern_signatures.png", dpi=160)
    plt.close(fig)


def plot_plateau(cards: pd.DataFrame) -> None:
    methods = ("semantic_seed", "full_dynamic", "semantic_gated_full")
    prototypes = ("release_long", "rejection_long", "range_short")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, prototype in zip(axes, prototypes):
        data = cards.loc[(cards.prototype == prototype) & cards.method.isin(methods)]
        for method, group in data.groupby("method"):
            points = [(q, np.average(rows.mean_signed_4h_bp, weights=rows.events))
                      for q, rows in group.groupby("quantile")]
            ax.plot([q for q, _ in points], [value for _, value in points], marker="o", label=method)
        ax.axhline(0, color="gray", lw=.7)
        ax.set_title(prototype)
        ax.set_xlabel("training similarity quantile")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("signed next-4h mean, bp")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "full_pattern_plateau.png", dpi=160)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    events = pd.read_parquet(REPORT / "full_pattern_events.parquet")
    assert events.available_at.max() < pd.Timestamp("2026-01-01", tz="UTC")
    event_horizons = horizon_outcomes(events)
    event_horizons.to_parquet(REPORT / "full_pattern_event_horizons.parquet", index=False, compression="zstd")
    horizon_rows = []
    for (method, proto, year), group in event_horizons.groupby(["method", "prototype", event_horizons.available_at.dt.year]):
        for horizon in (4, 12, 24):
            values = group[f"signed_{horizon}h_bp"].dropna()
            horizon_rows.append({"method": method, "prototype": proto, "year": year,
                                 "horizon_hours": horizon, "events": len(values),
                                 "mean_signed_bp": float(values.mean()),
                                 "median_signed_bp": float(values.median()),
                                 "positive_fraction": float((values > 0).mean())})
    pd.DataFrame(horizon_rows).to_csv(REPORT / "full_pattern_horizon_cards.csv", index=False)
    symbols = event_horizons.groupby(["method", "prototype", "symbol"]).agg(
        events=("signed_4h_bp", "size"), mean_signed_4h_bp=("signed_4h_bp", "mean")).reset_index()
    symbols.to_csv(REPORT / "full_pattern_symbol_cards.csv", index=False)
    signature = pd.read_csv(REPORT / "full_pattern_feature_signatures.csv")
    plot_signature(signature)
    cards = pd.read_csv(REPORT / "full_pattern_fold_cards.csv")
    plot_plateau(cards)
    for prototype in ("release_long", "rejection_long"):
        plot_event_path(events, prototype)
    positions = pd.read_parquet(REPORT / "full_pattern_positions.parquet")
    conflicts = pd.DataFrame({"method": [c.removesuffix("_conflict") for c in positions if c.endswith("_conflict")],
                              "fraction": [float(positions[c].mean()) for c in positions if c.endswith("_conflict")]})
    conflicts.to_csv(REPORT / "full_pattern_conflicts.csv", index=False)
    print("HORIZONS, dynamic selected candidates")
    h = pd.DataFrame(horizon_rows)
    print(h.loc[(h.method == "full_dynamic") & h.prototype.isin(("release_long", "rejection_long", "range_short"))].to_string(index=False))
    print("CONFLICTS")
    print(conflicts.to_string(index=False))


if __name__ == "__main__":
    main()
