"""First finite-pattern-library experiment: semantic similarity and evidence.

2023 defines percentile geometry and similarity thresholds; 2024 supplies
historical evidence; 2025 is a previously seen development check. 2026 rows
are excluded by the source loader. No prototype is called tradable here.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from crypto_event_research import load_symbol

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.crypto.prototypes import PROTOTYPES, QuantileReference, similarity, sparse_events


CFG = yaml.safe_load((ROOT / "configs/crypto_research.yaml").read_text(encoding="utf-8"))
FEATURE_DIR = ROOT / "data/crypto/features"
REPORT_DIR = ROOT / "reports/crypto"
FIG_DIR = REPORT_DIR / "figs"
EXTRA = ["upper_wick_med_5m", "lower_wick_med_5m"]
PROBABILITIES = (0.99, 0.995, 0.9975, 0.999)


def prepare() -> pd.DataFrame:
    pieces = []
    for symbol in CFG["data"]["symbols"]:
        outcomes = load_symbol(symbol)[["symbol", "available_at", "future_logret", "future_up_var", "future_down_var", "trend_15", "efficiency_15", "vol_intensity", "path", "flow"]]
        extras = pq.read_table(
            FEATURE_DIR / f"{symbol}.parquet", columns=["available_at", *EXTRA],
            filters=[("available_at", "<", pd.Timestamp("2026-01-01", tz="UTC"))],
        ).to_pandas()
        frame = outcomes.merge(extras, on="available_at", how="left", validate="one_to_one")
        frame["trend_change"] = frame.trend_15 - frame.trend_15.shift(16)
        frame["vol_change"] = frame.vol_intensity - frame.vol_intensity.shift(16)
        frame.rename(columns={"upper_wick_med_5m": "upper_wick", "lower_wick_med_5m": "lower_wick"}, inplace=True)
        pieces.append(frame.dropna().reset_index(drop=True))
    return pd.concat(pieces, ignore_index=True)


def day_block_interval(events: pd.DataFrame, *, start: str, stop: str, direction: int) -> tuple[float, float]:
    dates = pd.date_range(start, pd.Timestamp(stop) - pd.Timedelta(days=1), freq="D", tz="UTC")
    outcome = direction * events.future_logret * 10000
    day = events.available_at.dt.floor("D")
    summed = outcome.groupby(day).sum().reindex(dates, fill_value=0).to_numpy()
    counted = outcome.groupby(day).size().reindex(dates, fill_value=0).to_numpy()
    rng = np.random.default_rng(20260928)
    n = len(dates)
    starts = rng.integers(0, n, size=(1000, int(np.ceil(n / 7))))
    indices = ((starts[:, :, None] + np.arange(7)) % n).reshape(1000, -1)[:, :n]
    count = counted[indices].sum(axis=1)
    means = np.divide(summed[indices].sum(axis=1), count, out=np.full(len(count), np.nan), where=count > 0)
    return float(np.nanquantile(means, .025)), float(np.nanquantile(means, .975))


def matched_null(events: pd.DataFrame, period: pd.DataFrame, vol_percentile: pd.Series, direction: int) -> tuple[float, float, float]:
    """Match symbol, calendar month, UTC hour and coarse volatility quintile."""
    if events.empty:
        return np.nan, np.nan, np.nan
    pool = period[["symbol", "available_at", "future_logret"]].copy()
    pool["month"] = pool.available_at.dt.strftime("%Y-%m")
    pool["hour"] = pool.available_at.dt.hour
    pool["vol_bin"] = np.minimum((vol_percentile.loc[pool.index] * 5).astype(int), 4)
    keys = ["symbol", "month", "hour", "vol_bin"]
    groups = {key: pool.index.take(pos).to_numpy() for key, pos in pool.groupby(keys, sort=False).indices.items()}
    fallback = {key: pool.index.take(pos).to_numpy() for key, pos in pool.groupby(keys[:-1], sort=False).indices.items()}
    outcomes = direction * pool.future_logret * 10000
    rng = np.random.default_rng(20260928)
    draws = np.zeros(500)
    for idx in events.index:
        row = pool.loc[idx]
        key = (row.symbol, row.month, row.hour, row.vol_bin)
        candidates = groups.get(key)
        if candidates is None or len(candidates) < 3:
            candidates = fallback[(row.symbol, row.month, row.hour)]
        draws += outcomes.loc[rng.choice(candidates, size=len(draws))].to_numpy()
    draws /= len(events)
    actual = float((direction * events.future_logret * 10000).mean())
    return float(draws.mean()), actual - float(draws.mean()), float(np.mean(draws <= actual))


def draw_map() -> None:
    names = sorted({name for proto in PROTOTYPES for name in proto.coordinates})
    centers = np.full((len(PROTOTYPES), len(names)), np.nan)
    for i, proto in enumerate(PROTOTYPES):
        for j, name in enumerate(names):
            if name in proto.coordinates:
                centers[i, j] = proto.coordinates[name].center
    fig, ax = plt.subplots(figsize=(13, 5))
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#eeeeee")
    plot = ax.imshow(centers, vmin=0, vmax=1, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(names)), names, rotation=55, ha="right")
    ax.set_yticks(range(len(PROTOTYPES)), [p.name for p in PROTOTYPES])
    ax.set_title("Six semantic prototype centers (training percentiles; gray = unused)")
    fig.colorbar(plot, ax=ax, label="reference percentile")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "prototype_centers.png", dpi=160)
    plt.close(fig)


def draw_plateau(cards: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for ax, proto in zip(axes.flat, PROTOTYPES):
        for period, color in (("2024_certificate", "#2469a1"), ("2025_development", "#d47725")):
            part = cards.loc[(cards.prototype == proto.name) & (cards.period == period)]
            ax.plot(part.threshold_percentile, part.mean_signed_4h_bp, marker="o", color=color, label=period)
            for _, row in part.iterrows():
                ax.annotate(str(int(row.events)), (row.threshold_percentile, row.mean_signed_4h_bp), fontsize=7)
        ax.axhline(0, color="gray", linewidth=.8)
        ax.set_title(proto.name, fontsize=9)
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.supxlabel("Similarity threshold percentile fixed from 2023 discovery")
    fig.supylabel("Signed next-4h mean, bp (labels show sparse event count)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "prototype_threshold_plateau.png", dpi=160)
    plt.close(fig)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    panel = prepare()
    columns = sorted({name for proto in PROTOTYPES for name in proto.coordinates})
    discovery = panel.loc[(panel.available_at >= "2023-01-15") & (panel.available_at < "2024-01-01")]
    reference = QuantileReference(discovery, columns)
    mapped = reference.transform(panel)
    periods = {
        "2024_certificate": ("2024-01-01", "2025-01-01"),
        "2025_development": ("2025-01-01", "2026-01-01"),
    }
    cards = []
    trigger_rows = []
    for proto in PROTOTYPES:
        scores = similarity(mapped, proto)
        thresholds = {p: float(np.nanquantile(scores[discovery.index], p)) for p in PROBABILITIES}
        for period, (start, stop) in periods.items():
            period_mask = (panel.available_at >= start) & (panel.available_at < stop)
            period_frame = panel.loc[period_mask]
            for p, threshold in thresholds.items():
                mask = period_mask.to_numpy() & (scores >= threshold)
                events = sparse_events(panel, mask, min_gap_hours=4)
                signed = proto.direction * events.future_logret * 10000
                ci_low, ci_high = day_block_interval(events, start=start, stop=stop, direction=proto.direction) if len(events) else (np.nan, np.nan)
                null_mean, null_excess, null_percentile = matched_null(events, period_frame, mapped.vol_intensity, proto.direction) if p == .995 else (np.nan, np.nan, np.nan)
                cards.append({
                    "prototype": proto.name, "direction": proto.direction,
                    "period": period, "threshold_percentile": p, "threshold_score": threshold,
                    "events": len(events), "distinct_days": events.available_at.dt.floor("D").nunique(),
                    "symbols": events.symbol.nunique(),
                    "mean_signed_4h_bp": float(signed.mean()) if len(events) else np.nan,
                    "median_signed_4h_bp": float(signed.median()) if len(events) else np.nan,
                    "block7_ci_low_bp": ci_low, "block7_ci_high_bp": ci_high,
                    "matched_null_mean_bp": null_mean, "matched_excess_bp": null_excess,
                    "matched_null_percentile": null_percentile,
                    "mean_up_semivar": float(events.future_up_var.mean()) if len(events) else np.nan,
                    "mean_down_semivar": float(events.future_down_var.mean()) if len(events) else np.nan,
                })
                if p == .995 and len(events):
                    trigger_rows.append(events[["symbol", "available_at", "future_logret"]].assign(
                        prototype=proto.name, direction=proto.direction,
                        similarity=scores[events.index], threshold=threshold, period=period,
                    ))
    card_frame = pd.DataFrame(cards)
    card_frame.to_csv(REPORT_DIR / "prototype_similarity_cards.csv", index=False)
    if trigger_rows:
        pd.concat(trigger_rows).to_parquet(REPORT_DIR / "prototype_research_events.parquet", index=False, compression="zstd")
    metadata = {
        "discovery": "2023-01-15 to 2023-12-31: quantile geometry and similarity thresholds only",
        "certificate_period": "2024; historical evidence, not external validation",
        "development_check": "2025; previously viewed by earlier experiments",
        "terminal_period": "2026 untouched",
        "design_status": "handcrafted after earlier 2023-2025 project feedback; not preregistered or independent validation",
        "prototypes": [{"name": p.name, "direction": p.direction, "mechanism": p.mechanism,
                         "coordinates": {name: vars(coord) for name, coord in p.coordinates.items()}} for p in PROTOTYPES],
        "threshold_percentiles": PROBABILITIES,
        "event_refractory_hours": 4,
        "note": "research candidates only; a pointwise matched random control is reported at percentile 0.995, but it does not preserve clustered outcomes and is not a certification test",
    }
    (REPORT_DIR / "prototype_library_v0.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    draw_map()
    draw_plateau(card_frame)
    print(card_frame.loc[card_frame.threshold_percentile == .995, ["prototype", "period", "events", "distinct_days", "mean_signed_4h_bp", "block7_ci_low_bp", "block7_ci_high_bp"]].to_string(index=False))


if __name__ == "__main__":
    main()
