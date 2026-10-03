"""One-at-a-time registered parameter surfaces and plateau stability."""

from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate_parameter_surface(detector, bars: pd.DataFrame, ctx: pd.DataFrame, labels: pd.DataFrame):
    rows = []
    cache = {}
    base = detector.params.copy()
    for parameter, values in detector.param_grid.items():
        for value in values:
            key = tuple(sorted({**base, parameter: value}.items()))
            if key not in cache:
                candidate = type(detector)(**dict(key))
                signal = candidate.detect(bars, ctx)
                merged = signal.merge(labels, on=["root", "timestamp"], how="left", validate="one_to_one")
                horizon = candidate.default_horizon
                valid = merged.direction.ne(0) & merged[f"label_valid_{horizon}"]
                performance = (merged.loc[valid, "direction"] * merged.loc[valid, f"fwd_ret_z_{horizon}"]).mean()
                cache[key] = (float(performance), int(valid.sum()))
            performance, count = cache[key]
            rows.append({"detector_id": detector.detector_id, "parameter": parameter, "value": value, "performance": performance, "n": count, "is_default": value == base[parameter]})
    one_dimensional = pd.DataFrame(rows)
    ratios = []
    for _, curve in one_dimensional.groupby("parameter", sort=False):
        curve = curve.sort_values("value").reset_index(drop=True)
        if curve["performance"].notna().any():
            best_index = int(curve["performance"].idxmax())
            best = curve.loc[best_index, "performance"]
            neighbors = curve.loc[max(0, best_index - 1) : min(len(curve) - 1, best_index + 1), "performance"].drop(index=best_index, errors="ignore")
            if best > 0 and len(neighbors):
                ratios.append(float(neighbors.median() / best))
    parameters = list(detector.param_grid)[:2]
    surface_rows = []
    if len(parameters) == 2:
        first, second = parameters
        for first_value in detector.param_grid[first]:
            for second_value in detector.param_grid[second]:
                candidate_params = {**base, first: first_value, second: second_value}
                key = tuple(sorted(candidate_params.items()))
                if key not in cache:
                    candidate = type(detector)(**candidate_params)
                    signal = candidate.detect(bars, ctx)
                    merged = signal.merge(labels, on=["root", "timestamp"], how="left", validate="one_to_one")
                    horizon = candidate.default_horizon
                    valid = merged.direction.ne(0) & merged[f"label_valid_{horizon}"]
                    performance = (merged.loc[valid, "direction"] * merged.loc[valid, f"fwd_ret_z_{horizon}"]).mean()
                    cache[key] = (float(performance), int(valid.sum()))
                performance, count = cache[key]
                surface_rows.append({
                    "detector_id": detector.detector_id, "parameter_1": first, "value_1": first_value,
                    "parameter_2": second, "value_2": second_value, "performance": performance, "n": count,
                })
        surface_2d = pd.DataFrame(surface_rows)
        if surface_2d["performance"].notna().any():
            best = surface_2d.loc[surface_2d["performance"].idxmax()]
            first_values = list(detector.param_grid[first]); second_values = list(detector.param_grid[second])
            i, j = first_values.index(best["value_1"]), second_values.index(best["value_2"])
            neighborhood = surface_2d.loc[
                surface_2d["value_1"].isin(first_values[max(0, i - 1): i + 2])
                & surface_2d["value_2"].isin(second_values[max(0, j - 1): j + 2])
                & ~((surface_2d["value_1"] == best["value_1"]) & (surface_2d["value_2"] == best["value_2"]))
            ]
            if best["performance"] > 0 and len(neighborhood):
                ratios.append(float(neighborhood["performance"].median() / best["performance"]))
    else:
        surface_2d = pd.DataFrame()
    stability = float(np.clip(np.median(ratios), 0, 1)) if ratios else 0.0
    return one_dimensional, surface_2d, stability, len(cache)
