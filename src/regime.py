"""Detector-specific, sign-preregistered regime gates."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def _normal(value: pd.Series, scale: float = 2.0) -> pd.Series:
    return 1 - 2 * (value.abs() / scale).clip(0, 1)


def detector_gate(frame: pd.DataFrame, detector_id: str, minimum: float = 0.2, maximum: float = 1.5) -> pd.Series:
    high_er = 2 * frame["er_20"].clip(0, 1) - 1
    expansion = np.tanh(np.log(frame["vol_ratio"].clip(lower=1e-6)) / 0.4)
    low_sigma = 1 - 2 * frame["sigma_pct"].clip(0, 1)
    mid_sigma = 1 - 4 * (frame["sigma_pct"] - 0.5).abs().clip(0, 0.5)
    normal_liquidity = _normal(frame["liq_z"], 3.0)
    high_liquidity = np.tanh(frame["liq_z"] / 2)
    components: list[pd.Series]
    if detector_id == "D01":
        event_penalty = pd.Series(np.where(frame["session_phase"].isin(["US_OPEN", "US_CLOSE"]), -1.0, 1.0), index=frame.index)
        components = [-high_er, -expansion, mid_sigma, normal_liquidity, event_penalty]
    elif detector_id == "D02":
        components = [high_er, expansion, high_liquidity]
    elif detector_id == "D03":
        components = [high_er, mid_sigma, high_liquidity]
    elif detector_id == "D04":
        components = [low_sigma, expansion, normal_liquidity]
    elif detector_id in {"D05", "D06"}:
        components = [normal_liquidity, mid_sigma]
    elif detector_id == "D07":
        reversion = frame.get("meta_branch", pd.Series("none", index=frame.index)).eq("revert")
        phase_reversion = pd.Series(np.where(frame["session_phase"].eq("US_LUNCH"), 1.0, -0.25), index=frame.index)
        branch_score = pd.Series(np.where(reversion, (-high_er - expansion + phase_reversion) / 3, (high_er + expansion + high_liquidity) / 3), index=frame.index)
        components = [branch_score, normal_liquidity]
    elif detector_id == "D08":
        calendar = frame["is_month_end"].astype(float) + frame["is_quarter_end"].astype(float) - 0.5
        components = [calendar.clip(-1, 1), normal_liquidity]
    else:
        raise ValueError(detector_id)
    score = pd.concat(components, axis=1).mean(axis=1).fillna(0)
    return minimum + (maximum - minimum) / (1 + np.exp(-score))


def gate_curve(frame: pd.DataFrame, outcome: str = "outcome", quantiles: int = 10):
    valid = frame.dropna(subset=["detector_gate", outcome]).copy()
    if len(valid) < quantiles * 3:
        return pd.DataFrame(), {"isotonic_r2": np.nan, "adjacent_jump_ratio": np.nan, "passed": False}
    valid["gate_bin"] = pd.qcut(valid["detector_gate"].rank(method="first"), quantiles, labels=False)
    curve = valid.groupby("gate_bin", observed=True).agg(gate_mean=("detector_gate", "mean"), outcome_mean=(outcome, "mean"), n=(outcome, "size")).reset_index()
    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    curve["isotonic"] = model.fit_transform(curve["gate_mean"], curve["outcome_mean"], sample_weight=curve["n"])
    centered = curve["outcome_mean"] - np.average(curve["outcome_mean"], weights=curve["n"])
    residual = curve["outcome_mean"] - curve["isotonic"]
    denominator = float(np.sum(curve["n"] * centered**2))
    r2 = 1 - float(np.sum(curve["n"] * residual**2)) / denominator if denominator > 0 else 0.0
    amplitude = float(curve["isotonic"].max() - curve["isotonic"].min())
    jump = float(curve["isotonic"].diff().abs().max() / amplitude) if amplitude > 0 else np.inf
    metrics = {"isotonic_r2": r2, "adjacent_jump_ratio": jump, "passed": bool(r2 >= 0.6 and jump < 0.4)}
    return curve, metrics
