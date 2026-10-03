"""Unified causal detector contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = ["root", "timestamp", "direction", "magnitude", "horizon", "confidence"]


@dataclass
class Detector(ABC):
    detector_id: str
    hypothesis_id: str
    params: dict = field(default_factory=dict)
    param_grid: dict = field(default_factory=dict)
    default_horizon: int = 12
    expected_sign: int = 1
    required_context: tuple[str, ...] = ()

    @abstractmethod
    def detect(self, bars: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
        """Return one causal signal row per input bar."""

    def _frame(self, bars: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.required_context) - set(ctx.columns)
        if missing:
            raise ValueError(f"{self.detector_id} missing context: {sorted(missing)}")
        left = bars.sort_values(["root", "timestamp"], kind="stable").reset_index(drop=True)
        right = ctx.reset_index() if isinstance(ctx.index, pd.MultiIndex) else ctx.copy()
        keep = ["root", "timestamp", *(column for column in self.required_context if column not in left.columns)]
        keep = list(dict.fromkeys(keep))
        return left.merge(right[keep], on=["root", "timestamp"], how="left", validate="one_to_one")

    def _output(
        self,
        frame: pd.DataFrame,
        direction: pd.Series | np.ndarray,
        magnitude: pd.Series | np.ndarray,
        **meta: pd.Series | np.ndarray,
    ) -> pd.DataFrame:
        output = frame[["root", "timestamp"]].copy()
        output["direction"] = np.asarray(direction, dtype=np.int8)
        output["magnitude"] = np.nan_to_num(np.asarray(magnitude, dtype=float), nan=0.0).clip(0, 1)
        output["horizon"] = np.int16(self.default_horizon)
        output["confidence"] = 1.0
        for name, values in meta.items():
            output[f"meta_{name}"] = np.asarray(values)
        self.validate(output)
        return output

    @staticmethod
    def validate(output: pd.DataFrame) -> None:
        if not set(OUTPUT_COLUMNS).issubset(output.columns):
            raise ValueError("detector output contract incomplete")
        if not output["direction"].isin([-1, 0, 1]).all():
            raise ValueError("direction must be -1/0/+1")
        if not output["magnitude"].between(0, 1).all():
            raise ValueError("magnitude outside [0,1]")
        if not output["confidence"].between(0, 1).all():
            raise ValueError("confidence outside [0,1]")
