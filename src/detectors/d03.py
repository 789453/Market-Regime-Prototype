"""D03 / H-03: lazy-pullback trend continuation."""

import numpy as np

from src.detectors.base import Detector
from src.structure import causal_structure_features


class D03(Detector):
    def __init__(self, **params):
        defaults = {"kappa": 2.0, "retrace_lo": 0.3, "retrace_hi": 0.65, "time_ratio": 1.2, "volume_ratio": 0.85}
        defaults.update(params)
        super().__init__("D03", "H-03", defaults, {"kappa": [1.5, 2.0, 2.5, 3.0], "retrace_hi": [0.55, 0.65, 0.75]}, 24, 1, ("sigma_hat",))

    def detect(self, bars, ctx):
        f = self._frame(bars, ctx); p = self.params
        structure = causal_structure_features(f, kappa=p["kappa"])
        stable = structure.bars_since_confirm.between(2, 4)
        trigger = stable & structure.retrace_ratio.between(p["retrace_lo"], p["retrace_hi"]) & (structure.time_ratio > p["time_ratio"]) & (structure.volume_ratio < p["volume_ratio"])
        direction = np.where(trigger, structure.trend_direction.fillna(0), 0)
        magnitude = ((p["retrace_hi"] - structure.retrace_ratio).clip(0) / (p["retrace_hi"] - p["retrace_lo"])).where(trigger, 0)
        return self._output(f, direction, magnitude, retrace=structure.retrace_ratio, time_ratio=structure.time_ratio, volume_ratio=structure.volume_ratio)
