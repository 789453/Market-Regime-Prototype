"""D01 / H-01: liquidity-sweep reversal."""

import numpy as np

from src.detectors.base import Detector
from src.detectors.common import breakout_features


class D01(Detector):
    def __init__(self, **params):
        defaults = {"lookback": 48, "confirm": 4, "alpha": 0.3, "q_lo": 1.2}
        defaults.update(params)
        super().__init__("D01", "H-01", defaults, {"lookback": [24, 48, 96], "alpha": [0.2, 0.3, 0.4]}, 24, -1, ("sigma_hat",))

    def detect(self, bars, ctx):
        f = self._frame(bars, ctx); p = self.params
        event = breakout_features(f, p["lookback"], p["confirm"], p["alpha"])
        reclaimed_up = (event.breakout_side == 1) & (f.close < event.ref_high)
        reclaimed_down = (event.breakout_side == -1) & (f.close > event.ref_low)
        trigger = (event.volume_ratio < p["q_lo"]) & (reclaimed_up | reclaimed_down)
        direction = np.where(trigger, -event.breakout_side, 0)
        magnitude = ((event.breakout_strength - 1).clip(0, 2) / 2 * (p["q_lo"] / event.volume_ratio).clip(0, 2) / 2).where(trigger, 0)
        return self._output(f, direction, magnitude, breakout_side=event.breakout_side, volume_ratio=event.volume_ratio)
