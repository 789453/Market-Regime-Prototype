"""D02 / H-02: volume-confirmed breakout; shares D01 event logic."""

import numpy as np

from src.detectors.base import Detector
from src.detectors.common import breakout_features


class D02(Detector):
    def __init__(self, **params):
        defaults = {"lookback": 48, "confirm": 4, "alpha": 0.3, "q_hi": 1.8}
        defaults.update(params)
        super().__init__("D02", "H-02", defaults, {"lookback": [24, 48, 96], "q_hi": [1.5, 1.8, 2.2]}, 24, 1, ("sigma_hat",))

    def detect(self, bars, ctx):
        f = self._frame(bars, ctx); p = self.params
        event = breakout_features(f, p["lookback"], p["confirm"], p["alpha"])
        held_up = (event.breakout_side == 1) & (f.close > event.ref_high)
        held_down = (event.breakout_side == -1) & (f.close < event.ref_low)
        trigger = (event.volume_ratio > p["q_hi"]) & (held_up | held_down)
        direction = np.where(trigger, event.breakout_side, 0)
        magnitude = ((event.breakout_strength - 1).clip(0, 2) / 2 * (event.volume_ratio / p["q_hi"] - 1).clip(0, 2) / 2).where(trigger, 0)
        return self._output(f, direction, magnitude, breakout_side=event.breakout_side, volume_ratio=event.volume_ratio)
