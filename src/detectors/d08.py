"""D08 / H-08: US-close rebalance pressure."""

import numpy as np

from src.detectors.base import Detector


class D08(Detector):
    def __init__(self, **params):
        defaults = {"z_threshold": 0.8, "event_minute": 15 * 60 + 30}
        defaults.update(params)
        super().__init__("D08", "H-08", defaults, {"z_threshold": [0.5, 0.8, 1.1], "event_minute": [900, 930, 945]}, 12, -1, ("sigma_hat", "is_month_end", "is_quarter_end"))

    def detect(self, bars, ctx):
        f = self._frame(bars, ctx); p = self.params
        z = (f.ret / f.sigma_hat).fillna(0).groupby([f.root, f.session_id], sort=False).cumsum()
        event = f.root.isin(["ES", "NQ", "RTY"]) & f.local_minute.eq(p["event_minute"]) & ~f.is_half_day
        trigger = event & z.abs().gt(p["z_threshold"])
        direction = np.where(trigger, -np.sign(z), 0)
        calendar_boost = 1 + 0.25 * f.is_month_end.astype(float) + 0.25 * f.is_quarter_end.astype(float)
        magnitude = (((z.abs() - p["z_threshold"]) / 2).clip(0, 1) * calendar_boost).clip(0, 1).where(trigger, 0)
        return self._output(f, direction, magnitude, z_day=z)
