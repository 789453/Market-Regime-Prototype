"""D04 / H-04: directional break after volatility compression."""

import numpy as np

from src.detectors.base import Detector
from src.detectors.common import grouped_shift_rolling, rolling_percentile


class D04(Detector):
    def __init__(self, **params):
        defaults = {"v_lo": 0.8, "persist": 6, "lookback": 24, "range_window": 120, "range_pct": 0.25, "alpha": 0.3}
        defaults.update(params)
        super().__init__("D04", "H-04", defaults, {"v_lo": [0.7, 0.8, 0.9], "lookback": [12, 24, 48], "range_pct": [0.2, 0.25, 0.3]}, 24, 1, ("sigma_hat", "vol_ratio", "sigma_pct"))

    def detect(self, bars, ctx):
        f = self._frame(bars, ctx); p = self.params; roots = f.root
        compressed = f.vol_ratio.groupby(roots, sort=False).transform(
            lambda x: x.shift(1).rolling(p["persist"], min_periods=p["persist"]).max()
        ) < p["v_lo"]
        high = grouped_shift_rolling(f.high, roots, p["lookback"], "max")
        low = grouped_shift_rolling(f.low, roots, p["lookback"], "min")
        width = np.log(high / low)
        range_rank = rolling_percentile(width, roots, p["range_window"], min_periods=40)
        up = np.log(f.close / high) > p["alpha"] * f.sigma_hat
        down = np.log(low / f.close) > p["alpha"] * f.sigma_hat
        trigger = compressed & (range_rank < p["range_pct"]) & (up | down)
        direction = np.select([trigger & up, trigger & down], [1, -1], default=0)
        distance = np.maximum(np.log(f.close / high), np.log(low / f.close)) / (p["alpha"] * f.sigma_hat)
        return self._output(f, direction, ((distance - 1) / 2).clip(0, 1).where(trigger, 0), range_pct=range_rank, compressed=compressed)
