"""D07 / H-07: session VWAP reversion/continuation branches."""

import numpy as np

from src.detectors.base import Detector


class D07(Detector):
    def __init__(self, **params):
        defaults = {"d_lo": 0.8, "d_hi": 2.0, "slope_window": 12, "flat_slope": 0.15}
        defaults.update(params)
        super().__init__("D07", "H-07", defaults, {"d_lo": [0.5, 0.8], "d_hi": [1.5, 2.0, 2.5], "slope_window": [6, 12, 24]}, 12, -1, ("sigma_hat",))

    def detect(self, bars, ctx):
        f = self._frame(bars, ctx); p = self.params
        valid_volume = f.volume.where(f.is_valid, 0)
        weighted = (f.wap * valid_volume).groupby([f.root, f.session_id], sort=False).cumsum()
        cumulative_volume = valid_volume.groupby([f.root, f.session_id], sort=False).cumsum()
        vwap = weighted / cumulative_volume.replace(0, np.nan)
        elapsed = np.sqrt(f.bar_in_session.clip(lower=0) + 1)
        deviation = np.log(f.close / vwap) / (f.sigma_hat * elapsed)
        slope = np.log(vwap / vwap.groupby([f.root, f.session_id], sort=False).shift(p["slope_window"])) / (f.sigma_hat * np.sqrt(p["slope_window"]))
        reversion = deviation.abs().gt(p["d_hi"]) & slope.abs().lt(p["flat_slope"])
        continuation = deviation.abs().between(p["d_lo"], p["d_hi"]) & (np.sign(deviation) == np.sign(slope))
        direction = np.select([reversion, continuation], [-np.sign(deviation), np.sign(deviation)], default=0)
        magnitude = np.select([reversion, continuation], [((deviation.abs() - p["d_hi"]) / p["d_hi"]).clip(0, 1), ((deviation.abs() - p["d_lo"]) / (p["d_hi"] - p["d_lo"])).clip(0, 1)], default=0)
        return self._output(f, direction, magnitude, deviation=deviation, vwap_slope=slope, branch=np.where(reversion, "revert", np.where(continuation, "continue", "none")))
