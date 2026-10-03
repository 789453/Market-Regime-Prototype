"""D05 / H-05: nonlinear session-open repricing."""

import numpy as np
import pandas as pd

from src.detectors.base import Detector


class D05(Detector):
    def __init__(self, **params):
        defaults = {"confirm_bars": 3, "small_z": 0.5, "large_z": 1.5}
        defaults.update(params)
        super().__init__("D05", "H-05", defaults, {"confirm_bars": [2, 3, 4], "small_z": [0.4, 0.5, 0.6], "large_z": [1.2, 1.5, 1.8]}, 12, 1, ("sigma_hat",))

    def detect(self, bars, ctx):
        f = self._frame(bars, ctx); p = self.params
        previous = f.close.groupby(f.root, sort=False).shift(1)
        previous_session = f.session_id.groupby(f.root, sort=False).shift(1)
        session_open_prev = previous.where(f.session_id.ne(previous_session)).groupby(f.root, sort=False).ffill()
        k = p["confirm_bars"]
        globex_or_hk = f.bar_in_session.eq(k - 1)
        cash = f.root.isin(["ES", "NQ", "RTY"]) & f.local_minute.eq(9 * 60 + 30 + (k - 1) * 5)
        baseline_cash = f.close.groupby(f.root, sort=False).shift(k)
        baseline = session_open_prev.where(globex_or_hk, baseline_cash)
        event = globex_or_hk | cash
        z = np.log(f.close / baseline) / (f.sigma_hat * np.sqrt(k))
        reverse = event & z.abs().lt(p["small_z"])
        continue_ = event & z.abs().gt(p["large_z"])
        direction = np.select([reverse, continue_], [-np.sign(z), np.sign(z)], default=0)
        magnitude = np.select([reverse, continue_], [(p["small_z"] - z.abs()) / p["small_z"], ((z.abs() - p["large_z"]) / p["large_z"]).clip(0, 1)], default=0)
        return self._output(f, direction, magnitude, open_z=z, event_type=np.where(cash, "cash", "session"))
