"""D06 / H-06: causal synchronous ES-to-HSI relative-move residual."""

import numpy as np
import pandas as pd

from src.detectors.base import Detector


class D06(Detector):
    def __init__(self, **params):
        defaults = {"k": 1, "tail_q": 0.15, "min_bars": 1000}
        defaults.update(params)
        super().__init__("D06", "H-06", defaults, {"k": [1, 2, 3, 6, 12], "tail_q": [0.1, 0.15, 0.2]}, 12, -1, ("sigma_hat", "es_ret_z"))

    def detect(self, bars, ctx):
        f = self._frame(bars, ctx); p = self.params
        output_direction = pd.Series(0, index=f.index, dtype=np.int8)
        residual_all = pd.Series(np.nan, index=f.index)
        magnitude = pd.Series(0.0, index=f.index)
        hsi = f.root.eq("HSI")
        part = f.loc[hsi].copy()
        hsi_z = part.ret / part.sigma_hat
        own = hsi_z.rolling(p["k"], min_periods=p["k"]).sum()
        es = part.es_ret_z.rolling(p["k"], min_periods=p["k"]).sum()
        cov = own.expanding(p["min_bars"]).cov(es).shift(1)
        var = es.expanding(p["min_bars"]).var().shift(1)
        beta = cov / var.replace(0, np.nan)
        residual = own - beta * es
        lower = residual.expanding(p["min_bars"]).quantile(p["tail_q"]).shift(1)
        upper = residual.expanding(p["min_bars"]).quantile(1 - p["tail_q"]).shift(1)
        trigger = residual.lt(lower) | residual.gt(upper)
        output_direction.loc[part.index] = np.where(trigger, -np.sign(residual), 0).astype(np.int8)
        scale = (upper - lower).replace(0, np.nan) / 2
        magnitude.loc[part.index] = ((residual.abs() / scale - 1).clip(0, 1)).where(trigger, 0).fillna(0)
        residual_all.loc[part.index] = residual
        return self._output(f, output_direction, magnitude, residual=residual_all)
