"""Demand forecasters the planner can use.

A forecaster maps an :class:`~src.perfect_supplier.sim.Obs` to quantiles of
daily demand for the next ``horizon`` days (row 0 is tomorrow), shape
``(horizon, n_products, len(QUANTILES))``. The policy decides after
today's sales, so row 0 is tomorrow.

- :class:`OracleForecaster` knows the world's true expected path and CV.
  It is the ceiling, not a policy a shop could run.
- :class:`MovingAverageForecaster` is the textbook rate: the mean and
  standard deviation of the last ``window`` days of sales, flat over the
  horizon, normal quantiles.
- :class:`JepaForecaster` wraps a trained :class:`~src.perfect_supplier.jepa.JepaModel`.

A forecaster's ``levels`` are its quantile levels (``QUANTILES`` unless
``levels`` is passed).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from src.perfect_supplier.world import HISTORY_DAYS, World, demand_mean_path, uniform_half_width

QUANTILES = np.array([0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98])
# QUANTILES plus the upper tail: the critical ratios are above 0.99, so the tail sets safety stock.
TAIL_QUANTILES = np.concatenate([QUANTILES, [0.99, 0.995, 0.999]])
HORIZON = 35


class OracleForecaster:
    name = "oracle"

    def __init__(self, world: World, horizon: int = HORIZON, levels: np.ndarray = QUANTILES):
        self.mean = demand_mean_path(world)
        self.half = uniform_half_width(world.cv)
        self.horizon = horizon
        self.levels = np.asarray(levels)

    def __call__(self, obs) -> np.ndarray:
        start = HISTORY_DAYS + obs.day + 1
        mean = self.mean[start : start + self.horizon]
        if len(mean) < self.horizon:  # past the episode end: hold the last level
            mean = np.vstack([mean, np.repeat(mean[-1:], self.horizon - len(mean), axis=0)])
        u = 1 - self.half + 2 * self.half * self.levels
        return np.clip(mean[:, :, None] * u[None, None, :], 0, None)


class MovingAverageForecaster:
    name = "moving_average"

    def __init__(self, window: int = 28, horizon: int = HORIZON, levels: np.ndarray = QUANTILES):
        self.window, self.horizon = window, horizon
        self.levels = np.asarray(levels)

    def __call__(self, obs) -> np.ndarray:
        recent = obs.sales[-self.window :].astype(float)
        mu = recent.mean(axis=0)
        sd = recent.std(axis=0, ddof=1) if len(recent) > 1 else np.zeros_like(mu)
        q = mu[:, None] + sd[:, None] * norm.ppf(self.levels)[None, :]
        return np.repeat(np.clip(q, 0, None)[None], self.horizon, axis=0)
