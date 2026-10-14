"""Behaviour policies that write the training log.

The forecaster learns from sales, and sales are censored by whatever policy
ran the shop. The mixture spans lean and fat shelves so the log holds both
clean and censored stretches: every textbook family with parameters drawn
wide, plus a random-cover rule.
"""

from __future__ import annotations

import numpy as np

from src.perfect_supplier.sim import Decision, Obs
from src.perfect_supplier.textbook import FAMILIES, TextbookParams, TextbookPolicy


class RandomCoverPolicy:
    """On each review day, order up to ``(lag + cover) * rate`` with a fresh random cover."""

    name = "random_cover"

    def __init__(self, seed: int, review_interval: int, cover_range: tuple[float, float] = (0.0, 30.0)):
        self.rng = np.random.default_rng([seed, 7])
        self.review_interval = review_interval
        self.cover_range = cover_range

    def decide(self, obs: Obs) -> Decision:
        if (obs.day + 1) % self.review_interval:
            return Decision(np.zeros(obs.n_products, dtype=np.int64), {"cover": np.nan})
        cover = float(self.rng.uniform(*self.cover_range))
        rate = obs.sales[-14:].mean(axis=0)
        target = (obs.delivery_lag + cover) * rate
        qty = np.maximum(np.rint(target - obs.position), 0).astype(np.int64)
        return Decision(qty, {"cover": cover})


def sample_behavior(seed: int):
    """One behaviour policy for the training world with this seed."""
    rng = np.random.default_rng([seed, 5])
    if rng.random() < 0.25:
        policy = RandomCoverPolicy(seed, review_interval=int(rng.integers(1, 8)))
        policy.name = "random_cover"
        return policy
    family = FAMILIES[int(rng.integers(len(FAMILIES)))]
    params = TextbookParams(
        family=family,
        cover_horizon_ticks=int(rng.integers(1, 29)),
        safety_lead_pct_of_lag=float(rng.uniform(0.0, 1.5)),
        stockout_safety_bonus_pct_of_lag=float(rng.uniform(0.0, 1.0)),
        review_interval=int(rng.integers(1, 15)),
        demand_window=int(rng.integers(2, 29)),
    )
    return TextbookPolicy(params, name=family)
