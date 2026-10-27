"""Track T1 forecasters for the ``hidden_regimes`` world family.

:class:`BayesRegimeForecaster` is the known-model ceiling: it knows the
world's regime multipliers, noise, transition matrix, and each product's
non-regime mean path (base mean, idiosyncratic level, weekday; as the oracle
does), and filters the hidden regime from the sales the shop has seen. A
sold-out day only says demand was at least the sale. It draws demand paths
that follow one regime trajectory each, sampled from the belief and the
chain, so lead-time sums carry the regime uncertainty.
"""

from __future__ import annotations

import numpy as np

from src.perfect_supplier.forecast import HORIZON, QUANTILES
from src.perfect_supplier.world import (
    HISTORY_DAYS,
    World,
    base_mean_path,
    regime_transition,
    uniform_half_width,
)

LIKELIHOOD_FLOOR = 1e-4


def _uniform_cdf(y: np.ndarray, lam: np.ndarray, a: np.ndarray) -> np.ndarray:
    """P(lam * U < y) for U uniform on [1 - a, 1 + a]."""
    return np.clip((y / lam - (1 - a)) / (2 * a), 0, 1)


def regime_likelihood(sales: np.ndarray, censored: np.ndarray, mu: np.ndarray, mult, half) -> np.ndarray:
    """Per-day likelihood of every regime, ``(T, K)``, multiplied over products.

    ``sales``, ``censored``, ``mu`` are ``(T, P)``; demand is
    ``rint(mu * mult[k] * U)`` with ``U`` uniform of half-width ``half[k]``.
    """
    lam = mu[:, :, None] * np.asarray(mult)[None, None, :]  # (T, P, K)
    a = np.asarray(half)[None, None, :]
    s = sales[:, :, None].astype(float)
    exact = _uniform_cdf(s + 0.5, lam, a) - _uniform_cdf(s - 0.5, lam, a)
    at_least = 1 - _uniform_cdf(s - 0.5, lam, a)
    p = np.where(censored[:, :, None], at_least, exact)
    p = LIKELIHOOD_FLOOR + (1 - LIKELIHOOD_FLOOR) * p
    return np.exp(np.log(p).sum(axis=1))


class BayesRegimeForecaster:
    name = "bayes"

    def __init__(self, world: World, horizon: int = HORIZON):
        if not world.regime_mult:
            raise ValueError("the Bayes regime forecaster needs a hidden_regimes world")
        self.mu = base_mean_path(world)
        self.mult = np.asarray(world.regime_mult)
        self.half = uniform_half_width(world.cv) * np.asarray(world.regime_cv)
        self.trans = regime_transition(world)
        self.horizon = horizon
        K = len(self.mult)
        self._post = np.full(K, 1 / K)  # filtered belief after row self._seen - 1
        self._seen = 0

    def belief(self, obs) -> np.ndarray:
        """Filtered regime probabilities on the day of ``obs`` (after its sales)."""
        n = len(obs.sales)
        if n < self._seen:  # a new episode on the same world: start over
            self._post, self._seen = np.full(len(self.mult), 1 / len(self.mult)), 0
        lik = regime_likelihood(obs.sales[self._seen : n], obs.censored[self._seen : n],
                                self.mu[self._seen : n], self.mult, self.half)
        post = self._post
        for t, row in enumerate(lik):
            prior = post if self._seen + t == 0 else post @ self.trans
            post = prior * row
            post = post / post.sum()
        self._post, self._seen = post, n
        return post

    def paths(self, obs, n_paths: int, rng: np.random.Generator) -> np.ndarray:
        """Demand paths ``(n_paths, H, P)`` for days ``t + 1 .. t + H``, one regime trajectory each."""
        post = self.belief(obs)
        start = HISTORY_DAYS + obs.day + 1
        rows = np.minimum(np.arange(start, start + self.horizon), len(self.mu) - 1)
        K = len(self.mult)
        state = rng.choice(K, size=n_paths, p=post)
        regimes = np.empty((n_paths, self.horizon), dtype=np.int64)
        cum = np.cumsum(self.trans, axis=1)
        for h in range(self.horizon):
            u = rng.random(n_paths)
            state = np.minimum((u[:, None] > cum[state]).sum(axis=1), K - 1)
            regimes[:, h] = state
        a = self.half[regimes][:, :, None]
        noise = 1 + a * rng.uniform(-1, 1, (n_paths, self.horizon, self.mu.shape[1]))
        lam = self.mu[rows][None] * self.mult[regimes][:, :, None]
        return np.rint(np.clip(lam * noise, 0, None))

    def __call__(self, obs) -> np.ndarray:
        """Marginal quantiles ``(H, P, Q)`` of the paths, for diagnostics."""
        paths = self.paths(obs, 512, np.random.default_rng([obs.day, 11]))
        return np.quantile(paths, QUANTILES, axis=0).transpose(1, 2, 0)
