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

from dataclasses import dataclass

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


# ---------------------------------------------------------------------------
# Classical baseline: a Gaussian HMM fitted by Baum-Welch
# ---------------------------------------------------------------------------


@dataclass
class GaussianHmm:
    means: np.ndarray  # (K,)
    sds: np.ndarray  # (K,)
    trans: np.ndarray  # (K, K)
    start: np.ndarray  # (K,)

    def emission(self, x: np.ndarray) -> np.ndarray:
        """``(T, K)`` densities; a missing (nan) day has density 1 in every state."""
        z = (x[:, None] - self.means[None]) / self.sds[None]
        dens = np.exp(-0.5 * z**2) / (np.sqrt(2 * np.pi) * self.sds[None])
        return np.where(np.isnan(x)[:, None], 1.0, np.maximum(dens, 1e-300))

    def filter(self, x: np.ndarray) -> np.ndarray:
        """Filtered state probabilities ``(T, K)``."""
        return _forward(self.emission(x), self.trans, self.start)[0]


def _forward(e: np.ndarray, trans: np.ndarray, start: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T, K = e.shape
    alpha, c = np.empty((T, K)), np.empty(T)
    prior = start
    for t in range(T):
        a = prior * e[t]
        c[t] = a.sum()
        alpha[t] = a / c[t]
        prior = alpha[t] @ trans
    return alpha, c


def fit_gaussian_hmm(seqs: list[np.ndarray], n_states: int, *, n_iter: int = 100, tol: float = 1e-4,
                     seed: int = 0) -> GaussianHmm:
    """Baum-Welch on many sequences; nan entries are missing observations."""
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([s[~np.isnan(s)] for s in seqs])
    K = n_states
    means = np.quantile(pooled, (np.arange(K) + 0.5) / K) + rng.normal(0, 0.01 * pooled.std(), K)
    sds = np.full(K, pooled.std() / K)
    trans = np.full((K, K), 0.05 / max(K - 1, 1)) + np.eye(K) * (0.95 - 0.05 / max(K - 1, 1))
    start = np.full(K, 1 / K)
    last = -np.inf
    for _ in range(n_iter):
        hmm = GaussianHmm(means, sds, trans, start)
        num_t, gam0, w_sum, wx, wxx, ll = np.zeros((K, K)), np.zeros(K), np.zeros(K), np.zeros(K), np.zeros(K), 0.0
        for x in seqs:
            e = hmm.emission(x)
            alpha, c = _forward(e, trans, start)
            T = len(x)
            beta = np.ones((T, K))
            for t in range(T - 2, -1, -1):
                beta[t] = (trans @ (e[t + 1] * beta[t + 1])) / c[t + 1]
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True)
            xi = alpha[:-1, :, None] * trans[None] * (e[1:] * beta[1:])[:, None, :] / c[1:, None, None]
            num_t += xi.sum(axis=0)
            gam0 += gamma[0]
            seen = ~np.isnan(x)
            g = gamma[seen]
            w_sum += g.sum(axis=0)
            wx += (g * x[seen, None]).sum(axis=0)
            wxx += (g * x[seen, None] ** 2).sum(axis=0)
            ll += np.log(c).sum()
        means = wx / w_sum
        sds = np.sqrt(np.maximum(wxx / w_sum - means**2, 1e-6))
        trans = num_t / num_t.sum(axis=1, keepdims=True)
        start = gam0 / gam0.sum()
        if ll - last < tol * abs(ll):
            break
        last = ll
    return GaussianHmm(means, sds, trans, start)


def save_hmm(hmm: GaussianHmm, path) -> None:
    import json
    from pathlib import Path

    Path(path).write_text(json.dumps({k: getattr(hmm, k).tolist() for k in ("means", "sds", "trans", "start")}))


def load_hmm(path) -> GaussianHmm:
    import json
    from pathlib import Path

    raw = json.loads(Path(path).read_text())
    return GaussianHmm(**{k: np.asarray(v) for k, v in raw.items()})


def shop_log_sales(sales: np.ndarray, censored: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """The HMM's observation: log total shop sales, weekday-adjusted and centred.

    Returns ``(x, centre, weekday)``: ``x`` is ``(T,)`` with nan on days a
    product sold out (sales there are only a lower bound), ``centre`` the mean
    removed, ``weekday`` the additive log weekday index of row ``t % 7``.
    """
    raw = np.log(sales.sum(axis=1) + 1.0)
    raw = np.where(censored.any(axis=1), np.nan, raw)
    dow = np.arange(len(raw)) % 7
    centre = float(np.nanmean(raw))
    weekday = np.array([np.nanmean(raw[dow == d]) - centre if np.isfinite(raw[dow == d]).any() else 0.0
                        for d in range(7)])
    return raw - centre - weekday[dow], centre, weekday


def fit_hmm_on_sales(series: list[tuple[np.ndarray, np.ndarray]], n_states: int, seed: int = 0) -> GaussianHmm:
    """Baum-Welch on the shop observation of many ``(sales, censored)`` series."""
    return fit_gaussian_hmm([shop_log_sales(s, c)[0] for s, c in series], n_states, seed=seed)


class HmmForecaster:
    """The planner forecaster of a fitted :class:`GaussianHmm` (the classical regime baseline).

    It filters the regime from the shop observation, draws a regime
    trajectory per path from the belief and the chain, and turns it into
    total demand ``exp(centre + mean_k + weekday + noise_k) - 1``. Each
    product gets its share of the last 28 days' sales, times its own
    mean-one log-normal noise, whose sd is the product's residual sd over the
    last 56 days less the shop-level noise.
    """

    name = "hmm"

    def __init__(self, hmm: GaussianHmm, horizon: int = HORIZON):
        self.hmm, self.horizon = hmm, horizon

    def paths(self, obs, n_paths: int, rng: np.random.Generator) -> np.ndarray:
        hmm, H = self.hmm, self.horizon
        x, centre, weekday = shop_log_sales(obs.sales, obs.censored)
        post = hmm.filter(x)
        belief = post[-1]
        n = len(x)
        recent = obs.sales[-28:].astype(float)
        share = (recent.sum(axis=0) + 1.0) / (recent.sum() + recent.shape[1])
        # Per-product log residual sd around the filtered shop level, last 56 days.
        fitted = np.exp(centre + post[-56:] @ hmm.means + weekday[np.arange(n - 56, n) % 7]) - 1.0
        resid = np.log(obs.sales[-56:] + 1.0) - np.log(np.maximum(share[None] * fitted[:, None], 0) + 1.0)
        ok = ~obs.censored[-56:]
        sd = np.array([resid[ok[:, p], p].std() if ok[:, p].sum() > 2 else 0.3 for p in range(resid.shape[1])])
        shop_sd = float(belief @ hmm.sds)
        sd_p = np.sqrt(np.maximum(sd**2 - shop_sd**2, 0.0))
        K = len(hmm.means)
        state = rng.choice(K, size=n_paths, p=belief)
        cum = np.cumsum(hmm.trans, axis=1)
        log_total = np.empty((n_paths, H))
        for h in range(H):
            state = np.minimum((rng.random(n_paths)[:, None] > cum[state]).sum(axis=1), K - 1)
            log_total[:, h] = (centre + hmm.means[state] + weekday[(n + h) % 7]
                               + rng.normal(0, 1, n_paths) * hmm.sds[state])
        P = len(share)
        noise = rng.normal(0, 1, (n_paths, H, P)) * sd_p[None, None] - sd_p[None, None] ** 2 / 2  # mean-preserving
        return np.rint(np.maximum(share[None, None] * (np.exp(log_total[:, :, None] + noise) - 1.0), 0))

    def __call__(self, obs) -> np.ndarray:
        """Marginal quantiles ``(H, P, Q)`` of the paths (for diagnostics and the copula sampler)."""
        paths = self.paths(obs, 512, np.random.default_rng([obs.day, 11]))
        return np.quantile(paths, QUANTILES, axis=0).transpose(1, 2, 0)


# ---------------------------------------------------------------------------
# Copula sampler: the coherence baseline (track T2 builds its own in planner.py)
# ---------------------------------------------------------------------------


class CopulaSampler:
    """Paths from a quantile forecaster's marginals under a one-factor Gaussian copula.

    Every path draws one shared normal ``g`` and a normal ``e`` per (day,
    product); its uniforms are ``Phi(sqrt(rho) g + sqrt(1 - rho) e)``. So
    each day's marginal is the forecaster's, and ``rho`` sets how much one
    level error is shared across the horizon and the products. ``rho = 0`` is
    the planner's independent-day sampler.
    """

    def __init__(self, forecaster, rho: float):
        self.forecaster, self.rho = forecaster, float(rho)
        self.name = forecaster.name

    def __call__(self, obs) -> np.ndarray:
        return self.forecaster(obs)

    def paths(self, obs, n_paths: int, rng: np.random.Generator) -> np.ndarray:
        from scipy.stats import norm

        from src.perfect_supplier.planner import paths_from_uniforms

        q = self.forecaster(obs)
        H, P, _ = q.shape
        z = np.sqrt(self.rho) * rng.standard_normal((n_paths, 1, 1)) + np.sqrt(1 - self.rho) * rng.standard_normal(
            (n_paths, H, P))
        return paths_from_uniforms(q, norm.cdf(z))
