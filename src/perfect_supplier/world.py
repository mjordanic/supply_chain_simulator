"""One perfect-supplier world: economics, opening state, and the demand path.

Every product is active for the whole run. There is one supplier per shop and
it always ships the quantity the shop paid for; that quantity arrives after
``delivery_lag`` days. Demand is exogenous (lost sales do not change it), so
one ``world_seed`` gives the same demand path to every policy run on it.

Demand for product ``p`` on day ``t`` is::

    round(clip(base_mean[p] * level[p, t] * weekday[t % 7] * U, 0, None))

``U`` is uniform with mean 1 and coefficient of variation ``cv``; ``level`` is a
piecewise-constant multiplier that jumps at Poisson times. ``weekday`` is a
shop-wide day-of-week profile with mean 1.

The ``hidden_regimes`` family (track T1) multiplies every product's mean by
one shop-wide regime multiplier, ``regime_mult[r_t]``, and scales the noise cv
by ``regime_cv[r_t]``. ``r_t`` is a hidden Markov chain over ``K = 3``
regimes that leaves its state with probability ``switch_prob`` per day, to
one of the others uniformly. The idiosyncratic per-product level shifts stay.
A ``regimes`` dict (from ``world_families.hidden_regimes.params`` in the
protocol) overrides :data:`REGIME_DEFAULTS`; ``strength`` scales the log
multipliers.

A ``variant`` removes the order fee as a source of difference between
policies. ``one_product`` keeps product 0 of the base world (same lag, cv,
holding rate, demand path); the fee then buys one product's order, so
ordering products together cannot save fees. ``no_fee`` keeps all four
products and sets the order fee to 0.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

N_PRODUCTS = 4
UNIT_COST = 10.0
PRICE = 20.0
ORDER_FEE = 50.0
EPISODE_DAYS = 365
# Days of demand the shop has already seen before day 0. Every policy reads
# the same history, which was sold from ample stock, so it is not censored.
HISTORY_DAYS = 56

CV_CHOICES = (0.1, 0.3, 0.6)
HOLDING_RATE_CHOICES = (0.25 / 365, 1 / 365, 0.01)
LAG_RANGE = (2, 7)  # inclusive
MEAN_RANGE = (5.0, 20.0)  # log-uniform per product
WEEKLY_AMP_RANGE = (0.0, 0.4)
SHIFT_RATE_PER_DAY = 1 / 60
SHIFT_LOG_SD = 0.3
LEVEL_CLIP = (0.5, 2.0)

# Seed blocks keep training, tuning, and evaluation worlds apart.
SPLIT_SEED_OFFSET = {"train": 1_000_000, "tune": 2_000_000, "eval": 3_000_000}

VARIANTS = ("base", "one_product", "no_fee")
FAMILIES = ("perfect_supplier", "hidden_regimes")

REGIME_DEFAULTS = {
    "mult": (0.6, 1.0, 1.6),
    "cv_scale": (1.2, 1.0, 0.8),
    "switch_prob": (1 / 60, 1 / 20),  # per day, log-uniform per world
    "strength": 1.0,
}


@dataclass(frozen=True)
class World:
    world_seed: int
    split: str
    n_days: int
    delivery_lag: int
    holding_rate: float
    cv: float
    weekly_amp: float
    weekly_phase: float
    base_mean: tuple[float, ...]
    initial_on_hand: tuple[int, ...]
    initial_cash: float
    capacity: int
    unit_cost: float = UNIT_COST
    price: float = PRICE
    order_fee: float = ORDER_FEE
    variant: str = "base"
    family: str = "perfect_supplier"
    regime_mult: tuple[float, ...] = ()
    regime_cv: tuple[float, ...] = ()
    switch_prob: float = 0.0

    @property
    def n_products(self) -> int:
        return len(self.base_mean)

    def params(self) -> dict:
        """Flat scalar description, one column per field, for the summary frame."""
        out = {k: v for k, v in asdict(self).items() if not isinstance(v, tuple)}
        for p, (mu, inv) in enumerate(zip(self.base_mean, self.initial_on_hand)):
            out[f"base_mean_p{p}"] = mu
            out[f"initial_on_hand_p{p}"] = inv
        return out


def world_seed(split: str, index: int) -> int:
    return SPLIT_SEED_OFFSET[split] + int(index)


def sample_world(
    seed: int,
    split: str = "train",
    n_days: int = EPISODE_DAYS,
    variant: str = "base",
    family: str = "perfect_supplier",
    regimes: dict | None = None,
) -> World:
    """Draw the economics and opening state of one world from ``seed``.

    Every variant draws the base world's random numbers, so a seed gives the
    same lag, cv, holding rate, and product-0 demand in every variant. The
    ``hidden_regimes`` family draws them too and adds its regime parameters.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    if family not in FAMILIES:
        raise ValueError(f"unknown world family {family!r}")
    rng = np.random.default_rng([seed, 0])
    lag = int(rng.integers(LAG_RANGE[0], LAG_RANGE[1] + 1))
    lo, hi = np.log(MEAN_RANGE)
    means = np.exp(rng.uniform(lo, hi, N_PRODUCTS))
    if variant == "one_product":
        means = means[:1]
    total = float(means.sum())
    # Opening stock covers about one delivery lag; the shelf holds about
    # (2 * lag + 14) days of total mean demand; opening cash buys one
    # reorder of the pipeline (lag + 7 days of total mean demand).
    on_hand = tuple(int(round(m * lag)) for m in means)
    capacity = int(round((2 * lag + 14) * total))
    cash = UNIT_COST * (lag + 7) * total
    regime: dict = {}
    if family == "hidden_regimes":
        spec = {**REGIME_DEFAULTS, **(regimes or {})}
        mult = tuple(float(m) ** float(spec["strength"]) for m in spec["mult"])
        lo_p, hi_p = np.log(spec["switch_prob"])
        regime = {
            "regime_mult": mult,
            "regime_cv": tuple(float(c) for c in spec["cv_scale"]),
            "switch_prob": float(np.exp(np.random.default_rng([seed, 3]).uniform(lo_p, hi_p))),
        }
        # Shelf and opening cash scale with the busiest regime so neither binds.
        capacity = int(round(capacity * max(mult)))
        cash *= max(mult)
    return World(
        world_seed=int(seed),
        split=split,
        n_days=int(n_days),
        delivery_lag=lag,
        holding_rate=float(rng.choice(HOLDING_RATE_CHOICES)),
        cv=float(rng.choice(CV_CHOICES)),
        weekly_amp=float(rng.uniform(*WEEKLY_AMP_RANGE)),
        weekly_phase=float(rng.uniform(0, 2 * np.pi)),
        base_mean=tuple(float(m) for m in means),
        initial_on_hand=on_hand,
        initial_cash=float(cash),
        capacity=capacity,
        order_fee=0.0 if variant == "no_fee" else ORDER_FEE,
        variant=variant,
        family=family,
        **regime,
    )


def regime_path(world: World) -> np.ndarray:
    """Hidden regime index per day, shape ``(HISTORY_DAYS + n_days,)``; zeros outside ``hidden_regimes``."""
    total_days = HISTORY_DAYS + world.n_days
    K = len(world.regime_mult)
    if K == 0:
        return np.zeros(total_days, dtype=np.int64)
    rng = np.random.default_rng([world.world_seed, 4])
    out = np.empty(total_days, dtype=np.int64)
    current = int(rng.integers(K))
    for t in range(total_days):
        if t and rng.random() < world.switch_prob:
            current = (current + 1 + int(rng.integers(K - 1))) % K
        out[t] = current
    return out


def regime_transition(world: World) -> np.ndarray:
    """``(K, K)`` regime transition matrix."""
    K, p = len(world.regime_mult), world.switch_prob
    return (1 - p) * np.eye(K) + p / (K - 1) * (1 - np.eye(K))


def weekday_profile(world: World) -> np.ndarray:
    k = np.arange(7)
    return 1.0 + world.weekly_amp * np.sin(2 * np.pi * k / 7 + world.weekly_phase)


def demand_mean_path(world: World) -> np.ndarray:
    """Expected demand, shape ``(HISTORY_DAYS + n_days, n_products)``.

    Row ``HISTORY_DAYS`` is episode day 0. This is the oracle forecast.
    """
    mean = base_mean_path(world)
    if world.regime_mult:
        mean = mean * np.asarray(world.regime_mult)[regime_path(world)][:, None]
    return mean


def base_mean_path(world: World) -> np.ndarray:
    """:func:`demand_mean_path` without the regime multiplier."""
    rng = np.random.default_rng([world.world_seed, 1])
    total_days = HISTORY_DAYS + world.n_days
    # Levels for all base products are drawn so product p's path does not
    # depend on how many products the variant keeps.
    level = np.ones((total_days, N_PRODUCTS))
    for p in range(N_PRODUCTS):
        current = 1.0
        for t in range(total_days):
            if rng.random() < SHIFT_RATE_PER_DAY:
                current = float(np.clip(current * np.exp(rng.normal(0, SHIFT_LOG_SD)), *LEVEL_CLIP))
            level[t, p] = current
    # Day index counts from the start of the history so the weekday of
    # episode day t is (HISTORY_DAYS + t) % 7.
    weekday = weekday_profile(world)[np.arange(total_days) % 7]
    return np.asarray(world.base_mean)[None, :] * level[:, : world.n_products] * weekday[:, None]


def uniform_half_width(cv: float) -> float:
    """Half-width of a mean-1 uniform with coefficient of variation ``cv``."""
    return cv * np.sqrt(3.0)


def noise_half_width(world: World) -> np.ndarray:
    """Half-width of the mean-1 uniform noise per day, shape ``(HISTORY_DAYS + n_days,)``."""
    a = np.full(HISTORY_DAYS + world.n_days, uniform_half_width(world.cv))
    if world.regime_cv:
        a = a * np.asarray(world.regime_cv)[regime_path(world)]
    return a


def demand_path(world: World) -> np.ndarray:
    """Integer demand, shape ``(HISTORY_DAYS + n_days, n_products)``."""
    mean = demand_mean_path(world)
    rng = np.random.default_rng([world.world_seed, 2])
    a = noise_half_width(world)[:, None]
    # Unit uniforms scaled per day, so the base family draws the same numbers as before.
    u = 1 + a * rng.uniform(-1, 1, (mean.shape[0], N_PRODUCTS))[:, : world.n_products]
    return np.rint(np.clip(mean * u, 0, None)).astype(np.int64)
