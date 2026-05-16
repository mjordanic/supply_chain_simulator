"""Pure-function episode sampler for tuning studies.

``sample_episode(catalog, base_template, config, episode_seed) → TuningEpisodeSpec``
takes a product catalog, a base ``StoreTemplate``, a ``TuningConfig``, and a
single integer seed, then deterministically produces a fully-specified
``TuningEpisodeSpec``.

Seed splitting strategy
-----------------------
The single ``episode_seed`` is fanned into four independent sub-seeds:

  - ``assortment_seed`` — which K products are active this episode
  - ``capacity_seed``   — store capacity draw from ``config.capacity_dist``
  - ``balance_seed``    — opening cash draw from ``config.balance_dist``
  - ``world_seed``      — ``Scenario.world_seed`` (drives all world-side RNG)

Sub-seeds are derived deterministically via a lightweight hash so any subset
can be frozen for ablations without disturbing the others. No I/O, no
module-level mutable state, no global RNG.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from random import Random
from typing import TYPE_CHECKING

from src.sim.scenario import (
    DisruptionParams,
    ItemLifecycleParams,
    MarketParams,
    Scenario,
    StoreInstance,
    StoreTemplate,
    Ware,
)

if TYPE_CHECKING:
    from src.tuning.config import TuningConfig


@dataclass(frozen=True)
class TuningEpisodeSpec:
    """Complete, self-contained description of one tuning episode.

    Fields
    ------
    scenario
        Fully-initialised ``Scenario`` with ``n_steps == config.episode_length``,
        a single ``StoreInstance`` carrying the episode's capacity / balance /
        active-assortment, and ``world_seed`` derived from ``episode_seed``.
    active_subset
        The K product ids selected for this episode, in catalog order.
        Retained so per-tick KPI traces can iterate the active SKUs in a
        stable order.
    """

    scenario: Scenario
    active_subset: tuple[str, ...]


_SUB_SEED_PARAMS: dict[str, tuple[int, int]] = {
    "assortment": (0x9E37_79B9, 0x0000_0001),
    "capacity":   (0x6C62_272E, 0x0000_0002),
    "balance":    (0x517C_C1B7, 0x0000_0003),
    "world":      (0x27D4_EB2F, 0x0000_0004),
}
_MASK_32 = 0xFFFF_FFFF


def _derive_seed(episode_seed: int, purpose: str) -> int:
    prime, offset = _SUB_SEED_PARAMS[purpose]
    return (episode_seed * prime + offset) & _MASK_32


def _default_market_params() -> MarketParams:
    """Return a sensible ``MarketParams`` for tuning episodes when the catalog
    does not carry an authored one. Kept simple so signal is dominated by
    pricing / ordering decisions, not exotic market dynamics."""
    from src.sim.distributions import Constant, Normal, Uniform

    return MarketParams(
        cycle_len=365,
        cycle_amp=0.3,
        init_demand=1.0,
        init_supply=1.0,
        peak_factor=1.2,
        off_factor=0.7,
        season_months={"all_season": list(range(1, 13))},
        regions=["US"],
        correlation=0.7,
        trend_update_interval=20,
        min_value=0.2,
        max_value=2.0,
        stage_multipliers={
            "introduction": 0.7,
            "growth": 1.5,
            "maturity": 1.0,
            "decline": 0.2,
            "dead": 0.05,
        },
        price_elasticity=-1.5,
        promo_multiplier=1.0,
        demand_factor_min=0.1,
        supply_factor_min=0.01,
        cross_inv_lo=0.3,
        cross_inv_hi=0.7,
        cross_factor_range=(0.3, 1.6),
        trend=Constant(1.0),
        demand_shock=Normal(0.0, 0.01),
        supply_shock=Normal(0.0, 0.01),
        base_demand=Uniform(2, 8),
    )


def _default_disruption_params() -> DisruptionParams:
    from src.sim.distributions import Constant

    return DisruptionParams(
        event_prob=0.05,
        types=["natural_disaster", "economic_crisis"],
        regions=["US"],
        severity=Constant(0.01),
        duration=Constant(3),
    )


def _default_lifecycle_params() -> ItemLifecycleParams:
    stages = ["introduction", "growth", "maturity", "decline", "dead"]
    return ItemLifecycleParams(
        stages=stages,
        init_stage="maturity",
        default_stage_change_probs={s: 0.0 for s in stages},
    )


def sample_episode(
    catalog: list[Ware],
    base_template: StoreTemplate,
    config: "TuningConfig",
    episode_seed: int,
    *,
    market_params: MarketParams | None = None,
    disruption_params: DisruptionParams | None = None,
    lifecycle_params: ItemLifecycleParams | None = None,
    start_date: datetime | None = None,
) -> TuningEpisodeSpec:
    """Deterministically sample a fully-specified ``TuningEpisodeSpec``.

    Parameters
    ----------
    catalog:
        Full product universe (length >= ``config.K_active``).
    base_template:
        ``StoreTemplate`` carrying non-episodic knobs. The episodic fields
        (capacity, init_balance, init_active_products, init_stock_pct,
        init_freshness) are always overridden by ``sample_episode``.
    config:
        ``TuningConfig`` instance; only ``K_active``, ``capacity_dist``,
        ``balance_dist``, and ``episode_length`` are consumed here.
    episode_seed:
        Single integer seed. Any two calls with the same seed and the same
        ``(catalog, base_template, config)`` produce identical specs.
    market_params, disruption_params, lifecycle_params:
        Optional overrides; fall back to sensible defaults when ``None``.
    start_date:
        Episode start date; defaults to 2024-01-01.
    """
    assortment_seed = _derive_seed(episode_seed, "assortment")
    capacity_seed = _derive_seed(episode_seed, "capacity")
    balance_seed = _derive_seed(episode_seed, "balance")
    world_seed = _derive_seed(episode_seed, "world")

    capacity_rng = Random(capacity_seed)
    capacity = config.capacity_dist.sample(capacity_rng)

    balance_rng = Random(balance_seed)
    balance = config.balance_dist.sample(balance_rng)

    assortment_rng = Random(assortment_seed)
    all_pids = [w.product_id for w in catalog]
    K = config.K_active
    if K > len(all_pids):
        raise ValueError(
            f"sample_episode: K_active={K} exceeds catalog size {len(all_pids)}"
        )
    active_pids = assortment_rng.sample(all_pids, K)
    pid_order = {pid: i for i, pid in enumerate(all_pids)}
    active_subset: tuple[str, ...] = tuple(
        sorted(active_pids, key=lambda p: pid_order[p])
    )

    episode_template = StoreTemplate(
        id=base_template.id,
        region=base_template.region,
        capacity=int(capacity),
        init_balance=float(balance),
        init_stock_pct=0.0,
        delivery_lag=base_template.delivery_lag,
        holding_rate=base_template.holding_rate,
        order_fee=base_template.order_fee,
        init_active_count=K,
        init_active_products=list(active_subset),
        init_freshness="fresh",
    )

    store_instance = StoreInstance(
        template=episode_template,
        init_seed=assortment_seed,
        policy=None,
    )

    scenario = Scenario(
        catalog=catalog,
        market=market_params if market_params is not None else _default_market_params(),
        disruption=(
            disruption_params
            if disruption_params is not None
            else _default_disruption_params()
        ),
        item_lifecycle=(
            lifecycle_params
            if lifecycle_params is not None
            else _default_lifecycle_params()
        ),
        stores=[store_instance],
        n_steps=config.episode_length,
        start_date=start_date if start_date is not None else datetime(2024, 1, 1),
        world_seed=world_seed,
    )

    return TuningEpisodeSpec(
        scenario=scenario,
        active_subset=active_subset,
    )


__all__ = ["TuningEpisodeSpec", "sample_episode"]
