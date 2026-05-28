"""Canonical episode sampler for the sim layer.

``sample_episode(catalog, base_template, *, K_active, episode_length,
                 capacity_dist, balance_dist, episode_seed, ...) → EpisodeSpec``
takes a product catalog, a base ``StoreTemplate``, and broken-out primitive
kwargs (no ``Config`` object), then deterministically produces a fully-specified
``EpisodeSpec``.

The sim layer is Config-agnostic — callers from tuning or RL supply their
own Config-adapter shims that unpack their respective config objects and
forward primitives here.

Seed splitting strategy
-----------------------
The single ``episode_seed`` is fanned into four independent sub-seeds so
any subset of randomisations can be frozen for ablations without disturbing
the others:

  - ``assortment_seed`` — which K products are active this episode
  - ``capacity_seed``   — store capacity draw from ``capacity_dist``
  - ``balance_seed``    — opening cash draw from ``balance_dist``
  - ``world_seed``      — ``Scenario.world_seed`` (drives all world-side RNG)

RL adds a fifth ``slot`` stream in ``src.rl.episode_sampler``; the slot
stream and ``slot_permutation`` are RL-observation-encoder concerns and
stay in the RL layer (see ADR 0004).

Sub-seeds are derived deterministically via a lightweight hash:
``sub = (episode_seed * PRIME + OFFSET) & 0xFFFF_FFFF`` where each
sub-purpose uses a different ``(PRIME, OFFSET)`` pair. No external
library required; the full derivation is unit-testable in one line.

No I/O, no module-level mutable state, no global RNG — the function is
safe to call from multiple threads or vector envs simultaneously.

The three ``default_*_params()`` factories are public and are consumed by:
- ``sample_episode``'s ``is None`` fallbacks.
- ``src.sim.world_loader``'s synthetic-fallback path (issue 05).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from random import Random
from typing import Any

from src.sim.scenario import (
    DisruptionParams,
    ItemLifecycleParams,
    MarketParams,
    Scenario,
    StoreInstance,
    StoreTemplate,
    Ware,
)


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeSpec:
    """Complete, self-contained description of one sim episode.

    Fields
    ------
    scenario
        A fully-initialised ``Scenario`` with ``n_steps == episode_length``,
        a single ``StoreInstance`` carrying the episode's capacity / balance /
        active-assortment, and ``world_seed`` derived from ``episode_seed``.
    active_subset
        The K product ids selected for this episode, in catalog order.
        Retained so per-tick KPI traces can iterate the active SKUs in a
        stable order.

    Note: ``slot_permutation`` is deliberately absent — it is an
    RL-observation-encoder concern (ADR 0004). RL callers wrap this in
    ``RLEpisodeSpec(spec, slot_permutation)`` defined in
    ``src.rl.episode_sampler``.
    """

    scenario: Scenario
    active_subset: tuple[str, ...]


# ---------------------------------------------------------------------------
# Sub-seed derivation
# ---------------------------------------------------------------------------

# Sub-purposes; RL adds "slot" as its own in its own module.
# "allocation" (issue 03) drives the per-phase buyer shuffle via
# allocation_rng = Random(_derive_seed(world_seed, "allocation")).
# Not yet consumed — wired up in issue 05 (GraphSimulation).
_SUB_SEED_PARAMS: dict[str, tuple[int, int]] = {
    "assortment": (0x9E37_79B9, 0x0000_0001),
    "capacity":   (0x6C62_272E, 0x0000_0002),
    "balance":    (0x517C_C1B7, 0x0000_0003),
    "world":      (0x27D4_EB2F, 0x0000_0004),
    "init_stock": (0x85EB_CA6B, 0x0000_0006),
    "allocation": (0xA24B_AED4, 0x0000_0007),
}
_MASK_32 = 0xFFFF_FFFF


def _derive_seed(episode_seed: int, purpose: str) -> int:
    """Return a deterministic 32-bit sub-seed for ``purpose``.

    The derivation is a single multiply-add-mask step; all four purposes
    produce different values for any non-degenerate ``episode_seed``.
    """
    prime, offset = _SUB_SEED_PARAMS[purpose]
    return (episode_seed * prime + offset) & _MASK_32


# ---------------------------------------------------------------------------
# Default Scenario parameters (synthetic-fallback defaults)
# ---------------------------------------------------------------------------


def default_market_params() -> MarketParams:
    """Return a sensible ``MarketParams`` for synthetic/fallback episodes.

    All fields are kept simple (scalar or low-variance distributions) so
    the training signal is dominated by the store's pricing / ordering
    decisions, not by exotic market dynamics.
    """
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


def default_disruption_params() -> DisruptionParams:
    """Return default ``DisruptionParams`` (event_prob=0.05)."""
    from src.sim.distributions import Constant

    return DisruptionParams(
        event_prob=0.05,
        types=["natural_disaster", "economic_crisis"],
        regions=["US"],
        severity=Constant(0.01),
        duration=Constant(3),
    )


def default_lifecycle_params() -> ItemLifecycleParams:
    """Return default ``ItemLifecycleParams`` (maturity, zero-transition)."""
    stages = ["introduction", "growth", "maturity", "decline", "dead"]
    return ItemLifecycleParams(
        stages=stages,
        init_stage="maturity",
        default_stage_change_probs={s: 0.0 for s in stages},
    )


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def sample_episode(
    catalog: list[Ware],
    base_template: StoreTemplate,
    *,
    K_active: int,
    episode_length: int,
    capacity_dist: Any,
    balance_dist: Any,
    episode_seed: int,
    init_stock_pct_dist: Any | None = None,
    market_params: MarketParams | None = None,
    disruption_params: DisruptionParams | None = None,
    lifecycle_params: ItemLifecycleParams | None = None,
    start_date: datetime | None = None,
) -> EpisodeSpec:
    """Deterministically sample a fully-specified ``EpisodeSpec``.

    Parameters
    ----------
    catalog:
        The full product universe (length >= ``K_active``).  Must
        be built via ``load_catalog`` so product ids follow the ``P{i:04d}``
        convention.
    base_template:
        A ``StoreTemplate`` carrying the non-episodic knobs (region, delivery
        lag, holding rate, order fee, …).  The episodic fields ``capacity``,
        ``init_balance``, ``init_active_products``, ``init_stock_pct``, and
        ``init_freshness`` are **always overridden** by ``sample_episode`` and
        are ignored on the incoming template.
    K_active:
        Number of active SKUs per episode.
    episode_length:
        Number of ticks per episode (``Scenario.n_steps``).
    capacity_dist:
        Any object with a ``sample(rng) -> numeric`` method. Typically a
        ``Distribution`` from ``src.sim.distributions``.
    balance_dist:
        Same as ``capacity_dist`` but for opening balance.
    episode_seed:
        Single integer seed.  Any two calls with the same seed and the same
        ``(catalog, base_template, K_active, ...)`` produce identical
        ``EpisodeSpec`` objects — the function is pure.
    market_params, disruption_params, lifecycle_params:
        Optional overrides; fall back to sensible defaults when ``None``.
    start_date:
        Episode start date; defaults to 2024-01-01.

    Returns
    -------
    EpisodeSpec
        Fully deterministic description of the episode.
    """
    # --- derive independent sub-seeds ---
    assortment_seed = _derive_seed(episode_seed, "assortment")
    capacity_seed = _derive_seed(episode_seed, "capacity")
    balance_seed = _derive_seed(episode_seed, "balance")
    world_seed = _derive_seed(episode_seed, "world")

    # --- sample capacity and balance ---
    capacity_rng = Random(capacity_seed)
    capacity = capacity_dist.sample(capacity_rng)

    balance_rng = Random(balance_seed)
    balance = balance_dist.sample(balance_rng)

    if init_stock_pct_dist is not None:
        init_stock_seed = _derive_seed(episode_seed, "init_stock")
        init_stock_pct = float(init_stock_pct_dist.sample(Random(init_stock_seed)))
    else:
        init_stock_pct = 0.0

    # --- sample active subset (without replacement) ---
    assortment_rng = Random(assortment_seed)
    all_pids = [w.product_id for w in catalog]
    K = K_active
    if K > len(all_pids):
        raise ValueError(
            f"sample_episode: K_active={K} exceeds catalog size {len(all_pids)}"
        )
    active_pids = assortment_rng.sample(all_pids, K)
    # Sort to canonical catalog order so the tuple is stable
    pid_order = {pid: i for i, pid in enumerate(all_pids)}
    active_subset: tuple[str, ...] = tuple(
        sorted(active_pids, key=lambda p: pid_order[p])
    )

    # --- build the episode's StoreTemplate (overriding episodic fields) ---
    episode_template = StoreTemplate(
        id=base_template.id,
        region=base_template.region,
        capacity=int(capacity),
        init_balance=float(balance),
        init_stock_pct=init_stock_pct,
        delivery_lag=base_template.delivery_lag,
        holding_rate=base_template.holding_rate,
        order_fee=base_template.order_fee,
        init_active_count=K,
        init_active_products=list(active_subset),
        init_freshness="fresh",
    )

    # --- build the Scenario ---
    # Use the full catalog so every product id is known to ItemRegistry;
    # only the K active products will be in store.active_items.
    store_instance = StoreInstance(
        template=episode_template,
        init_seed=assortment_seed,  # deterministic, distinct from world
        policy=None,
    )

    scenario = Scenario(
        catalog=catalog,
        market=market_params if market_params is not None else default_market_params(),
        disruption=(
            disruption_params
            if disruption_params is not None
            else default_disruption_params()
        ),
        item_lifecycle=(
            lifecycle_params
            if lifecycle_params is not None
            else default_lifecycle_params()
        ),
        stores=[store_instance],
        n_steps=episode_length,
        start_date=start_date if start_date is not None else datetime(2024, 1, 1),
        world_seed=world_seed,
    )

    return EpisodeSpec(
        scenario=scenario,
        active_subset=active_subset,
    )


__all__ = [
    "EpisodeSpec",
    "sample_episode",
    "default_market_params",
    "default_disruption_params",
    "default_lifecycle_params",
]
