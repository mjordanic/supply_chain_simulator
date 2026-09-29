"""Tuning-side episode sampler: builds a graph-mode 3-node episode.

``sample_episode(catalog, base_template, config, episode_seed) → TuningEpisodeSpec``
builds a degenerate single-store tuning episode as a 3-node graph:

    FactoryNode("F_<pid>") → IntermediateNode("S") → DemandSinkNode("D_<pid>")

The intermediate node "S" is the node under policy optimisation.

``TuningEpisodeSpec`` is the canonical tuning episode descriptor. It carries
the graph-mode ``Scenario`` plus the per-episode ``capacity`` and ``balance``
so ``evaluator.py`` can normalise returns without inspecting ``stores[0]``.

``load_catalog_and_market_from_setup`` and ``make_synthetic_catalog`` are the
setup-directory helpers used when ``TuningConfig.setup_dir`` is set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from random import Random
from typing import TYPE_CHECKING, Any

from src.sim.episode_sampler import (
    _derive_seed as _sim_derive_seed,
    default_disruption_params,
    default_lifecycle_params,
    default_market_params,
    sample_episode_lifecycle,
)
from src.sim.scenario import (
    CANONICAL_STAGES,
    DisruptionParams,
    ItemLifecycleParams,
    MarketParams,
    Scenario,
    Ware,
    load_catalog,
)

if TYPE_CHECKING:
    from src.tuning.config import TuningConfig


# ---------------------------------------------------------------------------
# Public dataclass: TuningEpisodeSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TuningEpisodeSpec:
    """Complete, self-contained description of one tuning episode.

    Fields
    ------
    scenario
        A graph-mode ``Scenario`` with a 3-node graph
        (FactoryNode "F_<pid>" → IntermediateNode "S" → DemandSinkNode "D_<pid>").
    active_subset
        The K product ids selected for this episode, in catalog order.
    capacity
        The IntermediateNode capacity for this episode (int).
    balance
        The IntermediateNode opening cash for this episode (float).
    """

    scenario: Scenario
    active_subset: tuple[str, ...]
    capacity: int
    balance: float


# ---------------------------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------------------------

# The historical name in this module; also re-exported from sim for compatibility.
EpisodeSpec = TuningEpisodeSpec


# ---------------------------------------------------------------------------
# Graph-building helper
# ---------------------------------------------------------------------------


def _build_tuning_graph_scenario(
    *,
    catalog: list[Ware],
    active_subset: tuple[str, ...],
    capacity: int,
    balance: float,
    world_seed: int,
    episode_length: int,
    delivery_lag: int,
    holding_rate: float,
    order_fee: float,
    init_seed: int,
    init_stock_pct: float = 0.0,
    market_params: MarketParams | None = None,
    disruption_params: DisruptionParams | None = None,
    lifecycle_params: ItemLifecycleParams | None = None,
    start_date: datetime | None = None,
) -> Scenario:
    """Build a graph-mode Scenario for a single-policy tuning episode.

    The graph has exactly 3 node types:
      - FactoryNode("F_<pid>") — one per active product
      - IntermediateNode("S") — the node under policy evaluation
      - DemandSinkNode("D_<pid>") — one per active product

    Edges:
      F_<pid> → S (with default_lead_time = delivery_lag)
      S → D_<pid>  (with default_lead_time = 1 — intra-day)
    """
    from src.sim.distributions import Constant
    from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode
    from src.sim.scenario import NodeInstance
    from src.sim.graph import EdgeSpec

    pid_to_ware: dict[str, Ware] = {w.product_id: w for w in catalog}

    node_instances: list[Any] = []
    edge_specs: list[Any] = []

    # One factory per active product.
    for i, pid in enumerate(active_subset):
        ware = pid_to_ware[pid]
        factory_id = f"F_{pid}"
        factory = FactoryNode(
            id=factory_id,
            region="US",
            init_seed=init_seed + i,
            produces_product_id=pid,
            unit_cost=float(ware.unit_cost),
            capacity_per_tick=Constant(capacity * 4),  # ample production
            inventory=capacity * 2,  # pre-stocked at start
            list_price=float(ware.unit_cost),
            cash=0.0,
        )
        node_instances.append(NodeInstance(node=factory, init_seed=init_seed + i, policy=None))

    # One intermediate node "S" carrying all active products.
    # Use init_stock_pct to determine opening inventory (same as the old StoreTemplate path).
    total_init_stock = int(capacity * init_stock_pct)
    inv_per_product = total_init_stock // max(1, len(active_subset))
    initial_inventory = {pid: inv_per_product for pid in active_subset}
    list_prices = {pid: float(pid_to_ware[pid].base_price) for pid in active_subset}
    min_order_imposed = {pid: 0 for pid in active_subset}

    intermediate = IntermediateNode(
        id="S",
        region="US",
        init_seed=init_seed + len(active_subset),
        carried_products=set(active_subset),
        capacity=capacity,
        tags=["shop"],
        inventory=initial_inventory,
        pending={},
        list_prices=list_prices,
        min_order_imposed=min_order_imposed,
        cash=balance,
        holding_rate=holding_rate,
        order_fee=order_fee,
    )
    node_instances.append(NodeInstance(node=intermediate, init_seed=init_seed + len(active_subset), policy=None))

    # One demand sink per active product.
    for j, pid in enumerate(active_subset):
        sink = DemandSinkNode(
            id=f"D_{pid}",
            region="US",
            init_seed=init_seed + len(active_subset) + 1 + j,
            product_id=pid,
            demand_dist=_sink_demand_dist(pid_to_ware[pid]),
            income_rate=float(pid_to_ware[pid].base_price) * 10.0,  # ample income
            cash=balance,
            activation_tick={pid: 0},
        )
        node_instances.append(NodeInstance(node=sink, init_seed=init_seed + len(active_subset) + 1 + j, policy=None))

    # Build edges: factories → intermediate, intermediate → sinks.
    for pid in active_subset:
        edge_specs.append(EdgeSpec(
            supplier_id=f"F_{pid}",
            buyer_id="S",
            default_lead_time=delivery_lag,
        ))
        edge_specs.append(EdgeSpec(
            supplier_id="S",
            buyer_id=f"D_{pid}",
            default_lead_time=1,
        ))

    return Scenario(
        catalog=catalog,
        market=market_params if market_params is not None else default_market_params(),
        disruption=(
            disruption_params if disruption_params is not None else default_disruption_params()
        ),
        item_lifecycle=(
            lifecycle_params if lifecycle_params is not None else default_lifecycle_params()
        ),
        nodes=node_instances,
        edges=edge_specs,
        n_steps=episode_length,
        start_date=start_date if start_date is not None else datetime(2024, 1, 1),
        world_seed=world_seed,
    )


def _sink_demand_dist(ware: Ware):
    """Per-SKU demand. Missing scale fields keep the historical ``Uniform(2, 8)``."""
    from src.sim.distributions import Uniform

    mean = ware.demand_mean
    cv = ware.demand_cv
    if mean is None or cv is None:
        return Uniform(2, 8)
    half = float(mean) * float(cv) * math.sqrt(3.0)
    low = float(mean) - half
    high = float(mean) + half
    if low < 1e-3:
        low = 1e-3
    if high <= low:
        high = low + 1e-3
    return Uniform(low, high)


# Same multiply-add as the RL sampler's ``k`` stream.
_K_PRIME = 0x27D4_EB2F
_K_OFFSET = 0x0000_0009
_MASK_32 = 0xFFFF_FFFF


def _sample_k(config: "TuningConfig", episode_seed: int, n_catalog: int) -> int:
    """Pinned ``K_active``, or a draw from ``[K_min, K_max_episode]``."""
    if config.K_active is not None:
        return int(config.K_active)
    if n_catalog < 1:
        raise ValueError("sample_episode: catalog is empty")
    k_hi = min(int(config.K_max_episode), n_catalog)
    k_lo = min(max(int(config.K_min), 1), k_hi)
    seed = (episode_seed * _K_PRIME + _K_OFFSET) & _MASK_32
    return Random(seed).randint(k_lo, k_hi)


# ---------------------------------------------------------------------------
# Public function: sample_episode
# ---------------------------------------------------------------------------


def sample_episode(
    catalog: list[Ware],
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
        The full product universe. Must be at least as long as the pinned
        ``K_active``, or as ``K_min`` when K is sampled.
    config:
        ``TuningConfig`` instance. ``K_active`` pins the assortment; when it
        is ``None``, K is drawn from ``[K_min, K_max_episode]``. Also reads
        ``capacity_dist``, ``balance_dist``, ``delivery_lag``,
        ``holding_rate``, ``order_fee``, and ``episode_length``.
    episode_seed:
        Single integer seed. Any two calls with the same seed and the same
        ``(catalog, config)`` produce identical ``TuningEpisodeSpec``
        objects — the function is pure.
    market_params, disruption_params, lifecycle_params:
        Optional overrides. Market and disruption fall back to the default
        factories when ``None``. Lifecycle ``None`` samples per-SKU initial
        stages and a stage-change table from the episode seed; pass
        ``default_lifecycle_params()`` to keep the inert lifecycle.
    start_date:
        Episode start date; defaults to 2024-01-01.

    Returns
    -------
    TuningEpisodeSpec
        Fully deterministic description of the episode.
    """
    # --- derive independent sub-seeds ---
    assortment_seed = _sim_derive_seed(episode_seed, "assortment")
    capacity_seed = _sim_derive_seed(episode_seed, "capacity")
    balance_seed = _sim_derive_seed(episode_seed, "balance")
    world_seed = _sim_derive_seed(episode_seed, "world")

    # --- sample capacity and balance ---
    capacity_rng = Random(capacity_seed)
    capacity = int(config.capacity_dist.sample(capacity_rng))

    balance_rng = Random(balance_seed)
    balance = float(config.balance_dist.sample(balance_rng))

    # --- sample init_stock_pct if distribution provided ---
    init_stock_pct_dist = getattr(config, "init_stock_pct_dist", None)
    if init_stock_pct_dist is not None:
        init_stock_seed = _sim_derive_seed(episode_seed, "init_stock")
        init_stock_pct = float(init_stock_pct_dist.sample(Random(init_stock_seed)))
    else:
        init_stock_pct = 0.0

    # Lifecycle is part of the episode draw unless the caller pinned params.
    # The input catalog is left as authored.
    if lifecycle_params is None:
        lifecycle_params, catalog = sample_episode_lifecycle(catalog, episode_seed)

    # --- sample active subset (without replacement) ---
    assortment_rng = Random(assortment_seed)
    all_pids = [w.product_id for w in catalog]
    K = _sample_k(config, episode_seed, len(all_pids))
    if K > len(all_pids):
        raise ValueError(
            f"sample_episode: K={K} exceeds catalog size {len(all_pids)}"
        )
    active_pids = assortment_rng.sample(all_pids, K)
    # Sort to canonical catalog order so the tuple is stable.
    pid_order = {pid: i for i, pid in enumerate(all_pids)}
    active_subset: tuple[str, ...] = tuple(
        sorted(active_pids, key=lambda p: pid_order[p])
    )

    # --- build the graph-mode Scenario ---
    scenario = _build_tuning_graph_scenario(
        catalog=catalog,
        active_subset=active_subset,
        capacity=capacity,
        balance=balance,
        world_seed=world_seed,
        episode_length=config.episode_length,
        delivery_lag=config.delivery_lag,
        holding_rate=config.holding_rate,
        order_fee=config.order_fee,
        init_seed=assortment_seed,
        init_stock_pct=init_stock_pct,
        market_params=market_params,
        disruption_params=disruption_params,
        lifecycle_params=lifecycle_params,
        start_date=start_date,
    )

    return TuningEpisodeSpec(
        scenario=scenario,
        active_subset=active_subset,
        capacity=capacity,
        balance=balance,
    )


# ---------------------------------------------------------------------------
# Setup-directory helpers
# ---------------------------------------------------------------------------


def load_catalog_and_market_from_setup(
    setup_dir: "str | Any",
) -> "tuple[list[Ware], MarketParams]":
    """Load catalog + market params from a setup directory.

    Thin wrapper around ``setup_io._parse_catalog`` and
    ``setup_io._parse_market`` so the tuning stack can consume a setup
    directory without pulling in the full ``load_setup`` pipeline.

    Parameters
    ----------
    setup_dir:
        Path to a directory containing ``catalog.csv`` and ``setup.yaml``
        with at least a ``market:`` block.

    Returns
    -------
    (catalog, market_params)
    """
    from pathlib import Path as _Path

    import yaml as _yaml

    from src.sim.setup_io import _parse_catalog, _parse_market

    setup_dir = _Path(setup_dir)
    catalog_path = setup_dir / "catalog.csv"
    yaml_path = setup_dir / "setup.yaml"

    if not catalog_path.is_file():
        raise FileNotFoundError(
            f"load_catalog_and_market_from_setup: catalog.csv not found in {setup_dir}"
        )
    if not yaml_path.is_file():
        raise FileNotFoundError(
            f"load_catalog_and_market_from_setup: setup.yaml not found in {setup_dir}"
        )

    catalog = _parse_catalog(catalog_path)

    with yaml_path.open(encoding="utf-8") as f:
        doc = _yaml.safe_load(f) or {}

    if "market" not in doc:
        raise ValueError(
            f"load_catalog_and_market_from_setup: setup.yaml in {setup_dir} "
            "has no 'market:' block"
        )

    market = _parse_market(doc["market"], "setup.yaml.market")
    return catalog, market


def make_synthetic_catalog(n: int | None = None) -> "list[Ware]":
    """Build a synthetic catalog for CI / smoke runs (no LLM).

    The default size is ``SYNTHETIC_CATALOG_SIZE`` (inside 500–1000).
    SKUs differ in demand scale, margin, demand volatility, and lifecycle
    transition rate. Ids stay ``P{i:04d}``. Seasonality stays
    ``all_season`` so the demand spread is the SKU's own distribution.
    """
    from src.tuning.config import SYNTHETIC_CATALOG_SIZE

    if n is None:
        n = SYNTHETIC_CATALOG_SIZE
    span = max(1, n - 1)
    stages = CANONICAL_STAGES
    base_rates = {
        "introduction": 0.012,
        "growth": 0.012,
        "maturity": 0.01,
        "decline": 0.015,
        "dead": 0.003,
    }
    items = []
    for i in range(n):
        demand_mean = 0.5 * (80 ** (i / span))
        demand_cv = 0.05 + 0.45 * (((i * 17) % n) / span)
        margin = 0.08 + 0.67 * (((i * 13) % n) / span)
        tempo = 0.25 + 3.75 * (((i * 19) % n) / span)
        unit_cost = 4.0 + (i % 11)
        items.append(
            {
                "name": f"Product {i}",
                "category": "General",
                "related_products": [],
                "base_price": float(unit_cost / (1.0 - margin)),
                "unit_cost": float(unit_cost),
                "seasonality": "all_season",
                "stage_change_probs": {
                    stage: base_rates[stage] * tempo for stage in stages
                },
                "demand_mean": float(demand_mean),
                "demand_cv": float(demand_cv),
            }
        )
    return load_catalog(items)


__all__ = [
    "TuningEpisodeSpec",
    "EpisodeSpec",
    "sample_episode",
    "load_catalog_and_market_from_setup",
    "make_synthetic_catalog",
]
