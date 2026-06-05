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

from dataclasses import dataclass
from datetime import datetime
from random import Random
from typing import TYPE_CHECKING, Any

from src.sim.episode_sampler import (
    _derive_seed as _sim_derive_seed,
    default_disruption_params,
    default_lifecycle_params,
    default_market_params,
)
from src.sim.scenario import (
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
    from src.sim.distributions import Constant, Uniform
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
    )
    node_instances.append(NodeInstance(node=intermediate, init_seed=init_seed + len(active_subset), policy=None))

    # One demand sink per active product.
    for j, pid in enumerate(active_subset):
        sink = DemandSinkNode(
            id=f"D_{pid}",
            region="US",
            init_seed=init_seed + len(active_subset) + 1 + j,
            product_id=pid,
            demand_dist=Uniform(2, 8),
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
        stores=[],  # graph-mode scenario: no legacy stores
        nodes=node_instances,
        edges=edge_specs,
        n_steps=episode_length,
        start_date=start_date if start_date is not None else datetime(2024, 1, 1),
        world_seed=world_seed,
    )


# ---------------------------------------------------------------------------
# Public function: sample_episode
# ---------------------------------------------------------------------------


def sample_episode(
    catalog: list[Ware],
    base_template: Any,
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
        The full product universe (length >= ``config.K_active``).
    base_template:
        A ``StoreTemplate`` carrying non-episodic knobs (region, delivery
        lag, holding rate, order fee). Episodic fields (capacity, balance)
        are overridden by this sampler from ``config``.
    config:
        ``TuningConfig`` instance; ``K_active``, ``capacity_dist``,
        ``balance_dist``, ``delivery_lag``, ``holding_rate``, ``order_fee``,
        and ``episode_length`` are consumed here.
    episode_seed:
        Single integer seed. Any two calls with the same seed and the same
        ``(catalog, base_template, config)`` produce identical
        ``TuningEpisodeSpec`` objects — the function is pure.
    market_params, disruption_params, lifecycle_params:
        Optional overrides; fall back to sensible defaults when ``None``.
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

    # --- sample active subset (without replacement) ---
    assortment_rng = Random(assortment_seed)
    all_pids = [w.product_id for w in catalog]
    K = config.K_active
    if K > len(all_pids):
        raise ValueError(
            f"sample_episode: K_active={K} exceeds catalog size {len(all_pids)}"
        )
    active_pids = assortment_rng.sample(all_pids, K)
    # Sort to canonical catalog order so the tuple is stable.
    pid_order = {pid: i for i, pid in enumerate(all_pids)}
    active_subset: tuple[str, ...] = tuple(
        sorted(active_pids, key=lambda p: pid_order[p])
    )

    # --- build the graph-mode Scenario ---
    delivery_lag = getattr(base_template, "delivery_lag", config.delivery_lag)
    holding_rate = getattr(base_template, "holding_rate", config.holding_rate)
    order_fee = getattr(base_template, "order_fee", config.order_fee)

    scenario = _build_tuning_graph_scenario(
        catalog=catalog,
        active_subset=active_subset,
        capacity=capacity,
        balance=balance,
        world_seed=world_seed,
        episode_length=config.episode_length,
        delivery_lag=delivery_lag,
        holding_rate=holding_rate,
        order_fee=order_fee,
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


def make_synthetic_catalog(n: int = 100) -> "list[Ware]":
    """Build a synthetic n-product catalog for CI / smoke runs (no LLM).

    Products have stable P{i:04d} ids, consistent pricing, and
    ``all_season`` seasonality.
    """
    items = [
        {
            "name": f"Product {i}",
            "category": "General",
            "related_products": [],
            "base_price": float(10 + (i % 30)),
            "unit_cost": float(4 + (i % 10)),
            "seasonality": "all_season",
        }
        for i in range(n)
    ]
    return load_catalog(items)


__all__ = [
    "TuningEpisodeSpec",
    "EpisodeSpec",
    "sample_episode",
    "load_catalog_and_market_from_setup",
    "make_synthetic_catalog",
]
