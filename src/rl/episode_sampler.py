"""RL-side episode sampler: builds a 3-node graph episode + slot stream.

``sample_episode(catalog, base_template, config, episode_seed) → RLEpisodeSpec``
builds a degenerate single-store RL episode as a 3-node graph:

    FactoryNode("F") → IntermediateNode("S") → DemandSinkNode("D_<pid>")

The intermediate node "S" is the trainable node; the RL policy is attached
to it via ``policy_overrides_by_id={"S": RLIntermediatePolicy()}`` in ``RLEnv``.

``RLEpisodeSpec`` wraps the sim ``EpisodeSpec`` (which now carries a graph-mode
``Scenario``) plus the RL-specific ``slot_permutation``.

Seed splitting strategy
-----------------------
The single ``episode_seed`` is fanned into five independent sub-seeds:

  - ``assortment_seed`` — which K products are active (in sim)
  - ``capacity_seed``   — intermediate node capacity draw (in sim)
  - ``balance_seed``    — opening cash draw for intermediate (in sim)
  - ``world_seed``      — ``Scenario.world_seed`` (in sim)
  - ``slot_seed``       — slot-permutation of the active subset (RL-only)

Sub-seeds are derived deterministically via a lightweight hash.

No I/O, no module-level mutable state, no global RNG.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from random import Random
from typing import TYPE_CHECKING, Any

from src.sim.episode_sampler import (
    EpisodeSpec as _SimEpisodeSpec,
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
    from src.rl.configs.default import RLConfig


# ---------------------------------------------------------------------------
# Public dataclass: RLEpisodeSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RLEpisodeSpec:
    """Complete, self-contained description of one RL episode.

    Fields
    ------
    scenario
        A graph-mode ``Scenario`` with a 3-node graph
        (FactoryNode "F" → IntermediateNode "S" → DemandSinkNode "D_<pid>").
    active_subset
        The K product ids selected for this episode, in catalog order.
        K is sampled per episode from [K_min, K_max_episode] in the config.
    capacity
        The IntermediateNode capacity for this episode (int).
    balance
        The IntermediateNode opening cash for this episode (float).
    world_seed
        The world seed for the scenario (convenience pass-through).

    Note
    ----
    ``slot_permutation`` has been removed (ADR 0021). Permutation invariance
    is now structural via the shared-weight set actor-critic.
    """

    scenario: Scenario
    active_subset: tuple[str, ...]
    capacity: int
    balance: float
    world_seed: int


# ---------------------------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------------------------

# The historical name in this module; RL internal code used ``EpisodeSpec``.
# New code should use ``RLEpisodeSpec``.
EpisodeSpec = RLEpisodeSpec


# ---------------------------------------------------------------------------
# Sub-seed re-export (now just the 4 shared sim sub-seeds; slot stream retired)
# ---------------------------------------------------------------------------

# Re-export the shared params so test imports like
#   ``from src.rl.episode_sampler import _SUB_SEED_PARAMS``
# continue to work. The RL module no longer adds a "slot" stream.
from src.sim.episode_sampler import _SUB_SEED_PARAMS as _SIM_SUB_SEED_PARAMS

_SUB_SEED_PARAMS: dict[str, tuple[int, int]] = dict(_SIM_SUB_SEED_PARAMS)
# Also add a "k" sub-seed for per-episode K sampling.
_K_PRIME = 0x27D4EB2F
_K_OFFSET = 0x0000_0009
_MASK_32 = 0xFFFF_FFFF


def _derive_seed(episode_seed: int, purpose: str) -> int:
    """Return a deterministic 32-bit sub-seed for ``purpose``.

    Delegates to sim's ``_derive_seed`` for the four shared purposes;
    handles "k" (per-episode K sampling) locally.
    """
    if purpose in _SIM_SUB_SEED_PARAMS:
        return _sim_derive_seed(episode_seed, purpose)
    if purpose == "k":
        return (episode_seed * _K_PRIME + _K_OFFSET) & _MASK_32
    raise KeyError(f"Unknown sub-seed purpose: {purpose!r}")


# ---------------------------------------------------------------------------
# Graph-building helper
# ---------------------------------------------------------------------------


def _build_rl_graph_scenario(
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
    market_params: MarketParams | None = None,
    disruption_params: DisruptionParams | None = None,
    lifecycle_params: ItemLifecycleParams | None = None,
    start_date: datetime | None = None,
) -> Scenario:
    """Build a graph-mode Scenario for a single-product RL episode.

    The graph has exactly 3 nodes:
      - FactoryNode("F") — produces ``active_subset[0]`` (the primary SKU)
      - IntermediateNode("S") — the trainable node; carries all active products
      - DemandSinkNode("D_<pid>") — one sink per active product

    For the degenerate single-SKU case (K_active=1), there is 1 factory,
    1 intermediate, and 1 sink.  For K_active>1, there is still 1 factory
    per product, 1 shared intermediate, and 1 sink per product.

    Edges:
      F_<pid> → S (with default_lead_time = delivery_lag)
      S → D_<pid>  (with default_lead_time = 1 — intra-day)
    """
    from src.sim.distributions import Constant
    from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode
    from src.sim.scenario import NodeInstance
    from src.sim.graph import EdgeSpec

    # Resolve market params up front so node regions match the market's
    # region keys (a hardcoded "US" breaks setups with other region names).
    market = market_params if market_params is not None else default_market_params()
    region = market.regions[0]

    # Default disruption params target region "US"; retarget them to the
    # market's regions so events never hit a region absent from market_state.
    if disruption_params is None:
        disruption_params = replace(
            default_disruption_params(), regions=list(market.regions)
        )

    # Use first active pid's catalog entry to size the factory unit_cost.
    pid_to_ware: dict[str, Ware] = {w.product_id: w for w in catalog}

    nodes: list[Any] = []
    node_instances: list[Any] = []
    edge_specs: list[Any] = []

    # One factory per active product.
    for i, pid in enumerate(active_subset):
        ware = pid_to_ware[pid]
        factory_id = f"F_{pid}"
        factory = FactoryNode(
            id=factory_id,
            region=region,
            init_seed=init_seed + i,
            produces_product_id=pid,
            unit_cost=float(ware.unit_cost),
            capacity_per_tick=Constant(capacity * 4),  # ample production
            inventory=capacity * 2,  # pre-stocked at start
            list_price=float(ware.unit_cost),
            cash=0.0,
        )
        nodes.append(factory)
        node_instances.append(NodeInstance(node=factory, init_seed=init_seed + i, policy=None))

    # One intermediate node "S" carrying all active products.
    # Initial inventory split evenly across products.
    inv_per_product = max(1, capacity // max(1, len(active_subset)))
    initial_inventory = {pid: inv_per_product for pid in active_subset}

    # Set up list prices at MSRP (base_price).
    list_prices = {pid: float(pid_to_ware[pid].base_price) for pid in active_subset}
    min_order_imposed = {pid: 0 for pid in active_subset}

    intermediate = IntermediateNode(
        id="S",
        region=region,
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
    nodes.append(intermediate)
    node_instances.append(NodeInstance(node=intermediate, init_seed=init_seed + len(active_subset), policy=None))

    # One demand sink per active product.
    for j, pid in enumerate(active_subset):
        from src.sim.distributions import Uniform
        sink = DemandSinkNode(
            id=f"D_{pid}",
            region=region,
            init_seed=init_seed + len(active_subset) + 1 + j,
            product_id=pid,
            demand_dist=Uniform(2, 8),
            income_rate=float(pid_to_ware[pid].base_price) * 10.0,  # ample income
            cash=balance,
            activation_tick={pid: 0},
        )
        nodes.append(sink)
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

    scenario = Scenario(
        catalog=catalog,
        market=market,
        disruption=disruption_params,
        item_lifecycle=(
            lifecycle_params if lifecycle_params is not None else default_lifecycle_params()
        ),
        nodes=node_instances,
        edges=edge_specs,
        n_steps=episode_length,
        start_date=start_date if start_date is not None else datetime(2024, 1, 1),
        world_seed=world_seed,
    )

    return scenario


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def sample_episode(
    catalog: list[Ware],
    config: "RLConfig",
    episode_seed: int,
    *,
    market_params: MarketParams | None = None,
    disruption_params: DisruptionParams | None = None,
    lifecycle_params: ItemLifecycleParams | None = None,
    start_date: datetime | None = None,
) -> RLEpisodeSpec:
    """Deterministically sample a fully-specified ``RLEpisodeSpec``.

    Parameters
    ----------
    catalog:
        The full product universe.
    config:
        ``RLConfig`` instance.  K is sampled per episode from
        ``[K_min, K_max_episode]``; capacity, balance, delivery_lag,
        holding_rate, order_fee, and episode_length are consumed here.
    episode_seed:
        Single integer seed. Any two calls with the same seed and the same
        ``(catalog, config)`` produce identical ``RLEpisodeSpec``
        objects — the function is pure.
    market_params, disruption_params, lifecycle_params:
        Optional overrides; fall back to sensible defaults when ``None``.
    start_date:
        Episode start date; defaults to 2024-01-01.

    Returns
    -------
    RLEpisodeSpec
        Fully deterministic description of the episode.
    """
    # --- derive independent sub-seeds ---
    assortment_seed = _derive_seed(episode_seed, "assortment")
    capacity_seed = _derive_seed(episode_seed, "capacity")
    balance_seed = _derive_seed(episode_seed, "balance")
    world_seed = _derive_seed(episode_seed, "world")
    k_seed = _derive_seed(episode_seed, "k")

    # --- sample K for this episode ---
    k_rng = Random(k_seed)
    K_min = config.K_min
    K_max_ep = config.K_max_episode
    all_pids = [w.product_id for w in catalog]
    K_max_feasible = min(K_max_ep, len(all_pids))
    K = k_rng.randint(K_min, K_max_feasible)

    # --- sample capacity and balance ---
    capacity_rng = Random(capacity_seed)
    capacity = int(config.capacity_dist.sample(capacity_rng))

    balance_rng = Random(balance_seed)
    balance = float(config.balance_dist.sample(balance_rng))

    # --- sample active subset (without replacement) ---
    assortment_rng = Random(assortment_seed)
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
    scenario = _build_rl_graph_scenario(
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
        market_params=market_params,
        disruption_params=disruption_params,
        lifecycle_params=lifecycle_params,
        start_date=start_date,
    )

    return RLEpisodeSpec(
        scenario=scenario,
        active_subset=active_subset,
        capacity=capacity,
        balance=balance,
        world_seed=world_seed,
    )


def load_catalog_and_market_from_setup(
    setup_dir: "str | Path",
) -> "tuple[list[Ware], MarketParams]":
    """Load catalog + market params from a setup directory.

    Thin wrapper around ``setup_io._parse_catalog`` and
    ``setup_io._parse_market`` so the RL stack can consume a setup directory
    without pulling in the full ``load_setup`` pipeline (which also requires
    nodes/edges/run/disruption to be present).

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

    This is the small synthetic-setup helper the RL stack uses when no
    setup dir is supplied.  Products have stable P{i:04d} ids, consistent
    pricing, and ``all_season`` seasonality.
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
    "RLEpisodeSpec",
    "EpisodeSpec",
    "sample_episode",
    "load_catalog_and_market_from_setup",
    "make_synthetic_catalog",
]
