"""Tuning-side episode sampler: builds a graph-mode replenishment episode.

``sample_episode(catalog, config, episode_seed) → TuningEpisodeSpec``
builds one shop between factories and demand sinks:

    FactoryNode("F_<pid>_<i>") → IntermediateNode("S") → DemandSinkNode("D_<pid>")

Each active SKU has several factories. Their production, opening stock,
list price, and edge lead time are sampled for that episode (ADR 0031).
The intermediate node "S" is the node under policy optimisation.

``TuningEpisodeSpec`` is the canonical tuning episode descriptor. It carries
the graph-mode ``Scenario`` plus the per-episode ``capacity``, ``balance``,
``tightness``, ``cash_tightness``, and nominal ``delivery_lag`` so callers can read the
supply contract without walking the edges.

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
    LIFECYCLE_DEAD_PROB_HIGH,
    LIFECYCLE_DEAD_PROB_LOW,
    LIFECYCLE_OTHER_PROB_HIGH,
    LIFECYCLE_OTHER_PROB_LOW,
    LIFECYCLE_RATE_SCALE_HIGH,
    LIFECYCLE_RATE_SCALE_LOW,
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
        A graph-mode ``Scenario``: several ``FactoryNode``s per active SKU,
        one ``IntermediateNode`` "S", one ``DemandSinkNode`` per active SKU.
    active_subset
        The K product ids selected for this episode, in catalog order.
    capacity
        The IntermediateNode capacity for this episode (int).
    balance
        The IntermediateNode opening cash for this episode (float).
    delivery_lag
        Nominal lead time for this episode. Supplier edges differ around it.
    tightness
        Factor that sized capacity from this episode's demand.
        ``None`` when both pools were pinned by ``capacity_dist`` and
        ``balance_dist``.
    cash_tightness
        Factor that sized opening cash from this episode's demand
        valued at unit cost. ``None`` in the same pinned case.
    """

    scenario: Scenario
    active_subset: tuple[str, ...]
    capacity: int
    balance: float
    delivery_lag: int
    tightness: float | None = None
    cash_tightness: float | None = None


# ---------------------------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------------------------

# The historical name in this module; also re-exported from sim for compatibility.
EpisodeSpec = TuningEpisodeSpec


# ---------------------------------------------------------------------------
# Supply side (ADR 0031)
# ---------------------------------------------------------------------------

# Several factories per SKU. Index 0 is the fastest and the dearest.
_SUPPLIERS_PER_SKU = 3
_LEAD_OFFSETS = (0, 2, 4)
_LAG_MIN = 2
_LAG_MAX = 6
# Production per tick and opening stock, as multiples of that SKU's
# expected demand per tick. Summed across the SKU's suppliers.
_PROD_LOW = 0.15
_PROD_HIGH = 1.25
_OPEN_LOW = 0.0
_OPEN_HIGH = 4.0
# Markup over unit cost. Each step faster than the slowest supplier
# adds another draw of the speed premium.
_MARKUP_LOW = 0.05
_MARKUP_HIGH = 0.25
_PREMIUM_LOW = 0.10
_PREMIUM_HIGH = 0.35
# At tightness 1 the shelf holds this many ticks of expected demand.
# Opening cash has no extra horizon: it is cash tightness times the
# episode's demand valued at unit cost. The two factors are separate
# draws. One shared factor cannot do this: on a 365-tick episode the
# cash pool would always buy about twenty times what the shelf holds,
# so cash would never be what stops the order.
_COVER_HORIZON_TICKS = 14
_CAPACITY_TIGHTNESS_LOW = 0.5
_CAPACITY_TIGHTNESS_HIGH = 6.0
_CASH_TIGHTNESS_LOW = 0.02
_CASH_TIGHTNESS_HIGH = 0.8


def _expected_demand_per_tick(ware: Ware) -> float:
    """Catalog demand per tick. Missing scale keeps the ``Uniform(2, 8)`` midpoint."""
    if ware.demand_mean is None:
        return 5.0
    return max(float(ware.demand_mean), 1e-6)


def _pools_from_tightness(
    tightness: float,
    cash_tightness: float,
    wares: list[Ware],
    episode_length: int,
) -> tuple[int, float]:
    """Capacity and opening cash from the two tightness factors.

    Capacity is ``tightness × aggregate episode demand × cover fraction``,
    and the cover fraction is ``_COVER_HORIZON_TICKS / episode_length``,
    so the shelf holds ``tightness`` cover horizons of per-tick demand.
    Opening cash is ``cash_tightness × aggregate episode demand × unit cost``.
    """
    length = float(episode_length)
    daily = 0.0
    episode_cost = 0.0
    for ware in wares:
        rate = _expected_demand_per_tick(ware)
        daily += rate
        episode_cost += rate * length * float(ware.unit_cost)
    cover_fraction = _COVER_HORIZON_TICKS / length
    aggregate_units = daily * length
    capacity = max(1, int(round(tightness * aggregate_units * cover_fraction)))
    return capacity, float(cash_tightness * episode_cost)


def _mix_seed(seed: int, text: str) -> int:
    mixed = seed & _MASK_32
    for char in text:
        mixed = (mixed * 0x01000193 + ord(char)) & _MASK_32
    return mixed


def _nominal_delivery_lag(config: "TuningConfig", episode_seed: int) -> int:
    """Pinned lag, or a draw from ``[_LAG_MIN, _LAG_MAX]``."""
    if config.delivery_lag is not None:
        return int(config.delivery_lag)
    rng = Random(_sim_derive_seed(episode_seed, "delivery_lag"))
    return rng.randint(_LAG_MIN, _LAG_MAX)


def _supplier_plan(ware: Ware, episode_seed: int, nominal_lag: int) -> list[dict[str, Any]]:
    """Per-supplier production, opening stock, price, and lead time for one SKU."""
    rng = Random(_mix_seed(_sim_derive_seed(episode_seed, "supply"), ware.product_id))
    demand = _expected_demand_per_tick(ware)
    prod_factor = rng.uniform(_PROD_LOW, _PROD_HIGH)
    open_days = rng.uniform(_OPEN_LOW, _OPEN_HIGH)
    weights = [rng.uniform(0.4, 1.0) for _ in range(_SUPPLIERS_PER_SKU)]
    weight_sum = sum(weights)
    slow_markup = rng.uniform(_MARKUP_LOW, _MARKUP_HIGH)
    premium = rng.uniform(_PREMIUM_LOW, _PREMIUM_HIGH)
    plan: list[dict[str, Any]] = []
    for index, offset in enumerate(_LEAD_OFFSETS):
        share = weights[index] / weight_sum
        steps_faster = _SUPPLIERS_PER_SKU - 1 - index
        plan.append(
            {
                "index": index,
                "lead": int(nominal_lag) + offset,
                "production": max(0, int(round(prod_factor * demand * share))),
                "opening": max(0, int(round(open_days * demand * share))),
                "list_price": float(ware.unit_cost) * (1.0 + slow_markup + steps_faster * premium),
            }
        )
    return plan


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
    episode_seed: int,
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

    The graph has three node types:
      - FactoryNode("F_<pid>_<i>") — several per active product
      - IntermediateNode("S") — the node under policy evaluation
      - DemandSinkNode("D_<pid>") — one per active product

    Edges:
      F_<pid>_<i> → S (lead time differs across i)
      S → D_<pid>  (default_lead_time = 1 — intra-day)
    """
    from src.sim.distributions import Constant
    from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode
    from src.sim.scenario import NodeInstance
    from src.sim.graph import EdgeSpec

    pid_to_ware: dict[str, Ware] = {w.product_id: w for w in catalog}

    node_instances: list[Any] = []
    edge_specs: list[Any] = []

    n_factories = len(active_subset) * _SUPPLIERS_PER_SKU
    factory_slot = 0
    for pid in active_subset:
        ware = pid_to_ware[pid]
        for supplier in _supplier_plan(ware, episode_seed, delivery_lag):
            factory_id = f"F_{pid}_{supplier['index']}"
            factory = FactoryNode(
                id=factory_id,
                region="US",
                init_seed=init_seed + factory_slot,
                produces_product_id=pid,
                unit_cost=float(ware.unit_cost),
                capacity_per_tick=Constant(supplier["production"]),
                inventory=supplier["opening"],
                list_price=supplier["list_price"],
                cash=0.0,
            )
            node_instances.append(
                NodeInstance(node=factory, init_seed=init_seed + factory_slot, policy=None)
            )
            edge_specs.append(EdgeSpec(
                supplier_id=factory_id,
                buyer_id="S",
                default_lead_time=supplier["lead"],
            ))
            factory_slot += 1

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
        init_seed=init_seed + n_factories,
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
    node_instances.append(
        NodeInstance(node=intermediate, init_seed=init_seed + n_factories, policy=None)
    )

    # One demand sink per active product.
    for j, pid in enumerate(active_subset):
        sink = DemandSinkNode(
            id=f"D_{pid}",
            region="US",
            init_seed=init_seed + n_factories + 1 + j,
            product_id=pid,
            demand_dist=_sink_demand_dist(pid_to_ware[pid]),
            income_rate=float(pid_to_ware[pid].base_price) * 10.0,  # ample income
            cash=balance,
            activation_tick={pid: 0},
        )
        node_instances.append(
            NodeInstance(node=sink, init_seed=init_seed + n_factories + 1 + j, policy=None)
        )

    for pid in active_subset:
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
        ``capacity_dist`` and ``balance_dist`` (``None`` sizes that pool
        from its tightness factor), ``delivery_lag`` (``None``
        samples the nominal lag), ``holding_rate``, ``order_fee``, and
        ``episode_length``.
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

    delivery_lag = _nominal_delivery_lag(config, episode_seed)

    # Pools are sized after the assortment exists, because aggregate demand
    # is the active SKUs' demand. An explicit distribution still overrides
    # that side. Tightness is recorded only when it sized at least one pool.
    capacity_dist = config.capacity_dist
    balance_dist = config.balance_dist
    if capacity_dist is not None and balance_dist is not None:
        tightness = None
        cash_tightness = None
        capacity = int(capacity_dist.sample(Random(capacity_seed)))
        balance = float(balance_dist.sample(Random(balance_seed)))
    else:
        from src.sim.distributions import LogUniform

        tightness_rng = Random(_sim_derive_seed(episode_seed, "tightness"))
        tightness = float(
            LogUniform(_CAPACITY_TIGHTNESS_LOW, _CAPACITY_TIGHTNESS_HIGH).sample(
                tightness_rng
            )
        )
        cash_tightness = float(
            LogUniform(_CASH_TIGHTNESS_LOW, _CASH_TIGHTNESS_HIGH).sample(tightness_rng)
        )
        by_id = {ware.product_id: ware for ware in catalog}
        sized_capacity, sized_balance = _pools_from_tightness(
            tightness,
            cash_tightness,
            [by_id[pid] for pid in active_subset],
            config.episode_length,
        )
        if capacity_dist is None:
            capacity = sized_capacity
        else:
            capacity = int(capacity_dist.sample(Random(capacity_seed)))
        if balance_dist is None:
            balance = sized_balance
        else:
            balance = float(balance_dist.sample(Random(balance_seed)))

    # --- build the graph-mode Scenario ---
    scenario = _build_tuning_graph_scenario(
        catalog=catalog,
        active_subset=active_subset,
        capacity=capacity,
        balance=balance,
        world_seed=world_seed,
        episode_length=config.episode_length,
        delivery_lag=delivery_lag,
        episode_seed=episode_seed,
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
        delivery_lag=delivery_lag,
        tightness=tightness,
        cash_tightness=cash_tightness,
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


def episode_distribution_parameters(
    config: "TuningConfig",
    *,
    market_params: MarketParams | None = None,
    disruption_params: DisruptionParams | None = None,
    catalog_size: int | None = None,
) -> dict[str, Any]:
    """JSON-ready parameters ``sample_episode`` draws from for this config.

    A demand-relative pool records its log-uniform tightness. A pool with an
    explicit distribution records that distribution instead, so the artifact
    describes the episodes that were actually sampled. Market and disruption
    are the parameter objects the episode builder uses, in full.
    """
    from src.sim.distributions import LogUniform

    market = market_params if market_params is not None else default_market_params()
    disruption = (
        disruption_params if disruption_params is not None else default_disruption_params()
    )
    if config.capacity_dist is None:
        capacity_pool = "demand_relative"
        capacity_fields: dict[str, Any] = {
            "capacity_pool": capacity_pool,
            "capacity_tightness": LogUniform(
                _CAPACITY_TIGHTNESS_LOW, _CAPACITY_TIGHTNESS_HIGH
            ).to_dict(),
            "cover_horizon_ticks": _COVER_HORIZON_TICKS,
        }
    else:
        capacity_fields = {
            "capacity_pool": "pinned",
            "capacity_distribution": config.capacity_dist.to_dict(),
        }
    if config.balance_dist is None:
        cash_fields: dict[str, Any] = {
            "cash_pool": "demand_relative",
            "cash_tightness": LogUniform(
                _CASH_TIGHTNESS_LOW, _CASH_TIGHTNESS_HIGH
            ).to_dict(),
        }
    else:
        cash_fields = {
            "cash_pool": "pinned",
            "cash_distribution": config.balance_dist.to_dict(),
        }
    init_stock = config.init_stock_pct_dist
    return {
        "holding_rate": config.holding_rate,
        "order_fee": config.order_fee,
        "episode_length": config.episode_length,
        "K_active": config.K_active,
        "K_min": config.K_min,
        "K_max_episode": config.K_max_episode,
        "K_catalog": config.K_catalog,
        "catalog_size": (
            config.active_catalog_size() if catalog_size is None else catalog_size
        ),
        "delivery_lag": config.delivery_lag,
        "delivery_lag_sampled": config.delivery_lag is None,
        "delivery_lag_low": _LAG_MIN,
        "delivery_lag_high": _LAG_MAX,
        **capacity_fields,
        **cash_fields,
        "suppliers_per_sku": _SUPPLIERS_PER_SKU,
        "lead_time_offsets": list(_LEAD_OFFSETS),
        "production_multiple_of_demand": {"low": _PROD_LOW, "high": _PROD_HIGH},
        "opening_days_of_demand": {"low": _OPEN_LOW, "high": _OPEN_HIGH},
        "markup_over_unit_cost": {"low": _MARKUP_LOW, "high": _MARKUP_HIGH},
        "speed_premium": {"low": _PREMIUM_LOW, "high": _PREMIUM_HIGH},
        "init_stock_pct": None if init_stock is None else init_stock.to_dict(),
        "item_lifecycle": {
            "mode": "sampled_per_episode",
            "dead_stage_change_prob": {
                "type": "uniform",
                "low": LIFECYCLE_DEAD_PROB_LOW,
                "high": LIFECYCLE_DEAD_PROB_HIGH,
            },
            "other_stage_change_prob": {
                "type": "uniform",
                "low": LIFECYCLE_OTHER_PROB_LOW,
                "high": LIFECYCLE_OTHER_PROB_HIGH,
            },
            "authored_rate_scale": {
                "type": "uniform",
                "low": LIFECYCLE_RATE_SCALE_LOW,
                "high": LIFECYCLE_RATE_SCALE_HIGH,
            },
        },
        "market": market.to_dict(),
        "disruption": disruption.to_dict(),
    }


# Held-out share of the catalog. A stride across product-id order keeps
# both sides spread over the demand range. The high-demand tail is not
# the evaluation set by itself.
EVAL_CATALOG_FRACTION = 0.2


def partition_catalog(catalog: list[Ware]) -> tuple[list[Ware], list[Ware]]:
    """Split ``catalog`` into training SKUs and held-out evaluation SKUs.

    Evaluation SKUs are one fifth of the catalog, strided through product-id
    order, at least one SKU, and never the whole catalog. The two lists are
    disjoint. Callers that sample training episodes pass the first list;
    the unseen-SKU protocol passes the second.
    """
    if len(catalog) < 2:
        raise ValueError(
            "partition_catalog: need at least one training SKU and one evaluation SKU"
        )
    ordered = sorted(catalog, key=lambda ware: ware.product_id)
    n_eval = int(round(len(ordered) * EVAL_CATALOG_FRACTION))
    n_eval = min(max(n_eval, 1), len(ordered) - 1)
    step = len(ordered) / n_eval
    # Three fifths of the way through each stride. The midpoint of the
    # stride leaves the mixture log's seed block with no capacity-bound tick.
    phase = step * 0.6
    chosen: set[int] = set()
    for slot in range(n_eval):
        index = min(len(ordered) - 1, int(step * slot + phase))
        while index in chosen:
            index = (index + 1) % len(ordered)
        chosen.add(index)
    evaluation = [ordered[index] for index in sorted(chosen)]
    training = [ware for index, ware in enumerate(ordered) if index not in chosen]
    return training, evaluation


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
    "episode_distribution_parameters",
    "load_catalog_and_market_from_setup",
    "make_synthetic_catalog",
    "partition_catalog",
    "EVAL_CATALOG_FRACTION",
]
