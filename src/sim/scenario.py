"""Scenario dataclass and authoring helpers.

A ``Scenario`` is a flat declarative description of one experiment:
catalog, typed parameter bags (``MarketParams`` / ``DisruptionParams`` /
``ItemLifecycleParams``), the list of ``NodeInstance``s + ``EdgeSpec``s
that define the graph topology, ``n_steps``, ``start_date``, and
``world_seed``.

Phase 4 (issue 11): the legacy ``stores``-based path has been retired.
``Scenario.stores``, ``StoreInstance``, ``StoreTemplate``, and
``make_stores`` remain in this module for backward compatibility with
the RL/tuning layer (which will be migrated in issues 12–13), but the
graph engine (``Runner`` / ``Simulation`` / ``build_world``) requires
``scenario.is_graph == True`` and ignores ``stores``.

This module also hosts authoring helpers:

- ``load_catalog(items)`` — build ``[Ware]`` with stable ``P{i:04d}`` ids.
- ``make_nodes(triples)`` — turn ``(node, init_seed, policy)`` triples
  into a ``[NodeInstance]`` roster.
- ``make_stores(triples)`` — **(deprecated)** legacy roster builder.
- ``load_scenario_from_path(path)`` — import a Python module and return
  its top-level ``scenario`` symbol, preserving live Policy instances
  (used by ``main.py``).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import namedtuple
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping

if TYPE_CHECKING:
    # ``World`` is in ``src.sim.world``; no import cycle. Kept under
    # TYPE_CHECKING so the type annotation works without eager import.
    from src.sim.world import World

from src.sim.distributions import (
    Distribution,
    _REGISTRY as _DISTRIBUTION_REGISTRY,
    distribution_from_dict,
)


# A ``Ware`` is the static catalog record: 7 required + 5 optional
# per-Ware override fields, defaulting to ``None`` so ``ItemRegistry``
# falls back to ``ItemLifecycleParams`` when the author didn't specify.
# We use ``namedtuple`` rather than ``dataclass`` because tests still
# build large literal lists of Wares and the positional-construction
# ergonomics matter.
Ware = namedtuple(
    "Ware",
    [
        "product_id",
        "name",
        "category",
        "related_products",
        "base_price",
        "unit_cost",
        "seasonality",
        # Optional per-Ware lifecycle overrides (issue 02). When ``None``,
        # ``ItemRegistry`` falls back to ``ItemLifecycleParams.init_stage`` /
        # ``ItemLifecycleParams.default_stage_change_probs``.
        "init_stage",
        "stage_change_probs",
        # Optional per-Ware freshness overrides (issue 04). When ``None``,
        # ``ItemRegistry`` falls back to
        # ``ItemLifecycleParams.default_freshness_alpha`` /
        # ``default_freshness_decay``. ``freshness_alpha = 0`` (scalar) is
        # the staple-style "no hype curve" override.
        "freshness_alpha",
        "freshness_decay",
        # Optional per-Ware initial-stock weight (issue 08). When ``None``,
        # ``ItemRegistry`` falls back to
        # ``ItemLifecycleParams.default_init_stock_share``. Drives weighted
        # allocation of the ``capacity * init_stock_pct`` budget across
        # initially active SKUs in ``init_store_state``.
        "init_stock_share",
    ],
    defaults=(None, None, None, None, None),
)


def load_catalog(items: Iterable[Mapping[str, Any]]) -> list[Ware]:
    """Build a catalog of ``Ware``s, assigning ``P{i:04d}`` ids in order.

    ``product_id`` keys on the input dicts are silently discarded so the
    function is the single source of truth for id assignment — two
    scenarios authored against the same dict list will produce
    identical product_ids.
    """
    # Output buffer; populated in iteration order so the id sequence is
    # deterministic.
    out: list[Ware] = []
    for i, item in enumerate(items):
        # Shallow copy so we don't mutate caller-owned dicts.
        kwargs = dict(item)
        # Any caller-supplied ``product_id`` is replaced — this is the
        # canonical id-assignment point.
        kwargs.pop("product_id", None)
        # Normalise related_products entries into tuples — JSON
        # round-trip would otherwise deliver them as lists.
        related = kwargs.get("related_products", [])
        kwargs["related_products"] = [tuple(p) for p in related]
        out.append(Ware(product_id=f"P{i:04d}", **kwargs))
    return out


def _serialize(value: Any) -> Any:
    """Recursively convert ``Distribution`` instances and nested containers to JSON-friendly forms."""
    if isinstance(value, Distribution):
        return value.to_dict()
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(v) for v in value]
    return value


def _deserialize(value: Any) -> Any:
    """Inverse of ``_serialize``: rebuild ``Distribution`` instances from tagged dicts."""
    if isinstance(value, dict):
        # ``type`` key + matching registry entry ⇒ this dict is a serialised Distribution.
        if "type" in value and value["type"] in _DISTRIBUTION_REGISTRY:
            return distribution_from_dict(value)
        return {k: _deserialize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deserialize(v) for v in value]
    return value


def _ware_to_dict(w: Ware) -> dict[str, Any]:
    """Serialise a ``Ware`` to a JSON-friendly dict (lists in place of tuples)."""
    d: dict[str, Any] = {}
    for k, v in w._asdict().items():
        if k == "related_products":
            # Tuples → lists so JSON doesn't lose the inner shape.
            d[k] = [list(p) for p in v]
        else:
            d[k] = _serialize(v)
    return d


def _ware_from_dict(d: Mapping[str, Any]) -> Ware:
    """Inverse of ``_ware_to_dict``; rebuilds tuples and ``Distribution`` instances."""
    return Ware(
        product_id=d["product_id"],
        name=d["name"],
        category=d["category"],
        related_products=[tuple(p) for p in d.get("related_products", [])],
        base_price=d["base_price"],
        unit_cost=d["unit_cost"],
        seasonality=d["seasonality"],
        init_stage=_deserialize(d.get("init_stage")),
        stage_change_probs=_deserialize(d.get("stage_change_probs")),
        freshness_alpha=_deserialize(d.get("freshness_alpha")),
        freshness_decay=_deserialize(d.get("freshness_decay")),
        init_stock_share=_deserialize(d.get("init_stock_share")),
    )


# StoreTemplate fields whose runtime type is "scalar OR Distribution". Listed
# explicitly so a typo in a serialised field name fails loudly rather than
# silently materialising as a generic dict.
_TEMPLATE_FIELDS = (
    "id",
    "region",
    "capacity",
    "init_balance",
    "init_stock_pct",
    "delivery_lag",
    "holding_rate",
    "order_fee",
    "init_active_count",
)


@dataclass(frozen=True)
class StoreTemplate:
    """Reusable store profile.

    Stochastic fields hold ``Distribution`` instances and are sampled at
    ``Store`` construction time using the per-instance ``init_seed``. Holding
    distributions on the template lets two stores constructed from the same
    ``(template, init_seed)`` start bit-identical regardless of attached
    policy (issue 03 enforces this end-to-end).

    ``init_active_products`` (issue 06) is an optional explicit roster of
    product ids to activate at step 0. When set, it overrides the random
    ``init_rng.sample(catalog, init_active_count)`` fallback so a "fashion
    specialist" template can declare its starting SKUs literally. ``None``
    (default) preserves the random-sample behaviour for catalogs that
    don't author an explicit list.

    ``init_freshness`` (issue 07) selects between two step-0 freshness
    regimes for the initial active SKUs. ``"baseline"`` (default) is an
    established store: ``Store.freshness_multiplier(pid, 0) == 1`` for
    every initial active SKU — they skip the hype window. ``"fresh"`` is
    a grand-opening: ``activation_tick[pid] = 0`` for every initial
    active SKU so they enter at full hype (multiplier ``1 + α``).
    """

    # Short label for the template (used in run logs / store parquet).
    id: str
    # Region key — must appear in ``MarketParams.regions``.
    region: str
    # Total inventory capacity. Scalar or Distribution.
    capacity: int | float | Distribution
    # Opening cash balance.
    init_balance: int | float | Distribution
    # Fraction of capacity initially stocked (in [0, 1]).
    init_stock_pct: float | Distribution
    # Default delivery lead time.
    delivery_lag: int | float | Distribution
    # Per-step holding cost rate.
    holding_rate: float | Distribution
    # Fixed fee per non-zero order.
    order_fee: int | float | Distribution
    # Count of initial active SKUs (ignored when ``init_active_products`` set).
    init_active_count: int | float | Distribution
    # Optional explicit step-0 roster — overrides the random sample.
    init_active_products: list[str] | None = None
    # Step-0 freshness regime: established ("baseline") vs grand-opening ("fresh").
    init_freshness: Literal["baseline", "fresh"] = "baseline"

    def to_dict(self) -> dict[str, Any]:
        """Serialise every field via ``_serialize`` (Distributions → tagged dicts)."""
        return {f.name: _serialize(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "StoreTemplate":
        """Inverse of ``to_dict``; validates required fields and ``init_freshness``."""
        # Surface missing required fields as a single sorted error.
        missing = set(_TEMPLATE_FIELDS) - set(d)
        if missing:
            raise ValueError(
                f"StoreTemplate.from_dict: missing fields {sorted(missing)}"
            )
        kwargs: dict[str, Any] = {k: _deserialize(d[k]) for k in _TEMPLATE_FIELDS}
        # Optional roster — pass through if present and non-null.
        if "init_active_products" in d and d["init_active_products"] is not None:
            kwargs["init_active_products"] = list(d["init_active_products"])
        # Optional freshness mode with closed-enum validation.
        if "init_freshness" in d and d["init_freshness"] is not None:
            mode = d["init_freshness"]
            if mode not in ("baseline", "fresh"):
                raise ValueError(
                    f"StoreTemplate.from_dict: init_freshness must be "
                    f"'baseline' or 'fresh', got {mode!r}"
                )
            kwargs["init_freshness"] = mode
        return cls(**kwargs)


@dataclass
class StoreInstance:
    """One store entry in a ``Scenario``.

    The ``init_seed`` deterministically drives initial active-SKU selection
    and stock allocation. ``policy`` is intentionally *not* serialised by
    ``Scenario.to_json``: scenarios authored by the LLM are world artifacts,
    while policies are wired up in the experiment script.
    """

    # Reusable specification — multiple instances may share a template.
    template: StoreTemplate
    # Per-instance RNG seed; bit-identity contract: two instances with
    # the same ``(template, init_seed)`` start step 0 identical.
    init_seed: int
    # Decision-making brain — not serialised (re-attached on load).
    policy: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly form. Note: ``policy`` is intentionally omitted."""
        return {
            "template": self.template.to_dict(),
            "init_seed": self.init_seed,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "StoreInstance":
        """Rebuild a ``StoreInstance`` from JSON; ``policy`` always returns ``None``."""
        return cls(
            template=StoreTemplate.from_dict(d["template"]),
            init_seed=d["init_seed"],
            policy=None,
        )


def _node_to_dict(node: Any) -> dict[str, Any]:
    """Serialise a Node subclass to a JSON-friendly dict tagged with ``node_type``."""
    from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode

    d: dict[str, Any] = {"node_type": type(node).__name__}
    # Shared Node fields
    d["id"] = node.id
    d["region"] = node.region
    d["init_seed"] = node.init_seed
    # policy intentionally omitted (not serialised)
    # level omitted — it is computed by the graph builder, not stored in JSON

    if isinstance(node, FactoryNode):
        d["produces_product_id"] = node.produces_product_id
        d["unit_cost"] = node.unit_cost
        d["capacity_per_tick"] = _serialize(node.capacity_per_tick)
        d["inventory"] = node.inventory
        d["list_price"] = node.list_price
        d["cash"] = node.cash

    elif isinstance(node, IntermediateNode):
        d["carried_products"] = sorted(node.carried_products)
        d["capacity"] = _serialize(node.capacity)
        d["tags"] = list(node.tags)
        d["inventory"] = dict(node.inventory)
        d["pending"] = {k: dict(v) for k, v in node.pending.items()}
        d["list_prices"] = dict(node.list_prices)
        d["min_order_imposed"] = dict(node.min_order_imposed)
        d["cash"] = node.cash

    elif isinstance(node, DemandSinkNode):
        d["product_id"] = node.product_id
        d["demand_dist"] = _serialize(node.demand_dist)
        d["income_rate"] = node.income_rate
        d["cash"] = node.cash
        d["activation_tick"] = dict(node.activation_tick)

    else:
        raise TypeError(f"_node_to_dict: unknown node type {type(node).__name__!r}")

    return d


def _node_from_dict(d: Mapping[str, Any]) -> Any:
    """Inverse of ``_node_to_dict``; reconstructs the appropriate Node subclass."""
    from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode

    node_type = d.get("node_type")
    shared: dict[str, Any] = {
        "id": d["id"],
        "region": d["region"],
        "init_seed": d["init_seed"],
        # policy always None on deserialise
        "policy": None,
    }

    if node_type == "FactoryNode":
        return FactoryNode(
            **shared,
            produces_product_id=d["produces_product_id"],
            unit_cost=d["unit_cost"],
            capacity_per_tick=_deserialize(d["capacity_per_tick"]),
            inventory=d["inventory"],
            list_price=d["list_price"],
            cash=d.get("cash", 0.0),
        )

    elif node_type == "IntermediateNode":
        return IntermediateNode(
            **shared,
            carried_products=set(d["carried_products"]),
            capacity=_deserialize(d["capacity"]),
            tags=list(d["tags"]),
            inventory=dict(d["inventory"]),
            pending={k: dict(v) for k, v in d.get("pending", {}).items()},
            list_prices=dict(d["list_prices"]),
            min_order_imposed=dict(d["min_order_imposed"]),
            cash=d.get("cash", 0.0),
        )

    elif node_type == "DemandSinkNode":
        return DemandSinkNode(
            **shared,
            product_id=d["product_id"],
            demand_dist=_deserialize(d["demand_dist"]),
            income_rate=d["income_rate"],
            cash=d["cash"],
            activation_tick=dict(d["activation_tick"]),
        )

    else:
        raise ValueError(f"_node_from_dict: unknown node_type {node_type!r}")


def _edge_to_dict(edge: Any) -> dict[str, Any]:
    """Serialise an ``EdgeSpec`` to a JSON-friendly dict."""
    from src.sim.graph import EdgeSpec

    d: dict[str, Any] = {
        "supplier_id": edge.supplier_id,
        "buyer_id": edge.buyer_id,
        "default_lead_time": edge.default_lead_time,
    }
    if edge.per_product_lead_time is not None:
        d["per_product_lead_time"] = dict(edge.per_product_lead_time)
    return d


def _edge_from_dict(d: Mapping[str, Any]) -> Any:
    """Inverse of ``_edge_to_dict``; reconstructs an ``EdgeSpec``."""
    from src.sim.graph import EdgeSpec

    per_product = d.get("per_product_lead_time")
    return EdgeSpec(
        supplier_id=d["supplier_id"],
        buyer_id=d["buyer_id"],
        default_lead_time=d["default_lead_time"],
        per_product_lead_time=dict(per_product) if per_product else None,
    )


@dataclass
class NodeInstance:
    """One node entry in a graph ``Scenario``.

    The ``init_seed`` deterministically drives step-0 state. ``policy``
    is intentionally *not* serialised by ``Scenario.to_json``: scenarios
    authored by the LLM are world artifacts, while policies are wired up
    in the experiment script. This mirrors ``StoreInstance``.
    """

    # The node template.
    node: Any  # Node subclass instance
    # Per-instance RNG seed.
    init_seed: int
    # Decision-making brain — not serialised (re-attached on load).
    policy: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly form. Note: ``policy`` is intentionally omitted."""
        return {
            "node": _node_to_dict(self.node),
            "init_seed": self.init_seed,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "NodeInstance":
        """Rebuild a ``NodeInstance`` from JSON; ``policy`` always returns ``None``."""
        return cls(
            node=_node_from_dict(d["node"]),
            init_seed=d["init_seed"],
            policy=None,
        )


@dataclass
class MarketParams:
    """Typed parameter bag for the demand/supply environment.

    Domain values (``cycle_len``, seasonal factors, region list) and
    math-tuning values (price elasticity, divisors, clamp bounds) are flat at
    this layer — the old ``initial_parameters`` / ``dynamic_parameters``
    split is gone. Per-step stochastic fields are ``Distribution`` instances
    and are sampled by ``Market`` against an injected ``world_rng``.
    """

    # Period of the seasonal sine wave in steps.
    cycle_len: int | Distribution
    # Amplitude of the same wave.
    cycle_amp: float | Distribution
    # Step-0 demand / supply levels.
    init_demand: float
    init_supply: float
    # In-season / off-season demand multipliers.
    peak_factor: float | Distribution
    off_factor: float | Distribution
    # ``{label: [month1, …]}`` mapping for ``season_factor`` lookup.
    season_months: dict[str, list[int]]
    # Region keys driving per-region market_state allocation.
    regions: list[str]
    # Correlation between demand and supply shocks.
    correlation: float
    # How often (in steps) ``trend`` is re-drawn.
    trend_update_interval: int
    # Demand / supply state clamp range.
    min_value: float
    max_value: float
    # PLC stage demand multipliers — keys mirror ``CANONICAL_STAGES``.
    stage_multipliers: dict[str, float | Distribution]
    # Price elasticity exponent (should be negative).
    price_elasticity: float
    # Multiplier applied during promotions.
    promo_multiplier: float
    # Floor used inside the demand / supply factor calculations. Since
    # ``market_demand`` / ``market_supply`` live on a 0–2  scale
    # and are consumed directly as factors, no divisor is needed.
    demand_factor_min: float
    supply_factor_min: float
    # Inventory ratio band for the cross-product adjustment.
    cross_inv_lo: float
    cross_inv_hi: float
    # Cross-factor clamp range.
    cross_factor_range: tuple[float, float]
    # Trend factor distribution (sampled at construction + every interval).
    trend: Distribution
    # Per-step random shocks.
    demand_shock: Distribution
    supply_shock: Distribution
    # Base demand draw consumed once per (store, product, tick) in ``sample_demand``.
    base_demand: Distribution

    def to_dict(self) -> dict[str, Any]:
        """Recursive ``_serialize`` over every field."""
        return {f.name: _serialize(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "MarketParams":
        """Inverse of ``to_dict``. Restores tuple shapes lost via JSON."""
        out = {f.name: _deserialize(d[f.name]) for f in fields(cls) if f.name in d}
        # Tuple-typed range fields lose their tuple-ness through JSON; restore.
        if "cross_factor_range" in out and isinstance(out["cross_factor_range"], list):
            out["cross_factor_range"] = tuple(out["cross_factor_range"])
        return cls(**out)


@dataclass
class DisruptionParams:
    """Typed parameter bag for the ``EventEngine``."""

    # Per-tick Bernoulli probability of spawning a disruption event.
    event_prob: float
    # Allowed event types (one is uniformly chosen on spawn).
    types: list[str]
    # Region pool the event may target.
    regions: list[str]
    # Per-event severity sampled at spawn time.
    severity: Distribution
    # Per-event duration sampled at spawn time.
    duration: Distribution

    def to_dict(self) -> dict[str, Any]:
        """Serialise every field via ``_serialize``."""
        return {f.name: _serialize(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DisruptionParams":
        """Deserialise ``Distribution``-valued fields back into instances."""
        return cls(**{f.name: _deserialize(d[f.name]) for f in fields(cls) if f.name in d})


@dataclass
class ItemLifecycleParams:
    """Defaults for the global product lifecycle.

    ``stages`` is the canonical stage list (``[introduction, growth,
    maturity, decline, dead]``). ``default_stage_change_probs`` is the
    per-current-stage transition table consumed by
    ``LifecycleClock.advance_stage``; per-``Ware`` overrides on
    ``Ware.stage_change_probs`` win when set.

    ``default_freshness_alpha`` and ``default_freshness_decay`` (issue 03)
    are the catalog-wide ``(α, β)`` parameters for
    ``FreshnessCurve.multiplier``. Default to a no-op curve
    (``α = 0`` ⇒ multiplier identically 1.0) so existing scenarios that
    don't author them behave equivalently. Per-``Ware`` overrides land
    in issue 04.

    ``default_init_stock_share`` (issue 08) is the catalog-wide weight
    for initial-stock allocation across the active assortment.
    ``StoreInitializer.init_store_state`` distributes the
    ``capacity * init_stock_pct`` budget across initially active SKUs
    proportionally to normalised ``init_stock_share`` weights — staples
    (high share) receive more initial stock than fashion (low share)
    within the same budget. Default ``1.0`` makes every SKU weight-1
    so the allocation collapses to the pre-issue-08 even split,
    preserving prior behaviour for scenarios that don't author the field.
    """

    # Canonical stage list — keys must mirror ``CANONICAL_STAGES``.
    stages: list[str]
    # Initial stage for every catalog Ware (overridable per Ware).
    init_stage: str | Distribution
    # Catalog-wide per-stage transition table (overridable per Ware).
    default_stage_change_probs: dict[str, float | Distribution]
    # Catalog-wide freshness curve defaults.
    default_freshness_alpha: float | Distribution = 0.0
    default_freshness_decay: float | Distribution = 1.0
    # Catalog-wide initial-stock weight default.
    default_init_stock_share: float | Distribution = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise every field via ``_serialize``."""
        return {f.name: _serialize(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ItemLifecycleParams":
        """Inverse of ``to_dict``; per-field ``_deserialize`` to rebuild Distributions."""
        return cls(**{f.name: _deserialize(d[f.name]) for f in fields(cls) if f.name in d})


# Required top-level keys for ``Scenario.from_dict`` validation.
# Phase 4 (issue 11): ``stores`` is no longer required — graph-mode
# scenarios do not include it.  ``nodes`` / ``edges`` are likewise
# optional (they default to empty lists for legacy store-mode scenarios).
_SCENARIO_REQUIRED = (
    "catalog",
    "market",
    "disruption",
    "item_lifecycle",
    "n_steps",
    "start_date",
    "world_seed",
)


@dataclass
class Scenario:
    """Flat declarative description of one experiment.

    Supports both legacy ``stores``-based scenarios and new graph-based
    scenarios via ``nodes`` + ``edges``. The ``is_graph`` property
    distinguishes the two modes. In Phase 0 / Phase 1 both fields are
    optional; in Phase 4 ``nodes``/``edges`` become required and
    ``stores`` is retired.
    """

    # The full product catalog.
    catalog: list[Ware]
    # Market parameter bag.
    market: MarketParams
    # Disruption / event-engine parameters.
    disruption: DisruptionParams
    # Lifecycle / freshness defaults.
    item_lifecycle: ItemLifecycleParams
    # Per-store roster (legacy; instances may share templates and policies).
    stores: list[StoreInstance]
    # Number of ticks to run.
    n_steps: int
    # Wall-clock starting date.
    start_date: datetime
    # Seed for ``world_rng`` (deterministic world trajectory).
    world_seed: int
    # Graph-mode node roster (new; optional in Phase 0 / Phase 1).
    nodes: list[NodeInstance] = field(default_factory=list)
    # Graph-mode edge list (new; optional in Phase 0 / Phase 1).
    edges: list[Any] = field(default_factory=list)  # list[EdgeSpec]

    @property
    def is_graph(self) -> bool:
        """``True`` when this scenario has graph-mode nodes defined."""
        return len(self.nodes) > 0

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict (policies omitted on each store/node instance)."""
        d: dict[str, Any] = {
            "catalog": [_ware_to_dict(w) for w in self.catalog],
            "market": self.market.to_dict(),
            "disruption": self.disruption.to_dict(),
            "item_lifecycle": self.item_lifecycle.to_dict(),
            "n_steps": self.n_steps,
            "start_date": self.start_date.isoformat(),
            "world_seed": self.world_seed,
        }
        # Only include stores / graph fields when they are non-empty, so
        # graph-mode scenarios don't carry an empty ``stores`` key and
        # legacy-mode scenarios don't carry empty ``nodes`` / ``edges``.
        if self.stores:
            d["stores"] = [s.to_dict() for s in self.stores]
        if self.nodes:
            d["nodes"] = [ni.to_dict() for ni in self.nodes]
        if self.edges:
            d["edges"] = [_edge_to_dict(e) for e in self.edges]
        return d

    def to_json(self) -> str:
        """Compact JSON string. Round-trip via ``Scenario.from_json``."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Scenario":
        """Validate required keys then rebuild the full ``Scenario``."""
        missing = set(_SCENARIO_REQUIRED) - set(d)
        if missing:
            raise ValueError(f"Scenario.from_dict: missing keys {sorted(missing)}")
        return cls(
            catalog=[_ware_from_dict(w) for w in d["catalog"]],
            market=MarketParams.from_dict(d["market"]),
            disruption=DisruptionParams.from_dict(d["disruption"]),
            item_lifecycle=ItemLifecycleParams.from_dict(d["item_lifecycle"]),
            # Phase 4 (issue 11): ``stores`` is optional — graph-mode
            # scenarios do not include it; legacy scenarios still do.
            stores=[StoreInstance.from_dict(s) for s in d.get("stores", [])],
            n_steps=d["n_steps"],
            start_date=datetime.fromisoformat(d["start_date"]),
            world_seed=d["world_seed"],
            nodes=[NodeInstance.from_dict(n) for n in d.get("nodes", [])],
            edges=[_edge_from_dict(e) for e in d.get("edges", [])],
        )

    def catalog_df(self) -> Any:
        """One-row-per-Ware DataFrame; used by ``DataExporter`` + notebooks."""
        import pandas as pd

        rows = [
            {
                "product_id": w.product_id,
                "name": w.name,
                "category": w.category,
                "base_price": w.base_price,
                "unit_cost": w.unit_cost,
                # Derived margin so notebooks don't have to subtract.
                "margin": w.base_price - w.unit_cost,
                "seasonality": w.seasonality,
                "freshness_alpha": w.freshness_alpha,
                "freshness_decay": w.freshness_decay,
                "init_stage": w.init_stage,
                "stage_change_probs": w.stage_change_probs,
                "init_stock_share": w.init_stock_share,
                "related_products": list(w.related_products),
            }
            for w in self.catalog
        ]
        return pd.DataFrame(rows)

    def stores_df(self) -> Any:
        """One-row-per-StoreInstance DataFrame.

        ``policy_class`` is filled in from live ``Policy`` instances;
        scenarios reconstructed via ``from_json`` show ``None`` because
        policies are intentionally not serialised.
        """
        import pandas as pd

        rows = [
            {
                "store_id": i,
                "template_id": instance.template.id,
                "region": instance.template.region,
                "init_seed": instance.init_seed,
                "policy_class": (
                    type(instance.policy).__name__
                    if instance.policy is not None
                    else None
                ),
            }
            for i, instance in enumerate(self.stores)
        ]
        return pd.DataFrame(rows)

    def market_df(self) -> Any:
        """Single-row DataFrame view of ``MarketParams`` (one column per field)."""
        import pandas as pd
        from dataclasses import fields as dc_fields

        return pd.DataFrame(
            [{f.name: getattr(self.market, f.name) for f in dc_fields(self.market)}]
        )

    def disruption_df(self) -> Any:
        """Single-row DataFrame view of ``DisruptionParams``."""
        import pandas as pd
        from dataclasses import fields as dc_fields

        return pd.DataFrame(
            [
                {
                    f.name: getattr(self.disruption, f.name)
                    for f in dc_fields(self.disruption)
                }
            ]
        )

    def lifecycle_df(self) -> Any:
        """Single-row DataFrame view of ``ItemLifecycleParams``."""
        import pandas as pd
        from dataclasses import fields as dc_fields

        return pd.DataFrame(
            [
                {
                    f.name: getattr(self.item_lifecycle, f.name)
                    for f in dc_fields(self.item_lifecycle)
                }
            ]
        )

    def summary_df(self) -> Any:
        """Single-row top-level summary (counts, seed, start date)."""
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "n_steps": self.n_steps,
                    "start_date": self.start_date,
                    "world_seed": self.world_seed,
                    "n_stores": len(self.stores),
                    "n_products": len(self.catalog),
                }
            ]
        )

    def nodes_df(self) -> Any:
        """One-row-per-NodeInstance DataFrame.

        Columns: ``node_id``, ``node_type``, ``region``, ``init_seed``,
        ``policy_class`` (``None`` when policy not attached).
        """
        import pandas as pd

        rows = [
            {
                "node_id": ni.node.id,
                "node_type": type(ni.node).__name__,
                "region": ni.node.region,
                "init_seed": ni.init_seed,
                "policy_class": (
                    type(ni.policy).__name__ if ni.policy is not None else None
                ),
            }
            for ni in self.nodes
        ]
        return pd.DataFrame(rows)

    def edges_df(self) -> Any:
        """One-row-per-EdgeSpec DataFrame.

        Columns: ``supplier_id``, ``buyer_id``, ``default_lead_time``.
        """
        import pandas as pd

        rows = [
            {
                "supplier_id": e.supplier_id,
                "buyer_id": e.buyer_id,
                "default_lead_time": e.default_lead_time,
            }
            for e in self.edges
        ]
        return pd.DataFrame(rows)

    @classmethod
    def from_world(
        cls,
        world: "World",
        *,
        disruption: "DisruptionParams",
        item_lifecycle: "ItemLifecycleParams",
        stores: "list[StoreInstance] | None" = None,
        nodes: "list[NodeInstance] | None" = None,
        edges: "list[Any] | None" = None,
        n_steps: int,
        start_date: datetime,
        world_seed: int,
    ) -> "Scenario":
        """Build a ``Scenario`` from an LLM-generated ``World`` + author-supplied pieces.

        ``World`` provides the catalog and market; the caller fills in
        disruption parameters, lifecycle defaults, the per-store roster
        (legacy) or graph topology (Phase 4+), and the seeds.

        Phase 4 (issue 11): ``stores`` is deprecated and optional; pass
        ``nodes`` + ``edges`` instead for graph-mode scenarios.
        """
        return cls(
            catalog=world.catalog,
            market=world.market,
            disruption=disruption,
            item_lifecycle=item_lifecycle,
            stores=stores if stores is not None else [],
            nodes=nodes if nodes is not None else [],
            edges=edges if edges is not None else [],
            n_steps=n_steps,
            start_date=start_date,
            world_seed=world_seed,
        )

    @classmethod
    def from_json(cls, s: str) -> "Scenario":
        """Parse a JSON string into a ``Scenario`` (policies always ``None``)."""
        try:
            payload = json.loads(s)
        except json.JSONDecodeError as e:
            raise ValueError(f"Scenario.from_json: malformed JSON ({e})") from e
        if not isinstance(payload, dict):
            raise ValueError(
                "Scenario.from_json: top-level JSON must be an object"
            )
        return cls.from_dict(payload)


def make_stores(
    triples: Iterable[tuple[StoreTemplate, int, Any]],
) -> list[StoreInstance]:
    """Build a roster from a literal list of ``(template, init_seed, policy)``.

    The triple list *is* the roster. CRN comparisons are expressed by
    repeating ``(template, init_seed)`` with different policies; robustness
    sweeps by varying ``init_seed``; mixed rosters by writing the literal
    list. There is no regime abstraction above this — k-way comparisons,
    paired runs, and singletons are all the same shape.
    """
    # Materialise so we can validate non-emptiness and iterate twice
    # (the comprehension below).
    triples = list(triples)
    if not triples:
        raise ValueError("make_stores: triples must be non-empty")
    return [
        StoreInstance(template=template, init_seed=init_seed, policy=policy)
        for template, init_seed, policy in triples
    ]


def make_nodes(
    triples: Iterable[tuple[Any, int, Any]],
) -> list[NodeInstance]:
    """Build a graph node roster from a list of ``(node, init_seed, policy)`` triples.

    Mirrors ``make_stores``. The triple list is the roster; CRN comparisons
    are expressed by repeating ``(node, init_seed)`` with different policies.
    """
    triples = list(triples)
    if not triples:
        raise ValueError("make_nodes: triples must be non-empty")
    return [
        NodeInstance(node=node, init_seed=init_seed, policy=policy)
        for node, init_seed, policy in triples
    ]


def load_scenario_from_path(path: str | Path) -> "Scenario":
    """Import ``path`` as a Python module and return its ``scenario`` symbol.

    The returned ``Scenario`` has live ``Policy`` instances on each
    ``StoreInstance`` (as authored in the script).  Use this instead of
    ``Scenario.from_json`` when you need policies attached — e.g. in
    notebooks or the CLI — because ``from_json`` is for historical-run
    inspection and always returns ``policy=None``.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ImportError: if the module spec cannot be built.
        AttributeError: if the loaded module has no ``scenario`` attribute.
        TypeError: if ``module.scenario`` is not a ``Scenario`` instance.
    """
    # Coerce string input to Path for the existence check.
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"load_scenario_from_path: scenario file not found: {path}"
        )

    # Build a private module spec — the name is prefixed with an
    # underscore + the file stem so multiple loads don't collide in
    # ``sys.modules``.
    spec = importlib.util.spec_from_file_location(f"_scenario_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"load_scenario_from_path: could not build module spec for {path}"
        )
    module = importlib.util.module_from_spec(spec)
    # Register the module before executing it so relative imports inside
    # the scenario file resolve correctly.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    if not hasattr(module, "scenario"):
        raise AttributeError(
            f"load_scenario_from_path: {path} does not expose a top-level"
            " `scenario` attribute"
        )
    obj = module.scenario
    if not isinstance(obj, Scenario):
        raise TypeError(
            f"load_scenario_from_path: {path}.scenario is"
            f" {type(obj).__name__}, expected Scenario"
        )
    return obj
