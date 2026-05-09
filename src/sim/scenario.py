"""Scenario dataclass and authoring helpers.

A ``Scenario`` is a flat declarative description of one experiment: catalog,
typed parameter bags (``MarketParams`` / ``DisruptionParams`` /
``ItemLifecycleParams``), the list of ``StoreInstance``s that will run inside
it, ``n_steps``, ``start_date``, and ``world_seed``. Replaces the six-dict /
two-level config system (``src/config.py`` + ``ConfigResolver``).

Stochastic fields take ``Distribution`` instances; sampling cadence is
determined by *where the distribution is consumed*, not by an
``initial_parameters`` / ``dynamic_parameters`` split.

Policies are *not* serialised — the LLM never authors policies. ``to_json``
omits the ``policy`` field of each ``StoreInstance``; ``from_json`` reads
back instances with ``policy=None`` and the caller re-attaches policies.
"""

from __future__ import annotations

import json
from collections import namedtuple
from dataclasses import dataclass, fields
from datetime import datetime
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping

if TYPE_CHECKING:
    from src.llm.world_builder import World

from src.sim.distributions import (
    Distribution,
    _REGISTRY as _DISTRIBUTION_REGISTRY,
    distribution_from_dict,
)


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
    """Build a catalog of ``Ware``s, assigning ``P{i:04d}`` ids in order."""
    out: list[Ware] = []
    for i, item in enumerate(items):
        kwargs = dict(item)
        kwargs.pop("product_id", None)
        related = kwargs.get("related_products", [])
        kwargs["related_products"] = [tuple(p) for p in related]
        out.append(Ware(product_id=f"P{i:04d}", **kwargs))
    return out


def _serialize(value: Any) -> Any:
    if isinstance(value, Distribution):
        return value.to_dict()
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(v) for v in value]
    return value


def _deserialize(value: Any) -> Any:
    if isinstance(value, dict):
        if "type" in value and value["type"] in _DISTRIBUTION_REGISTRY:
            return distribution_from_dict(value)
        return {k: _deserialize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deserialize(v) for v in value]
    return value


def _ware_to_dict(w: Ware) -> dict[str, Any]:
    d: dict[str, Any] = {}
    for k, v in w._asdict().items():
        if k == "related_products":
            d[k] = [list(p) for p in v]
        else:
            d[k] = _serialize(v)
    return d


def _ware_from_dict(d: Mapping[str, Any]) -> Ware:
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

    id: str
    region: str
    capacity: int | float | Distribution
    init_balance: int | float | Distribution
    init_stock_pct: float | Distribution
    delivery_lag: int | float | Distribution
    holding_rate: float | Distribution
    order_fee: int | float | Distribution
    init_active_count: int | float | Distribution
    init_active_products: list[str] | None = None
    init_freshness: Literal["baseline", "fresh"] = "baseline"

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _serialize(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "StoreTemplate":
        missing = set(_TEMPLATE_FIELDS) - set(d)
        if missing:
            raise ValueError(
                f"StoreTemplate.from_dict: missing fields {sorted(missing)}"
            )
        kwargs: dict[str, Any] = {k: _deserialize(d[k]) for k in _TEMPLATE_FIELDS}
        if "init_active_products" in d and d["init_active_products"] is not None:
            kwargs["init_active_products"] = list(d["init_active_products"])
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

    template: StoreTemplate
    init_seed: int
    policy: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "template": self.template.to_dict(),
            "init_seed": self.init_seed,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "StoreInstance":
        return cls(
            template=StoreTemplate.from_dict(d["template"]),
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

    cycle_len: int | Distribution
    cycle_amp: float | Distribution
    init_demand: float
    init_supply: float
    peak_factor: float | Distribution
    off_factor: float | Distribution
    season_months: dict[str, list[int]]
    regions: list[str]
    correlation: float
    trend_update_interval: int
    min_value: float
    max_value: float
    stage_multipliers: dict[str, float | Distribution]
    price_elasticity: float
    promo_multiplier: float
    demand_factor_min: float
    demand_divisor: float
    supply_factor_min: float
    supply_divisor: float
    demand_range: tuple[float, float]
    cross_inv_lo: float
    cross_inv_hi: float
    cross_factor_range: tuple[float, float]
    trend: Distribution
    demand_shock: Distribution
    supply_shock: Distribution
    base_demand: Distribution

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _serialize(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "MarketParams":
        out = {f.name: _deserialize(d[f.name]) for f in fields(cls) if f.name in d}
        # Tuple-typed range fields lose their tuple-ness through JSON; restore.
        for tuple_field in ("demand_range", "cross_factor_range"):
            if tuple_field in out and isinstance(out[tuple_field], list):
                out[tuple_field] = tuple(out[tuple_field])
        return cls(**out)


@dataclass
class DisruptionParams:
    event_prob: float
    types: list[str]
    regions: list[str]
    severity: Distribution
    duration: Distribution

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _serialize(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DisruptionParams":
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

    stages: list[str]
    init_stage: str | Distribution
    default_stage_change_probs: dict[str, float | Distribution]
    default_freshness_alpha: float | Distribution = 0.0
    default_freshness_decay: float | Distribution = 1.0
    default_init_stock_share: float | Distribution = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _serialize(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ItemLifecycleParams":
        return cls(**{f.name: _deserialize(d[f.name]) for f in fields(cls) if f.name in d})


_SCENARIO_REQUIRED = (
    "catalog",
    "market",
    "disruption",
    "item_lifecycle",
    "stores",
    "n_steps",
    "start_date",
    "world_seed",
)


@dataclass
class Scenario:
    """Flat declarative description of one experiment."""

    catalog: list[Ware]
    market: MarketParams
    disruption: DisruptionParams
    item_lifecycle: ItemLifecycleParams
    stores: list[StoreInstance]
    n_steps: int
    start_date: datetime
    world_seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalog": [_ware_to_dict(w) for w in self.catalog],
            "market": self.market.to_dict(),
            "disruption": self.disruption.to_dict(),
            "item_lifecycle": self.item_lifecycle.to_dict(),
            "stores": [s.to_dict() for s in self.stores],
            "n_steps": self.n_steps,
            "start_date": self.start_date.isoformat(),
            "world_seed": self.world_seed,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Scenario":
        missing = set(_SCENARIO_REQUIRED) - set(d)
        if missing:
            raise ValueError(f"Scenario.from_dict: missing keys {sorted(missing)}")
        return cls(
            catalog=[_ware_from_dict(w) for w in d["catalog"]],
            market=MarketParams.from_dict(d["market"]),
            disruption=DisruptionParams.from_dict(d["disruption"]),
            item_lifecycle=ItemLifecycleParams.from_dict(d["item_lifecycle"]),
            stores=[StoreInstance.from_dict(s) for s in d["stores"]],
            n_steps=d["n_steps"],
            start_date=datetime.fromisoformat(d["start_date"]),
            world_seed=d["world_seed"],
        )

    def catalog_df(self) -> Any:
        import pandas as pd

        rows = [
            {
                "product_id": w.product_id,
                "name": w.name,
                "category": w.category,
                "base_price": w.base_price,
                "unit_cost": w.unit_cost,
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
        import pandas as pd
        from dataclasses import fields as dc_fields

        return pd.DataFrame(
            [{f.name: getattr(self.market, f.name) for f in dc_fields(self.market)}]
        )

    def disruption_df(self) -> Any:
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

    @classmethod
    def from_world(
        cls,
        world: "World",
        *,
        disruption: "DisruptionParams",
        item_lifecycle: "ItemLifecycleParams",
        stores: "list[StoreInstance]",
        n_steps: int,
        start_date: datetime,
        world_seed: int,
    ) -> "Scenario":
        return cls(
            catalog=world.catalog,
            market=world.market,
            disruption=disruption,
            item_lifecycle=item_lifecycle,
            stores=stores,
            n_steps=n_steps,
            start_date=start_date,
            world_seed=world_seed,
        )

    @classmethod
    def from_json(cls, s: str) -> "Scenario":
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
    triples = list(triples)
    if not triples:
        raise ValueError("make_stores: triples must be non-empty")
    return [
        StoreInstance(template=template, init_seed=init_seed, policy=policy)
        for template, init_seed, policy in triples
    ]
