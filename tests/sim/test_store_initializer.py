"""Unit tests for the pure ``init_store_state`` extraction (issue 05).

Pin the bit-identity contract — same ``(template, init_seed, catalog)``
⇒ identical ``InitialStoreState`` — as one explicit assertion against
the extracted function instead of as an emergent property of
``Store.__init__``. Subsequent slices (06, 07, 08) plug new authoring
fields into this seam.
"""

from __future__ import annotations

import pytest

from src.sim.distributions import Uniform
from src.sim.policy import NoopPolicy
from src.sim.scenario import StoreTemplate, Ware, load_catalog
from src.sim.store import Store
from src.sim.store_initializer import InitialStoreState, init_store_state


def _template() -> StoreTemplate:
    return StoreTemplate(
        id="t",
        region="US",
        capacity=Uniform(800, 1200),
        init_balance=10000,
        init_stock_pct=0.2,
        delivery_lag=3,
        holding_rate=0.005,
        order_fee=100,
        init_active_count=2,
    )


def _catalog() -> list[Ware]:
    return load_catalog(
        [
            {
                "name": "A",
                "category": "x",
                "related_products": [],
                "base_price": 10.0,
                "unit_cost": 5.0,
                "seasonality": "all_season",
            },
            {
                "name": "B",
                "category": "x",
                "related_products": [],
                "base_price": 20.0,
                "unit_cost": 8.0,
                "seasonality": "all_season",
            },
            {
                "name": "C",
                "category": "x",
                "related_products": [],
                "base_price": 15.0,
                "unit_cost": 6.0,
                "seasonality": "all_season",
            },
        ]
    )


def test_init_store_state_bit_identity():
    """Same (template, init_seed, catalog) ⇒ identical InitialStoreState."""
    template = _template()
    catalog = _catalog()

    s1 = init_store_state(template, init_seed=42, catalog=catalog)
    s2 = init_store_state(template, init_seed=42, catalog=catalog)

    assert s1 == s2


def test_init_store_state_different_seeds_diverge():
    """Sanity: different init_seed must NOT produce identical state."""
    template = _template()
    catalog = _catalog()

    s1 = init_store_state(template, init_seed=1, catalog=catalog)
    s2 = init_store_state(template, init_seed=2, catalog=catalog)

    # At minimum capacity (sampled from Uniform) must differ for these seeds.
    assert (s1.capacity, s1.active_ids) != (s2.capacity, s2.active_ids)


def test_init_store_state_returns_dataclass_with_expected_fields():
    """InitialStoreState carries the four field families the issue calls out."""
    state = init_store_state(_template(), init_seed=1, catalog=_catalog())

    assert isinstance(state, InitialStoreState)
    # Resolved scalars
    assert isinstance(state.capacity, float)
    assert isinstance(state.balance, float)
    # Active SKU set
    assert isinstance(state.active_ids, list)
    assert len(state.active_ids) == 2
    assert all(pid in {"P0000", "P0001", "P0002"} for pid in state.active_ids)
    # Per-product initial stock allocation
    assert set(state.inventory.keys()) == {"P0000", "P0001", "P0002"}
    assert all(isinstance(v, int) for v in state.inventory.values())
    # Activation tick map
    assert state.activation_tick == {}


def test_init_store_state_inventory_only_for_active():
    """Inactive catalog Wares get stock=0; active Wares share the budget."""
    state = init_store_state(_template(), init_seed=1, catalog=_catalog())

    active_set = set(state.active_ids)
    total_stock_budget = int(state.capacity * state.init_stock_pct)
    allocated = sum(state.inventory.values())

    for pid, stock in state.inventory.items():
        if pid in active_set:
            assert stock > 0
        else:
            assert stock == 0
    assert allocated <= total_stock_budget


# ----------------------------------------------- init_active_products (issue 06)


def _template_with_active(products: list[str] | None) -> StoreTemplate:
    return StoreTemplate(
        id="t",
        region="US",
        capacity=1000,
        init_balance=10000,
        init_stock_pct=0.2,
        delivery_lag=3,
        holding_rate=0.005,
        order_fee=100,
        init_active_count=2,
        init_active_products=products,
    )


def test_explicit_init_active_products_drives_store_active_items():
    """Explicit list ⇒ Store.active_items is exactly that list (and order)."""
    catalog = _catalog()
    template = _template_with_active(["P0002", "P0000"])

    store = Store(template, init_seed=1, policy=NoopPolicy(), catalog=catalog)

    assert store.active_items == ["P0002", "P0000"]
    # Inventory carries every catalog product, with stock only on the
    # explicitly active ones.
    assert set(store.inventory.keys()) == {"P0000", "P0001", "P0002"}
    assert store.inventory["P0001"] == 0
    assert store.inventory["P0000"] > 0
    assert store.inventory["P0002"] > 0


def test_explicit_init_active_products_independent_of_init_seed():
    """Explicit list short-circuits the random sample — same active set across seeds."""
    catalog = _catalog()
    template = _template_with_active(["P0001", "P0002"])

    s1 = Store(template, init_seed=1, policy=NoopPolicy(), catalog=catalog)
    s2 = Store(template, init_seed=999, policy=NoopPolicy(), catalog=catalog)

    assert s1.active_items == ["P0001", "P0002"]
    assert s2.active_items == ["P0001", "P0002"]


def test_unset_init_active_products_falls_back_to_random_sample():
    """``None`` ⇒ random sample of size init_active_count, deterministic given seed."""
    catalog = _catalog()
    template = _template_with_active(None)

    s1 = Store(template, init_seed=42, policy=NoopPolicy(), catalog=catalog)
    s2 = Store(template, init_seed=42, policy=NoopPolicy(), catalog=catalog)

    assert len(s1.active_items) == 2
    assert s1.active_items == s2.active_items
    # Active set is a subset of the catalog's product ids.
    assert set(s1.active_items).issubset({"P0000", "P0001", "P0002"})


def test_init_active_products_rejects_unknown_id():
    """A typo'd product id fails loudly rather than silently disappearing."""
    catalog = _catalog()
    template = _template_with_active(["P0000", "P9999"])

    with pytest.raises(ValueError, match="P9999"):
        Store(template, init_seed=1, policy=NoopPolicy(), catalog=catalog)


def test_explicit_init_active_products_empty_list_is_valid():
    """An empty explicit list ⇒ no products active at step 0."""
    catalog = _catalog()
    template = _template_with_active([])

    store = Store(template, init_seed=1, policy=NoopPolicy(), catalog=catalog)

    assert store.active_items == []
    # All inventory entries are zero because no products are active.
    assert all(stock == 0 for stock in store.inventory.values())


# ----------------------------------------------- init_freshness mode (issue 07)


def _template_with_freshness(mode: str) -> StoreTemplate:
    return StoreTemplate(
        id="t",
        region="US",
        capacity=1000,
        init_balance=10000,
        init_stock_pct=0.2,
        delivery_lag=3,
        holding_rate=0.005,
        order_fee=100,
        init_active_count=2,
        init_freshness=mode,
    )


def test_init_freshness_defaults_to_baseline():
    """``StoreTemplate.init_freshness`` defaults to ``"baseline"``."""
    template = _template()  # no init_freshness specified
    assert template.init_freshness == "baseline"


def test_init_freshness_baseline_keeps_step0_multiplier_one():
    """Baseline mode ⇒ ``Store.freshness_multiplier(pid, 0) == 1`` for
    every initial active SKU, even with a non-zero hype α active."""
    catalog = _catalog()
    template = _template_with_freshness("baseline")
    store = Store(
        template,
        init_seed=1,
        policy=NoopPolicy(),
        catalog=catalog,
        freshness_alpha=0.4,
        freshness_decay=30.0,
    )
    assert store.active_items, "sanity: initial roster non-empty"
    for pid in store.active_items:
        assert store.freshness_multiplier(pid, current_step=0) == 1.0


def test_init_freshness_fresh_yields_full_hype_at_step0():
    """Fresh mode ⇒ ``Store.freshness_multiplier(pid, 0) == 1 + α`` for
    every initial active SKU (grand-opening, full hype window)."""
    catalog = _catalog()
    template = _template_with_freshness("fresh")
    alpha = 0.4
    store = Store(
        template,
        init_seed=1,
        policy=NoopPolicy(),
        catalog=catalog,
        freshness_alpha=alpha,
        freshness_decay=30.0,
    )
    assert store.active_items, "sanity: initial roster non-empty"
    for pid in store.active_items:
        assert store.freshness_multiplier(pid, current_step=0) == pytest.approx(
            1.0 + alpha
        )


def test_init_freshness_fresh_does_not_set_tick_for_inactive_skus():
    """Only initial *active* SKUs get an ``activation_tick`` entry under
    fresh mode — inactive SKUs continue to evaluate to multiplier 1.0."""
    catalog = _catalog()
    template = _template_with_freshness("fresh")
    state = init_store_state(template, init_seed=1, catalog=catalog)
    assert set(state.activation_tick.keys()) == set(state.active_ids)
    assert all(v == 0 for v in state.activation_tick.values())


def test_init_freshness_baseline_leaves_activation_tick_empty():
    """Baseline mode ⇒ ``activation_tick`` is empty so initial active
    SKUs fall through ``Store.freshness_multiplier``'s "no entry → 1"
    branch, preserving pre-issue-07 behaviour."""
    template = _template_with_freshness("baseline")
    state = init_store_state(template, init_seed=1, catalog=_catalog())
    assert state.activation_tick == {}


# ---------------------------------------- init_stock_share weighted alloc (issue 08)


class _StubRegistry:
    """Minimal duck-type for ``init_store_state`` 's registry seam.

    The pure-function path only needs ``init_stock_share(pid) -> float``,
    so a real ``ItemRegistry`` (with its ``world_rng`` plumbing) is
    overkill for unit tests that pin the allocation contract.
    """

    def __init__(self, weights: dict[str, float]) -> None:
        self._weights = weights

    def init_stock_share(self, product_id: str) -> float:
        return self._weights[product_id]


def _template_for_share() -> StoreTemplate:
    """Template with a deterministic 100-unit budget across all 3 SKUs.

    capacity=500, init_stock_pct=0.2, init_active_count=3 ⇒ total_stock = 100.
    With three active SKUs the legacy even-split allocates 33/33/33 and
    drops the spare unit; weighted (2,1,1) splits the same 100 budget as
    50/25/25. The single-SKU active-count value (3) lets the explicit
    ``init_active_products=[…three ids…]`` path drive the test without
    rolling against ``init_rng.sample``.
    """
    return StoreTemplate(
        id="t",
        region="US",
        capacity=500,
        init_balance=10000,
        init_stock_pct=0.2,
        delivery_lag=3,
        holding_rate=0.005,
        order_fee=100,
        init_active_count=3,
        init_active_products=["P0000", "P0001", "P0002"],
    )


def test_weighted_init_stock_share_2_1_1_distributes_proportionally():
    """Weights ``(2, 1, 1)`` across three active SKUs of the same cost
    distribute the budget 2:1:1 within the ``capacity * init_stock_pct``
    cap.

    AC pin from the issue: the public surface is ``Store.inventory``
    after construction.
    """
    catalog = _catalog()
    template = _template_for_share()
    registry = _StubRegistry({"P0000": 2.0, "P0001": 1.0, "P0002": 1.0})

    state = init_store_state(
        template, init_seed=1, catalog=catalog, item_registry=registry
    )

    # 100 budget split 2:1:1 ⇒ 50/25/25.
    assert state.inventory == {"P0000": 50, "P0001": 25, "P0002": 25}


def test_weighted_init_stock_share_respects_budget_cap():
    """Allocated stock never exceeds ``capacity * init_stock_pct``."""
    catalog = _catalog()
    template = _template_for_share()
    # Asymmetric weights with rounding crumbs (7,3,1 over 100 budget ⇒
    # 63 + 27 + 9 = 99). The dropped crumb is the budget cap, not a bug.
    registry = _StubRegistry({"P0000": 7.0, "P0001": 3.0, "P0002": 1.0})

    state = init_store_state(
        template, init_seed=1, catalog=catalog, item_registry=registry
    )

    total_stock_budget = int(state.capacity * state.init_stock_pct)
    allocated = sum(state.inventory.values())
    assert allocated <= total_stock_budget
    # Sanity: the heavy SKU receives more than the light SKUs.
    assert state.inventory["P0000"] > state.inventory["P0001"]
    assert state.inventory["P0001"] > state.inventory["P0002"]


def test_uniform_default_share_preserves_legacy_even_split():
    """The catalog-wide default of ``1.0`` makes weighted allocation
    bit-identical to the pre-issue-08 even split — the precondition for
    the regression scenarios pass unchanged."""
    catalog = _catalog()
    template = _template_for_share()
    registry = _StubRegistry({"P0000": 1.0, "P0001": 1.0, "P0002": 1.0})

    weighted = init_store_state(
        template, init_seed=1, catalog=catalog, item_registry=registry
    )
    legacy = init_store_state(
        template, init_seed=1, catalog=catalog, item_registry=None
    )

    assert weighted.inventory == legacy.inventory


def test_no_registry_argument_falls_back_to_even_split():
    """Plain-Store path (T4 unit-test seam, registry-less) keeps using
    the legacy even-split branch."""
    catalog = _catalog()
    template = _template_for_share()
    state = init_store_state(template, init_seed=1, catalog=catalog)

    # 100 / 3 ⇒ 33 each, last unit dropped.
    assert state.inventory == {"P0000": 33, "P0001": 33, "P0002": 33}


def test_zero_weight_gives_zero_stock_to_that_sku():
    """A weight of 0 (e.g. an active item the author doesn't want
    pre-stocked) receives 0 stock; sibling actives split the full
    budget."""
    catalog = _catalog()
    template = _template_for_share()
    registry = _StubRegistry({"P0000": 1.0, "P0001": 1.0, "P0002": 0.0})

    state = init_store_state(
        template, init_seed=1, catalog=catalog, item_registry=registry
    )

    assert state.inventory["P0002"] == 0
    # 100 budget / 2 effective weights ⇒ 50 each.
    assert state.inventory["P0000"] == 50
    assert state.inventory["P0001"] == 50


def test_all_zero_weights_falls_back_to_even_split():
    """Pathological zero-sum weights ⇒ no allocation ratio exists, so
    the legacy even-split branch takes over (preserves the
    ``no-registry-no-weighted`` fallback intent)."""
    catalog = _catalog()
    template = _template_for_share()
    registry = _StubRegistry({"P0000": 0.0, "P0001": 0.0, "P0002": 0.0})

    state = init_store_state(
        template, init_seed=1, catalog=catalog, item_registry=registry
    )

    # Same as the registry-less even-split.
    assert state.inventory == {"P0000": 33, "P0001": 33, "P0002": 33}


def test_weighted_init_stock_share_through_store_with_real_registry():
    """End-to-end: a real ``ItemRegistry`` driving ``Store`` construction
    propagates per-``Ware`` ``init_stock_share`` overrides into the
    public ``Store.inventory`` surface (the AC's testing-notes pin)."""
    from random import Random

    from src.sim.item_registry import ItemRegistry
    from src.sim.scenario import ItemLifecycleParams

    stages = ["introduction", "growth", "maturity", "decline", "dead"]
    lifecycle = ItemLifecycleParams(
        stages=stages,
        init_stage="maturity",
        default_stage_change_probs={s: 0.0 for s in stages},
    )

    catalog = _catalog()
    catalog[0] = catalog[0]._replace(init_stock_share=2.0)
    catalog[1] = catalog[1]._replace(init_stock_share=1.0)
    catalog[2] = catalog[2]._replace(init_stock_share=1.0)

    registry = ItemRegistry(lifecycle, catalog, Random(0))
    template = _template_for_share()
    store = Store(
        template,
        init_seed=1,
        policy=NoopPolicy(),
        catalog=catalog,
        item_registry=registry,
    )

    # 100 budget split 2:1:1 ⇒ 50/25/25.
    assert store.inventory == {"P0000": 50, "P0001": 25, "P0002": 25}
