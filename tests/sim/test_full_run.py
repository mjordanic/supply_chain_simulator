"""T7: Full integration vertical run (issue 07).

A small ``Scenario`` (mini catalog, 2 stores, 50 steps) drives every
subsystem end-to-end. The acceptance criteria are:

- ``Runner`` runs the full per-step loop (observe → decide → advance →
  log) without raising.
- The run-log shape is complete: every store keyed, every product keyed,
  every metric series of length ``n_steps + 1``.
- At least one order is dispatched, one delivery arrives, one promotion
  fires and expires, and at least one product is activated or
  deactivated through the run.
- ``DataExporter`` accepts the new ``Scenario`` shape and produces
  parquet, JSON, and PNG outputs that re-parse cleanly.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import pandas as pd
import pytest

from src.sim.data_exporter import DataExporter
from src.sim.distributions import Constant, Normal, Uniform
from src.sim.policy import BaselinePolicy
from src.sim.runner import Runner
from src.sim.scenario import (
    DisruptionParams,
    ItemLifecycleParams,
    MarketParams,
    Scenario,
    StoreInstance,
    StoreTemplate,
    load_catalog,
)


N_STEPS = 50


def _mini_catalog():
    """5-product catalog so review_interval has products to cycle through."""
    return load_catalog(
        [
            {
                "name": "Widget A",
                "category": "Widgets",
                "related_products": [],
                "base_price": 20.0,
                "unit_cost": 12.0,
                "seasonality": "all_season",
            },
            {
                "name": "Widget B",
                "category": "Widgets",
                "related_products": [["Widget A", 0.5]],
                "base_price": 30.0,
                "unit_cost": 18.0,
                "seasonality": "all_season",
            },
            {
                "name": "Widget C",
                "category": "Widgets",
                "related_products": [],
                "base_price": 15.0,
                "unit_cost": 8.0,
                "seasonality": "all_season",
            },
            {
                "name": "Gadget A",
                "category": "Gadgets",
                "related_products": [],
                "base_price": 25.0,
                "unit_cost": 15.0,
                "seasonality": "all_season",
            },
            {
                "name": "Gadget B",
                "category": "Gadgets",
                "related_products": [],
                "base_price": 18.0,
                "unit_cost": 10.0,
                "seasonality": "all_season",
            },
        ]
    )


def _mini_market():
    return MarketParams(
        cycle_len=365,
        cycle_amp=0.001,
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
        },
        price_elasticity=-1.5,
        promo_multiplier=1.0,
        demand_factor_min=0.1,
        supply_factor_min=0.01,
        cross_inv_lo=0.3,
        cross_inv_hi=0.7,
        cross_factor_range=(0.3, 1.6),
        trend=Constant(1.0),
        # Demand shocks are loud enough that promos and stockouts both occur
        # within 50 steps without the test having to tune store math directly.
        demand_shock=Normal(0.0, 0.05),
        supply_shock=Normal(0.0, 0.05),
        base_demand=Uniform(2, 8),
    )


def _mini_disruption():
    return DisruptionParams(
        event_prob=0.05,
        types=["natural_disaster", "economic_crisis"],
        regions=["US"],
        severity=Constant(0.01),
        duration=Constant(3),
    )


def _mini_lifecycle():
    stages = ["introduction", "growth", "maturity", "decline", "dead"]
    return ItemLifecycleParams(
        stages=stages,
        init_stage="maturity",
        default_stage_change_probs={s: 0.0 for s in stages},
    )


def _mini_template():
    return StoreTemplate(
        id="mini",
        region="US",
        capacity=200,
        init_balance=10000.0,
        init_stock_pct=0.6,
        delivery_lag=2,
        holding_rate=0.005,
        order_fee=10.0,
        init_active_count=3,
    )


def _mini_policy(seed: int) -> BaselinePolicy:
    """BaselinePolicy tuned to surface every behaviour the test checks for.

    - ``promo_threshold=0.15`` is below the per-product allocation
      (capacity / n_active ~ 67 → initial 40 units already exceeds
      0.15 × 200 = 30), so promotions fire on the very first decide.
    - ``review_interval=10`` lands four catalog reviews inside 50 steps.
    - ``target_active_count=4`` invites activation; mini catalog is 5 items.
    - Promo windows of 3-5 steps mean any promo fires AND expires within
      50 steps.
    """
    return BaselinePolicy(
        policy_seed=seed,
        min_qty=1,
        init_qty_factor=0.3,
        promo_len=Uniform(3, 5),
        promo_cd_len=5,
        review_interval=10,
        promo_threshold=0.15,
        target_active_count=4,
        active_margin=0,
        slow_sales_limit=2,
        stock_lo_ratio=0.2,
        stock_hi_ratio=0.6,
        price_up_factor=1.1,
        price_down_factor=0.9,
        history_window=4,
        trend_threshold=0.05,
        cross_price_adj=0.05,
        max_history=50,
        inactive_price_factor=0.5,
        reorder_factor=0.3,
        qty_factor=0.5,
        order_cd_len=3,
        order_cd_jitter=0.0,
        promo_discount=0.7,
    )


def _mini_scenario() -> Scenario:
    template = _mini_template()
    return Scenario(
        catalog=_mini_catalog(),
        market=_mini_market(),
        disruption=_mini_disruption(),
        item_lifecycle=_mini_lifecycle(),
        stores=[
            StoreInstance(template=template, init_seed=1, policy=_mini_policy(seed=10)),
            StoreInstance(template=template, init_seed=2, policy=_mini_policy(seed=20)),
        ],
        n_steps=N_STEPS,
        start_date=datetime(2024, 1, 1),
        world_seed=42,
    )


@pytest.fixture(scope="module")
def full_run_log():
    return Runner(_mini_scenario()).run()


def test_run_log_top_level_keys_present(full_run_log):
    """Top-level run-log keys are intact."""
    assert set(full_run_log.keys()) >= {"global", "stores"}
    assert set(full_run_log["global"].keys()) >= {
        "time",
        "market_supply",
        "market_demand",
        "events",
        "products",
    }


def test_global_time_series_have_correct_length(full_run_log):
    """``simulation_step`` and ``simulation_date`` both have ``n_steps + 1`` entries."""
    assert len(full_run_log["global"]["time"]["simulation_step"]) == N_STEPS + 1
    assert len(full_run_log["global"]["time"]["simulation_date"]) == N_STEPS + 1


def test_global_market_state_has_correct_length(full_run_log):
    """Each region's supply / demand series spans the full timeline."""
    for region, series in full_run_log["global"]["market_supply"].items():
        assert len(series) == N_STEPS + 1, f"market_supply[{region}]"
    for region, series in full_run_log["global"]["market_demand"].items():
        assert len(series) == N_STEPS + 1, f"market_demand[{region}]"


def test_global_events_occurrences_length(full_run_log):
    """Event occurrences are appended once per ticked step (no step-0 event)."""
    assert len(full_run_log["global"]["events"]["occurrences"]) == N_STEPS


def test_global_lifecycle_per_product_length(full_run_log):
    """Every product has a lifecycle_stage series of length ``n_steps + 1``."""
    products = full_run_log["global"]["products"]
    catalog_pids = [w.product_id for w in _mini_catalog()]
    assert set(products.keys()) == set(catalog_pids)
    for pid, data in products.items():
        assert len(data["lifecycle_stage"]) == N_STEPS + 1, pid


def test_every_store_keyed_with_full_metric_series(full_run_log):
    """Every store has every catalog product, with each metric of length n_steps + 1."""
    catalog_pids = [w.product_id for w in _mini_catalog()]
    expected_metrics = {
        "inventory",
        "demand",
        "sales",
        "order_quantity",
        "outstanding_orders",
        "promotion_status",
        "active_status",
        "price",
        "revenue",
        "total_cost",
        "holding_cost",
        "profit",
    }

    assert set(full_run_log["stores"].keys()) == {0, 1}
    for store_id, store_log in full_run_log["stores"].items():
        assert len(store_log["balance"]) == N_STEPS + 1
        assert len(store_log["active_product_count"]) == N_STEPS + 1
        assert set(store_log["products"].keys()) == set(catalog_pids), store_id
        for pid, product_log in store_log["products"].items():
            assert set(product_log.keys()) == expected_metrics
            for metric, series in product_log.items():
                assert len(series) == N_STEPS + 1, (store_id, pid, metric)


def test_at_least_one_order_dispatched(full_run_log):
    """Across all stores and steps, BaselinePolicy places at least one order."""
    total_orders = 0
    for store_log in full_run_log["stores"].values():
        for product_log in store_log["products"].values():
            total_orders += sum(product_log["order_quantity"])
    assert total_orders > 0


def test_at_least_one_delivery_arrived(full_run_log):
    """At least one inventory increase across the run is attributable to a delivery.

    Detect deliveries by looking for any positive inventory delta against the
    baseline of "inventory only ever decreases via sales". The combined
    invariant inventory[t] − sales[t] − inventory[t-1] >= 0 means any
    positive value indicates a delivery between t-1 and t.
    """
    delivered = 0
    for store_log in full_run_log["stores"].values():
        for product_log in store_log["products"].values():
            inv = product_log["inventory"]
            sales = product_log["sales"]
            for t in range(1, len(inv)):
                # Stock can only increase (relative to the previous step
                # net of sales) via a delivery callback firing this step.
                delta = inv[t] - (inv[t - 1] - sales[t])
                if delta > 0:
                    delivered += 1
    assert delivered > 0


def test_at_least_one_promotion_fired_and_expired(full_run_log):
    """A promotion both starts and finishes within the run.

    Detected by finding a transition ``Regular Price → On Promotion → Regular Price``
    in the per-product promotion_status series.
    """
    found_full_cycle = False
    for store_log in full_run_log["stores"].values():
        for product_log in store_log["products"].values():
            on = False
            for status in product_log["promotion_status"]:
                if status == "On Promotion":
                    on = True
                elif on and status == "Regular Price":
                    found_full_cycle = True
                    break
            if found_full_cycle:
                break
        if found_full_cycle:
            break
    assert found_full_cycle, "no promotion fired AND expired within the run"


def test_at_least_one_activate_or_deactivate(full_run_log):
    """At least one product flips its active_status across the run."""
    flipped = False
    for store_log in full_run_log["stores"].values():
        for product_log in store_log["products"].values():
            statuses = product_log["active_status"]
            if any(a != b for a, b in zip(statuses, statuses[1:])):
                flipped = True
                break
        if flipped:
            break
    assert flipped, "no product was ever activated or deactivated during the run"


def test_balance_series_starts_at_initial_balance(full_run_log):
    """Step-0 balance equals the initial template-derived balance."""
    for store_log in full_run_log["stores"].values():
        # Initial template balance is 10000 (from _mini_template), and no
        # accounting has happened yet at the step-0 baseline.
        assert store_log["balance"][0] == 10000.0


# ---------------------------------------------------------------- DataExporter


@pytest.fixture(scope="module")
def exported_outputs(full_run_log, tmp_path_factory):
    """Run ``DataExporter.export_all`` once and return ``(folder, scenario, log)``."""
    folder = tmp_path_factory.mktemp("export")
    scenario = _mini_scenario()
    DataExporter(scenario, full_run_log).export_all(str(folder))
    return str(folder), scenario, full_run_log


def test_exporter_writes_scenario_json(exported_outputs):
    folder, scenario, _ = exported_outputs
    path = os.path.join(folder, "config", "scenario.json")
    assert os.path.exists(path)
    with open(path) as f:
        payload = json.load(f)
    # Round-trip survives the exporter.
    assert payload["world_seed"] == scenario.world_seed
    assert payload["n_steps"] == scenario.n_steps


def test_exporter_writes_run_log_json(exported_outputs):
    folder, _, log = exported_outputs
    path = os.path.join(folder, "data", "run_log.json")
    assert os.path.exists(path)
    with open(path) as f:
        payload = json.load(f)
    assert payload["global"]["time"]["simulation_step"] == log["global"]["time"][
        "simulation_step"
    ]


def test_exporter_writes_products_parquet(exported_outputs):
    folder, scenario, _ = exported_outputs
    path = os.path.join(folder, "data", "products.parquet")
    df = pd.read_parquet(path)
    assert len(df) == len(scenario.catalog)
    assert {"product_id", "name", "category", "base_price", "unit_cost"} <= set(
        df.columns
    )


def test_exporter_products_parquet_includes_resolved_freshness(exported_outputs):
    """Issue 04 AC: ``freshness_alpha`` / ``freshness_decay`` columns are
    populated for every catalog product. The mini scenario authors no
    overrides, so the columns must hold the catalog-wide defaults."""
    folder, scenario, _ = exported_outputs
    path = os.path.join(folder, "data", "products.parquet")
    df = pd.read_parquet(path)
    assert {"freshness_alpha", "freshness_decay"} <= set(df.columns)
    default_alpha = scenario.item_lifecycle.default_freshness_alpha
    default_decay = scenario.item_lifecycle.default_freshness_decay
    for alpha in df["freshness_alpha"]:
        assert alpha == default_alpha
    for decay in df["freshness_decay"]:
        assert decay == default_decay


def test_exporter_products_parquet_includes_resolved_init_stock_share(
    exported_outputs,
):
    """Issue 08 AC: ``init_stock_share`` column is populated for every
    catalog product. The mini scenario authors no overrides, so the
    column must hold the catalog-wide default ``1.0``."""
    folder, scenario, _ = exported_outputs
    path = os.path.join(folder, "data", "products.parquet")
    df = pd.read_parquet(path)
    assert "init_stock_share" in df.columns
    default_share = scenario.item_lifecycle.default_init_stock_share
    for share in df["init_stock_share"]:
        assert share == default_share


def test_exporter_writes_stores_parquet(exported_outputs):
    folder, scenario, _ = exported_outputs
    path = os.path.join(folder, "data", "stores.parquet")
    df = pd.read_parquet(path)
    assert len(df) == len(scenario.stores)
    assert df["policy_type"].iloc[0] == "BaselinePolicy"


def test_exporter_writes_timeseries_parquet(exported_outputs):
    folder, scenario, log = exported_outputs
    path = os.path.join(folder, "data", "timeseries.parquet")
    df = pd.read_parquet(path)
    n_stores = len(scenario.stores)
    n_products = len(scenario.catalog)
    n_steps = len(log["global"]["time"]["simulation_step"])
    assert len(df) == n_stores * n_products * n_steps
    assert {
        "simulation_step",
        "store_id",
        "product_id",
        "inventory",
        "price",
        "promotion_status",
    } <= set(df.columns)


def test_exporter_writes_overview_png(exported_outputs):
    folder, _, _ = exported_outputs
    path = os.path.join(folder, "reports", "overview.png")
    assert os.path.exists(path)
    # PNG signature is 8 bytes: 89 50 4E 47 0D 0A 1A 0A.
    with open(path, "rb") as f:
        head = f.read(8)
    assert head == b"\x89PNG\r\n\x1a\n"
