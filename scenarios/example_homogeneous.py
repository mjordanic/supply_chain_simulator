"""Example: a homogeneous fleet of stores running a single policy.

Demonstrates ``make_stores(triples)`` with three triples that share one
``StoreTemplate`` and one ``HeuristicPolicy`` instance but vary
``init_seed`` so each store's step-0 active SKU set and stock allocation
is distinct. The right starting point for evaluating one policy across
several heterogeneous starting conditions (a "robustness sweep").

Run it directly::

    uv run python scenarios/example_homogeneous.py

or hand the path to the CLI shim::

    uv run python main.py scenarios/example_homogeneous.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Standalone execution: ensure the project root is importable so
# ``src.sim.*`` resolves regardless of CWD.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.sim.data_exporter import DataExporter
from src.sim.distributions import Constant, Normal, Uniform
from src.sim.policy import HeuristicPolicy
from src.sim.runner import Runner
from src.sim.scenario import (
    DisruptionParams,
    ItemLifecycleParams,
    MarketParams,
    Scenario,
    StoreTemplate,
    load_catalog,
    make_stores,
)


# Five-product toy catalog. ``load_catalog`` assigns ``P0000``–``P0004`` ids.
_CATALOG = load_catalog(
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
            # Cross-product correlation with Widget A — feeds into
            # ``Market.cross_demand_factor``.
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


# Hand-authored ``MarketParams`` — domain knobs + math defaults flat
# at one level. All seasonal months pooled into ``all_season`` so the
# example doesn't exercise seasonality.
_MARKET = MarketParams(
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
    # Constant trend ⇒ no drift; the cycle and shocks dominate.
    trend=Constant(1.0),
    demand_shock=Normal(0.0, 0.01),
    supply_shock=Normal(0.0, 0.01),
    base_demand=Uniform(2, 8),
)


# Low-probability disruption events with fixed severity / duration.
_DISRUPTION = DisruptionParams(
    event_prob=0.05,
    types=["natural_disaster", "economic_crisis"],
    regions=["US"],
    severity=Constant(0.01),
    duration=Constant(3),
)


# Canonical 5-stage product lifecycle list.
_LIFECYCLE_STAGES = ["introduction", "growth", "maturity", "decline", "dead"]
# Strict-terminal lifecycle: every transition probability is zero so
# every product stays at its ``init_stage`` for the whole run.
_LIFECYCLE = ItemLifecycleParams(
    stages=_LIFECYCLE_STAGES,
    init_stage="maturity",
    default_stage_change_probs={s: 0.0 for s in _LIFECYCLE_STAGES},
)


# Reusable store specification. Scalars only — no ``Distribution`` wraps,
# so every store from this template starts with identical numerics modulo
# ``init_rng`` selecting active SKUs.
_TEMPLATE = StoreTemplate(
    id="standard",
    region="US",
    capacity=200,
    init_balance=10000.0,
    init_stock_pct=0.4,
    delivery_lag=2,
    holding_rate=0.005,
    order_fee=10.0,
    init_active_count=3,
)


# Single ``HeuristicPolicy`` shared across all three stores. Sharing the
# instance also means a single ``policy_rng`` is consumed across all
# three stores' decision streams.
_POLICY = HeuristicPolicy(
    policy_seed=1000,
    min_qty=1,
    init_qty_factor=0.3,
    promo_len=Uniform(3, 5),
    promo_cd_len=5,
    review_interval=10,
    promo_threshold=0.4,
    target_active_count=4,
    slow_sales_limit=2,
    history_window=4,
    max_history=50,
    promo_discount=0.7,
)


# Scenario top-level binding — the CLI shim and standalone ``main``
# both pick this up.
scenario = Scenario(
    catalog=_CATALOG,
    market=_MARKET,
    disruption=_DISRUPTION,
    item_lifecycle=_LIFECYCLE,
    stores=make_stores(
        [
            # Three stores, same template + same policy, distinct seeds.
            (_TEMPLATE, 1, _POLICY),
            (_TEMPLATE, 2, _POLICY),
            (_TEMPLATE, 3, _POLICY),
        ]
    ),
    n_steps=50,
    start_date=datetime(2024, 1, 1),
    world_seed=42,
)


def main() -> None:
    """Run the scenario and dump artifacts to ``data/example_homogeneous``."""
    run_log = Runner(scenario).run()
    # Co-locate artifacts under the project's ``data/`` folder.
    output = _PROJECT_ROOT / "data" / "example_homogeneous"
    DataExporter(scenario, run_log).export_all(str(output))
    print(f"Wrote run artifacts to {output}")


if __name__ == "__main__":
    main()
