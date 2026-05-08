"""Example: paired comparison of two policies on bit-identical world data.

Demonstrates ``make_stores(triples)`` with paired triples: each pair of
adjacent triples shares ``(template, init_seed)`` so the two stores in the
pair are bit-identical at step 0; they differ only in attached ``Policy``.
Combined with the world-rng seeding contract (``policy_rng`` and
``world_rng`` never share state) this gives Common-Random-Numbers variance
reduction across the two policy groups.

Output layout: pair ``i`` lives at indices ``2 * i`` (``policy_a``) and
``2 * i + 1`` (``policy_b``).

Run it directly::

    uv run python scenarios/example_paired_comparison.py

or hand the path to the CLI shim::

    uv run python main.py scenarios/example_paired_comparison.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.sim.data_exporter import DataExporter
from src.sim.distributions import Constant, Normal, Uniform
from src.sim.policy import BaselinePolicy
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


_MARKET = MarketParams(
    cycle_len=365,
    cycle_amp=0.1,
    init_demand=100.0,
    init_supply=100.0,
    peak_factor=1.2,
    off_factor=0.7,
    season_months={"all_season": list(range(1, 13))},
    regions=["US"],
    correlation=0.7,
    trend_update_interval=20,
    min_value=20.0,
    max_value=200.0,
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
    demand_divisor=100.0,
    supply_factor_min=0.01,
    supply_divisor=100.0,
    demand_range=(80.0, 120.0),
    cross_inv_lo=0.3,
    cross_inv_hi=0.7,
    cross_factor_range=(0.3, 1.6),
    trend=Constant(1.0),
    demand_shock=Normal(0.0, 5.0),
    supply_shock=Normal(0.0, 5.0),
    base_demand=Uniform(2, 8),
)


_DISRUPTION = DisruptionParams(
    event_prob=0.05,
    types=["natural_disaster", "economic_crisis"],
    regions=["US"],
    severity=Constant(1.0),
    duration=Constant(3),
)


_LIFECYCLE_STAGES = ["introduction", "growth", "maturity", "decline", "dead"]
_LIFECYCLE = ItemLifecycleParams(
    stages=_LIFECYCLE_STAGES,
    init_stage="maturity",
    default_stage_change_probs={s: 0.0 for s in _LIFECYCLE_STAGES},
)


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


# Two BaselinePolicy variants differing in how aggressively they discount
# slow-moving stock. Hyperparameters that don't matter for the contrast
# stay at module-level defaults.
_POLICY_AGGRESSIVE = BaselinePolicy(
    policy_seed=1001,
    promo_threshold=0.30,  # promote earlier
    promo_discount=0.6,    # bigger markdown
    review_interval=10,
)

_POLICY_CONSERVATIVE = BaselinePolicy(
    policy_seed=2002,
    promo_threshold=0.70,  # promote only when stock is heavy
    promo_discount=0.85,   # gentler markdown
    review_interval=10,
)


scenario = Scenario(
    catalog=_CATALOG,
    market=_MARKET,
    disruption=_DISRUPTION,
    item_lifecycle=_LIFECYCLE,
    stores=make_stores(
        [
            (_TEMPLATE, 1, _POLICY_AGGRESSIVE),
            (_TEMPLATE, 1, _POLICY_CONSERVATIVE),
            (_TEMPLATE, 2, _POLICY_AGGRESSIVE),
            (_TEMPLATE, 2, _POLICY_CONSERVATIVE),
        ]
    ),
    n_steps=50,
    start_date=datetime(2024, 1, 1),
    world_seed=42,
)


def main() -> None:
    """Run the scenario and dump artifacts to ``data/example_paired_comparison``."""
    run_log = Runner(scenario).run()
    output = _PROJECT_ROOT / "data" / "example_paired_comparison"
    DataExporter(scenario, run_log).export_all(str(output))
    print(f"Wrote run artifacts to {output}")


if __name__ == "__main__":
    main()
