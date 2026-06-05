"""Phase-2 contention scenario: 2 factories + 2 shops + 2 sinks.

Topology (2F-2S contention scenario)::

    F_lo (cheap, slow) ──────────┐
                                  ├──► S1 ──► Sink1
    F_hi (premium, fast) ────────┤
                                  ├──► S2 ──► Sink2
    F_lo ────────────────────────┘

Both shops compete for inventory from both factories.  F_lo is cheaper
(unit_cost=5) but has limited capacity.  F_hi is more expensive
(unit_cost=8) but produces more per tick.

Both shops use ``OrderUpToPolicy`` (the canonical CRN anchor re-rooted on
``MultiSupplierTextbookPolicy``) and route orders to both factories via the
default cheapest-first strategy.

Expected dynamics:
- Early ticks: both shops prefer F_lo (cheapest).  When demand bursts,
  F_lo's stock depletes faster and its ``fill_rate_recent`` drops.
- Contention bursts produce a visible ``fill_rate_recent`` drop visible
  in the run log.
- Cash conservation holds: total system cash grows by
  ``sum(income_rate) × n_steps``.

Run::

    uv run python scenarios/example_two_factories_two_shops.py

Output is written to ``runs/example_two_factories_two_shops/``.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.sim.distributions import Constant, Normal
from src.sim.graph import EdgeSpec
from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode
from src.sim.policy import (
    DefaultDemandSinkPolicy,
    OrderUpToPolicy,
    StaticFactoryPolicy,
)
from src.sim.runner import Runner as GraphRunner
from src.sim.scenario import (
    DisruptionParams,
    ItemLifecycleParams,
    MarketParams,
    NodeInstance,
    Scenario,
    load_catalog,
)


# ---------------------------------------------------------------------------
# Scenario parameters
# ---------------------------------------------------------------------------

WORLD_SEED = 42
N_STEPS = 120

# Factory parameters.
F_LO_UNIT_COST = 5.0   # cheap factory
F_LO_CAPACITY = 30     # limited capacity — creates contention bursts
F_HI_UNIT_COST = 8.0   # premium factory
F_HI_CAPACITY = 60     # higher capacity

# Shop parameters.
SHOP_LIST_PRICE = 12.0
SHOP_CAPACITY = 300

# Sink parameters.
SINK_DEMAND_MEAN = 15
SINK_DEMAND_STD = 3
SINK_INCOME_RATE = 300.0


def _build_scenario() -> Scenario:
    catalog = load_catalog([
        {
            "name": "Widget",
            "category": "general",
            "related_products": [],
            "base_price": SHOP_LIST_PRICE,
            "unit_cost": F_LO_UNIT_COST,
            "seasonality": "all_season",
        }
    ])
    pid = catalog[0].product_id  # "P0000"

    # ── Factories ──────────────────────────────────────────────────────────
    f_lo = FactoryNode(
        id="f-lo",
        region="US",
        init_seed=1,
        produces_product_id=pid,
        unit_cost=F_LO_UNIT_COST,
        capacity_per_tick=F_LO_CAPACITY,
        inventory=100,
        list_price=F_LO_UNIT_COST,
        cash=0.0,
    )
    f_lo.policy = StaticFactoryPolicy(
        capacity_per_tick=F_LO_CAPACITY,
        unit_cost=F_LO_UNIT_COST,
        # Base-stock: bound inventory while staying generous enough to feed
        # BOTH shops (each draws up to ~(lag+safety)·rate + cover·rate ≈ 270
        # at rate 15). f_lo is the cheaper, preferred source so it gets the
        # larger target.
        target_inventory=300,
        policy_seed=10,
    )

    f_hi = FactoryNode(
        id="f-hi",
        region="US",
        init_seed=2,
        produces_product_id=pid,
        unit_cost=F_HI_UNIT_COST,
        capacity_per_tick=F_HI_CAPACITY,
        inventory=100,
        list_price=F_HI_UNIT_COST,
        cash=0.0,
    )
    f_hi.policy = StaticFactoryPolicy(
        capacity_per_tick=F_HI_CAPACITY,
        unit_cost=F_HI_UNIT_COST,
        # Base-stock buffer behind f_lo: covers residual shop pull when f_lo
        # runs dry, bounding inventory instead of piling up unsold stock.
        target_inventory=400,
        policy_seed=11,
    )

    # ── Shops (IntermediateNodes) using OrderUpToPolicy (multi-supplier) ────
    # delivery_lag=2 for the cheaper factory, will read lead time from the edge.
    # unit_cost is the average of both factories (approximate for pilot sizing).
    shop1 = IntermediateNode(
        id="shop-1",
        region="US",
        init_seed=3,
        carried_products={pid},
        capacity=SHOP_CAPACITY,
        tags=["shop"],
        inventory={pid: 30},
        pending={},
        list_prices={pid: SHOP_LIST_PRICE},
        min_order_imposed={pid: 0},
        cash=1000.0,
    )
    shop1.policy = OrderUpToPolicy(
        cover_horizon_ticks=14,
        safety_lead_pct_of_lag=1 / 3,
        delivery_lag=3,         # default delivery lag used by inner policy
        unit_cost=F_LO_UNIT_COST,
        list_price_out=SHOP_LIST_PRICE,
        policy_seed=20,
    )

    shop2 = IntermediateNode(
        id="shop-2",
        region="US",
        init_seed=4,
        carried_products={pid},
        capacity=SHOP_CAPACITY,
        tags=["shop"],
        inventory={pid: 30},
        pending={},
        list_prices={pid: SHOP_LIST_PRICE},
        min_order_imposed={pid: 0},
        cash=1000.0,
    )
    shop2.policy = OrderUpToPolicy(
        cover_horizon_ticks=14,
        safety_lead_pct_of_lag=1 / 3,
        delivery_lag=3,
        unit_cost=F_LO_UNIT_COST,
        list_price_out=SHOP_LIST_PRICE,
        policy_seed=21,
    )

    # ── Demand sinks ────────────────────────────────────────────────────────
    sink1 = DemandSinkNode(
        id="sink-1",
        region="US",
        init_seed=5,
        product_id=pid,
        demand_dist=Normal(mean=SINK_DEMAND_MEAN, std=SINK_DEMAND_STD),
        income_rate=SINK_INCOME_RATE,
        cash=1000.0,
    )
    sink1.policy = DefaultDemandSinkPolicy(policy_seed=30)

    sink2 = DemandSinkNode(
        id="sink-2",
        region="US",
        init_seed=6,
        product_id=pid,
        demand_dist=Normal(mean=SINK_DEMAND_MEAN, std=SINK_DEMAND_STD),
        income_rate=SINK_INCOME_RATE,
        cash=1000.0,
    )
    sink2.policy = DefaultDemandSinkPolicy(policy_seed=31)

    # ── Edges ───────────────────────────────────────────────────────────────
    # Both shops can source from both factories.
    # F_lo has a longer lead time (slow but cheap).
    # F_hi has a shorter lead time (fast but premium-priced).
    edges = [
        EdgeSpec(supplier_id="f-lo",    buyer_id="shop-1", default_lead_time=4),
        EdgeSpec(supplier_id="f-hi",    buyer_id="shop-1", default_lead_time=2),
        EdgeSpec(supplier_id="f-lo",    buyer_id="shop-2", default_lead_time=4),
        EdgeSpec(supplier_id="f-hi",    buyer_id="shop-2", default_lead_time=2),
        EdgeSpec(supplier_id="shop-1",  buyer_id="sink-1", default_lead_time=1),
        EdgeSpec(supplier_id="shop-2",  buyer_id="sink-2", default_lead_time=1),
    ]

    lifecycle = ItemLifecycleParams(
        stages=["introduction", "growth", "maturity", "decline", "dead"],
        init_stage="maturity",
        default_stage_change_probs={
            s: 0.0
            for s in ["introduction", "growth", "maturity", "decline", "dead"]
        },
    )
    disruption = DisruptionParams(
        event_prob=0.0,
        types=["natural_disaster"],
        regions=["US"],
        severity=Constant(0.0),
        duration=Constant(1),
    )
    market = MarketParams(
        cycle_len=365,
        cycle_amp=0.0,
        init_demand=1.0,
        init_supply=1.0,
        peak_factor=1.0,
        off_factor=1.0,
        season_months={},
        regions=["US"],
        correlation=0.0,
        trend_update_interval=100,
        min_value=0.5,
        max_value=2.0,
        stage_multipliers={
            "introduction": 1.0,
            "growth": 1.0,
            "maturity": 1.0,
            "decline": 1.0,
            "dead": 0.0,
        },
        price_elasticity=0.0,
        promo_multiplier=1.0,
        demand_factor_min=0.1,
        supply_factor_min=0.01,
        cross_inv_lo=0.3,
        cross_inv_hi=0.7,
        cross_factor_range=(0.5, 1.5),
        trend=Constant(1.0),
        demand_shock=Constant(0.0),
        supply_shock=Constant(0.0),
        base_demand=Constant(SINK_DEMAND_MEAN),
    )

    return Scenario(
        catalog=catalog,
        market=market,
        disruption=disruption,
        item_lifecycle=lifecycle,
        n_steps=N_STEPS,
        start_date=datetime(2024, 1, 1),
        world_seed=WORLD_SEED,
        nodes=[
            NodeInstance(node=f_lo,   init_seed=1),
            NodeInstance(node=f_hi,   init_seed=2),
            NodeInstance(node=shop1,  init_seed=3),
            NodeInstance(node=shop2,  init_seed=4),
            NodeInstance(node=sink1,  init_seed=5),
            NodeInstance(node=sink2,  init_seed=6),
        ],
        edges=edges,
    )


# Module-level ``scenario`` so the CLI shim (``main.py``) can load and run
# this file like any other scenario. ``main()`` below preserves the
# standalone smoke-test output when the file is executed directly.
scenario = _build_scenario()


def main() -> None:
    runner = GraphRunner(scenario)
    log = runner.run()

    out_dir = _PROJECT_ROOT / "runs" / "example_two_factories_two_shops"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "run_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, default=str)

    print(f"2F-2S contention scenario: {N_STEPS} ticks, world_seed={WORLD_SEED}")
    print(f"Run log written to: {log_path}")

    # Print summary.
    last_tick = log["ticks"][-1] if log["ticks"] else {}
    print("\nFinal state (last tick):")
    for node_id, cash in last_tick.get("node_cash", {}).items():
        inv = last_tick.get("node_inventory", {}).get(node_id, "N/A")
        print(f"  {node_id}: cash={cash:.2f}, inventory={inv}")

    # Check fill_rate_recent — should show contention drops for f_lo.
    print("\nChecking fill_rate_recent traces (f-lo vs f-hi)...")
    fill_rates_flo: list[float] = []
    for tick_log in log["ticks"]:
        fr = tick_log.get("fill_rate_recent", {})
        if isinstance(fr, dict):
            flo_rate = fr.get("f-lo", {})
            if isinstance(flo_rate, dict):
                for pid_fr in flo_rate.values():
                    fill_rates_flo.append(float(pid_fr))
                    break

    if fill_rates_flo:
        min_fr = min(fill_rates_flo)
        max_fr = max(fill_rates_flo)
        print(f"  f-lo fill_rate_recent range: [{min_fr:.3f}, {max_fr:.3f}]")
        if min_fr < 0.9:
            print("  Contention drops detected (fill_rate_recent < 0.9) — scenario working correctly.")
        else:
            print("  No contention drops detected (fill_rate_recent stayed high).")
    else:
        print("  fill_rate_recent data not available in log format.")


if __name__ == "__main__":
    main()
