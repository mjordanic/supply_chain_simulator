"""20-item fashion-retail world, single shop, single policy.

The world is built once via the LLM and cached at
``data/worlds/fashion_retail_20/world.json``. Subsequent runs reload it
silently. Approximate first-run cost: ~6 small calls + chunked
correlations + chunked freshness.

Run it directly::

    uv run python scenarios/llm_world_20.py

or via the CLI shim::

    uv run python main.py scenarios/llm_world_20.py

Phase 4 (issue 11): migrated to the graph engine. The rescaled store
template is expanded into a factory→shop→sink sub-graph via
``world_to_graph``; policies are then attached to the synthesised nodes.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Pull ``OPENAI_API_KEY`` (and friends) out of ``.env`` before any
# OpenAI client is constructed.
load_dotenv()
# Verbose logging so the chunked LLM calls are visible during the
# (slow) first-run build.
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.llm.openai_client import OpenAIClient
from src.llm.world_builder import WorldBuilder, load_or_build_world
from src.sim.data_exporter import DataExporter
from src.sim.distributions import Constant, Normal, Uniform
from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode
from src.sim.policy import (
    DefaultDemandSinkPolicy,
    OrderUpToPolicy,
    StaticFactoryPolicy,
)
from src.sim.runner import Runner
from src.sim.scenario import (
    DisruptionParams,
    ItemLifecycleParams,
    Scenario,
)
from src.sim.world import world_to_graph


# Build parameters.
_ARCHETYPE = "fashion_retail"
_N_ITEMS = 20
_N_STORES = 1
_N_STEPS = 500

# Calibration constants (see module docstring for rationale).
_TARGET_ACTIVE = 10
_FLAGSHIP_CAPACITY = 10000
_FLAGSHIP_BALANCE = 1000000.0
# Higher than _TARGET_ACTIVE so the policy has slack to deactivate slow movers.
_INIT_ACTIVE_COUNT = 10


# Build / load the LLM world. Larger ``max_retries`` because chunked
# correlation calls have a tail of borderline-malformed responses.
_client = OpenAIClient()
_builder = WorldBuilder(archetype=_ARCHETYPE, client=_client, max_retries=5)
_world = load_or_build_world(
    f"{_ARCHETYPE}_{_N_ITEMS}",
    lambda: _builder.build(n_items=_N_ITEMS),
)


# Rescale the first authored template for this catalog.
_base_template = next(iter(_world.store_templates.values()))
_template = replace(
    _base_template,
    id="flagship_20",
    capacity=_FLAGSHIP_CAPACITY,
    init_balance=_FLAGSHIP_BALANCE,
    init_active_count=_INIT_ACTIVE_COUNT,
    init_freshness="baseline",
    init_stock_pct=0.0,
    order_fee=10.0,
)


# Cover the same horizon as the store's delivery lead time. Guard
# against the Distribution case — if the LLM ever emits a sampled
# delivery_lag, the caller must sample it (with the store's
# ``init_seed``) rather than silently casting to int.
if not isinstance(_template.delivery_lag, (int, float)):
    raise TypeError(
        f"delivery_lag must be scalar for this scenario, got "
        f"{type(_template.delivery_lag).__name__}"
    )
_COVER_HORIZON_TICKS = int(_template.delivery_lag)


# Policy factory: one ``OrderUpToPolicy`` per shop. A fresh instance per
# shop is REQUIRED for the TextbookReorderPolicy family — the
# rate-estimator logs (``sales_log`` / ``inv_before_settle_log``) are
# keyed by ``pid`` only, so sharing one instance across shops would
# conflate their per-pid sales into a single log and corrupt the rate
# estimate. Distinct ``policy_seed`` per shop also keeps any
# ``policy_rng`` draws independent.
def _build_policy(seed: int) -> OrderUpToPolicy:
    return OrderUpToPolicy(policy_seed=seed,
        cover_horizon_ticks=_COVER_HORIZON_TICKS,
        safety_lead_pct_of_lag=1.0,
        opening_budget_pct=0.50,
        stockout_safety_bonus_pct_of_lag=1.0,
        min_qty=0)


# Graph engine (Phase 4): synthesise a topology from the World. Build one
# store template per shop so ``world_to_graph`` emits ``_N_STORES``
# independent factory→shop→sink sub-graphs (distinct node ids give each
# shop its own deterministic init seed, mirroring the old per-store
# ``init_seed`` diversity).
_world = replace(
    _world,
    store_templates={
        f"{_template.id}_{i}": replace(_template, id=f"{_template.id}_{i}")
        for i in range(_N_STORES)
    },
)
_topology = world_to_graph(_world, sink_density=1.0)


# Attach policies to the synthesised graph: a static factory, one fresh
# ``OrderUpToPolicy`` per shop, and the default greedy demand-sink policy.
_shop_idx = 0
for _ni in _topology["node_instances"]:
    _node = _ni.node
    if isinstance(_node, FactoryNode):
        _ni.policy = StaticFactoryPolicy(
            capacity_per_tick=_node.capacity_per_tick,
            unit_cost=_node.unit_cost,
        )
    elif isinstance(_node, IntermediateNode):
        _ni.policy = _build_policy(seed=1000 + _shop_idx)
        _shop_idx += 1
    elif isinstance(_node, DemandSinkNode):
        _ni.policy = DefaultDemandSinkPolicy()


# Stage list kept as a module constant so the comprehension below stays readable.
_LIFECYCLE_STAGES = ["introduction", "growth", "maturity", "decline", "dead"]
scenario = Scenario.from_world(
    _world,
    disruption=DisruptionParams(
        event_prob=0.03,
        # Wider disruption type pool — exercises every branch in
        # ``WorldEvent.apply``.
        types=[
            "natural_disaster",
            "economic_crisis",
            "pandemic",
            "political_unrest",
        ],
        regions=_world.market.regions,
        severity=Uniform(0.005, 0.02),
        duration=Uniform(5, 20),
    ),
    item_lifecycle=ItemLifecycleParams(
        stages=_LIFECYCLE_STAGES,
        init_stage="maturity",
        # Non-zero transitions so the PLC actually moves products
        # across stages during the run.
        default_stage_change_probs={
            "introduction": 0.004,
            "growth": 0.003,
            "maturity": 0.0005,
            "decline": 0.005,
            "dead": 0.0001,
        },
    ),
    nodes=_topology["node_instances"],
    edges=_topology["edges"],
    n_steps=_N_STEPS,
    start_date=datetime(2024, 1, 1),
    world_seed=42,
)


def main() -> None:
    """Run the scenario and dump artifacts to ``data/llm_world_20``."""
    run_log = Runner(scenario).run()
    output = _PROJECT_ROOT / "data" / "llm_world_20"
    DataExporter(scenario, run_log).export_all(str(output))
    n_shops = sum(
        1 for ni in scenario.nodes if isinstance(ni.node, IntermediateNode)
    )
    print(f"Archetype: {_ARCHETYPE}")
    print(f"Catalog: {len(_world.catalog)} items")
    print(f"Shops: {n_shops} "
          f"(template={_template.id}, capacity={_template.capacity})")
    print(f"Regions: {_world.market.regions}")
    print(f"Wrote run artifacts to {output}")


if __name__ == "__main__":
    main()
