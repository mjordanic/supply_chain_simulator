"""Example: build a world from an LLM, run it through the simulator.

Requires ``OPENAI_API_KEY`` in the environment. ``WorldBuilder.build``
issues a few small LLM calls (market domain, taxonomy, catalog naming,
chunked correlations, chunked freshness, store templates) and returns a
``World(catalog, market, store_templates)``. Disruption, item lifecycle,
policy, and seeds remain hand-authored here.

The ``load_or_build_world`` helper caches the produced ``World`` on
disk; the first run requires an OpenAI key, subsequent runs load
locally without any network access.

Run it directly::

    OPENAI_API_KEY=... uv run python scenarios/example_llm_world.py

or hand the path to the CLI shim::

    OPENAI_API_KEY=... uv run python main.py scenarios/example_llm_world.py

Phase 4 (issue 11): migrated to the graph engine via ``world_to_graph``.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

# Standalone execution: project root on the import path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.llm.openai_client import OpenAIClient
from src.llm.world_builder import WorldBuilder, load_or_build_world
from src.sim.data_exporter import DataExporter
from src.sim.distributions import Constant
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


# LLM build parameters — kept tiny to keep first-run cost low.
_ARCHETYPE = "fashion_retail"
_N_ITEMS = 12
_N_STORES = 3


# OpenAI client + WorldBuilder. The cache lookup happens inside
# ``load_or_build_world``: if ``data/worlds/fashion_retail_12/world.json``
# exists, no LLM calls are issued.
_client = OpenAIClient()
_builder = WorldBuilder(archetype=_ARCHETYPE, client=_client)
_world = load_or_build_world(
    "fashion_retail_12",
    lambda: _builder.build(n_items=_N_ITEMS),
)


# Take the first LLM-authored template (an arbitrary but stable choice).
_template = next(iter(_world.store_templates.values()))


# Author the policy here — never authored by the LLM. The legacy
# ``HeuristicPolicy`` was retired in Phase 4 (issue 11); the graph engine's
# shop equivalent is ``OrderUpToPolicy``. A fresh instance per shop keeps
# each shop's per-pid policy state independent.
def _build_policy(seed: int) -> OrderUpToPolicy:
    return OrderUpToPolicy(policy_seed=seed)


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
            # Base-stock target (4x capacity) bounds factory inventory.
            target_inventory=4 * _node.capacity_per_tick,
        )
    elif isinstance(_node, IntermediateNode):
        _ni.policy = _build_policy(seed=1000 + _shop_idx)
        _shop_idx += 1
    elif isinstance(_node, DemandSinkNode):
        _ni.policy = DefaultDemandSinkPolicy()


# ``Scenario.from_world`` merges the LLM-derived catalog/market with
# author-supplied disruption / lifecycle / graph topology / seeds.
scenario = Scenario.from_world(
    _world,
    disruption=DisruptionParams(
        event_prob=0.05,
        types=["natural_disaster", "economic_crisis"],
        regions=_world.market.regions,
        severity=Constant(0.01),
        duration=Constant(3),
    ),
    item_lifecycle=ItemLifecycleParams(
        stages=["introduction", "growth", "maturity", "decline", "dead"],
        init_stage="maturity",
        # Strict-terminal lifecycle for the demo (every transition prob = 0).
        default_stage_change_probs={
            s: 0.0
            for s in ["introduction", "growth", "maturity", "decline", "dead"]
        },
    ),
    nodes=_topology["node_instances"],
    edges=_topology["edges"],
    n_steps=50,
    start_date=datetime(2024, 1, 1),
    world_seed=42,
)


def main() -> None:
    """Run the scenario and dump artifacts to ``data/example_llm_world``."""
    run_log = Runner(scenario).run()
    output = _PROJECT_ROOT / "data" / "example_llm_world"
    DataExporter(scenario, run_log).export_all(str(output))
    # Echo the LLM-derived world structure so the operator can sanity-check.
    print(f"Archetype: {_ARCHETYPE}")
    print(f"Catalog ({len(_world.catalog)} items):")
    for w in _world.catalog:
        print(f"  {w.product_id}  {w.category:<20}  {w.name}")
    print(f"Templates: {list(_world.store_templates.keys())}")
    print(f"Regions: {_world.market.regions}")
    print(f"Wrote run artifacts to {output}")


if __name__ == "__main__":
    main()
