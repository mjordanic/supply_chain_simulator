"""100-item fashion-retail world, built via LLM and cached locally.

Requires ``OPENAI_API_KEY`` in the environment on the first run
(cache miss). Subsequent runs load from
``data/worlds/fashion_retail_100/world.json`` silently.

Run it directly::

    OPENAI_API_KEY=... uv run python scenarios/llm_world_100.py

or via the CLI shim::

    OPENAI_API_KEY=... uv run python main.py scenarios/llm_world_100.py
"""

from __future__ import annotations

import sys
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
from src.sim.policy import BaselinePolicy
from src.sim.runner import Runner
from src.sim.scenario import (
    DisruptionParams,
    ItemLifecycleParams,
    Scenario,
    make_stores,
)


# Build parameters.
_ARCHETYPE = "fashion_retail"
_N_ITEMS = 100
_N_STORES = 3


# Build (or load from cache) the LLM-generated world.
_client = OpenAIClient()
_builder = WorldBuilder(archetype=_ARCHETYPE, client=_client)
_world = load_or_build_world(
    "fashion_retail_100",
    lambda: _builder.build(n_items=_N_ITEMS),
)


# First template returned by the LLM — sized for a generic fashion store.
_template = next(iter(_world.store_templates.values()))


# Author the policy.
_policy = BaselinePolicy(
    policy_seed=1000,
    min_qty=1,
    init_qty_factor=0.3,
    min_promo_len=3,
    max_promo_len=5,
    promo_cd_len=5,
    review_interval=10,
    promo_threshold=0.4,
    target_active_count=4,
    slow_sales_limit=2,
    history_window=4,
    max_history=50,
    promo_discount=0.7,
)


# Wire LLM-derived catalog/market into a complete Scenario.
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
        # Strict-terminal lifecycle for this demo.
        default_stage_change_probs={
            s: 0.0
            for s in ["introduction", "growth", "maturity", "decline", "dead"]
        },
    ),
    stores=make_stores(
        [(_template, 1 + i, _policy) for i in range(_N_STORES)]
    ),
    n_steps=50,
    start_date=datetime(2024, 1, 1),
    world_seed=42,
)


def main() -> None:
    """Run the scenario and dump artifacts to ``data/llm_world_100``."""
    run_log = Runner(scenario).run()
    output = _PROJECT_ROOT / "data" / "llm_world_100"
    DataExporter(scenario, run_log).export_all(str(output))
    # Echo the world structure for sanity checks.
    print(f"Archetype: {_ARCHETYPE}")
    print(f"Catalog ({len(_world.catalog)} items):")
    for w in _world.catalog:
        print(f"  {w.product_id}  {w.category:<20}  {w.name}")
    print(f"Templates: {list(_world.store_templates.keys())}")
    print(f"Regions: {_world.market.regions}")
    print(f"Wrote run artifacts to {output}")


if __name__ == "__main__":
    main()
