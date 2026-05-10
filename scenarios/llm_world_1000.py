"""1000-item fashion-retail world, 5 stores, single policy.

The world is built once via the LLM and cached at
``data/worlds/fashion_retail_1000/world.json``. Subsequent runs reload it
silently. Approximate first-run cost: ~6 small calls + chunked
correlations (~20) + chunked freshness (~20) ≈ 46 LLM calls.

Run it directly::

    uv run python scenarios/llm_world_1000.py

or via the CLI shim::

    uv run python main.py scenarios/llm_world_1000.py

Notes on calibration for a 1000-item catalog:

- ``target_active_count=80`` — a fashion store carries roughly 5-15 % of a
  large back-catalog at any moment; 8 % keeps the active set diverse
  without overwhelming a single ``BaselinePolicy`` review pass.
- The LLM-authored templates are sized for a generic fashion store and
  do not know the catalog size. We rescale ``capacity``, ``init_balance``
  and ``init_active_count`` so a flagship can hold the active set plus
  reorder headroom, then use a single rescaled template across all five
  stores (varying ``init_seed`` for diversity in the starting roster).
- Lifecycle transitions are non-trivial: every SKU starts at ``maturity``
  but can drift toward ``decline``/``dead``, so the run exercises the PLC
  machinery (ADR 0001/0002) instead of freezing every product at maturity.
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
_N_ITEMS = 1000
_N_STORES = 5
_N_STEPS = 2000

# Calibration constants (see module docstring for rationale).
_TARGET_ACTIVE = 80
_FLAGSHIP_CAPACITY = 5_000
_FLAGSHIP_BALANCE = 250_000.0
# Higher than _TARGET_ACTIVE so the policy has slack to deactivate slow movers.
_INIT_ACTIVE_COUNT = 100


# Build / load the LLM world. Larger ``max_retries`` because chunked
# correlation calls have a tail of borderline-malformed responses.
_client = OpenAIClient()
_builder = WorldBuilder(archetype=_ARCHETYPE, client=_client, max_retries=5)
_world = load_or_build_world(
    f"{_ARCHETYPE}_{_N_ITEMS}",
    lambda: _builder.build(n_items=_N_ITEMS),
)


# Rescale the first authored template for the 1000-item catalog.
_base_template = next(iter(_world.store_templates.values()))
_template = replace(
    _base_template,
    id="flagship_1000",
    capacity=_FLAGSHIP_CAPACITY,
    init_balance=_FLAGSHIP_BALANCE,
    init_active_count=_INIT_ACTIVE_COUNT,
    init_freshness="baseline",
)


# Policy: ~20 kwargs tuned for the larger catalog (longer cooldowns,
# tighter reorder factor, wider promo windows).
_policy = BaselinePolicy(
    policy_seed=1000,
    min_qty=5,
    init_qty_factor=0.25,
    review_interval=10,
    target_active_count=_TARGET_ACTIVE,
    promo_threshold=0.45,
    promo_discount=0.7,
    min_promo_len=4,
    max_promo_len=8,
    promo_cd_len=8,
    slow_sales_limit=4,
    history_window=10,
    max_history=20,
    reorder_factor=0.3,
    qty_factor=0.6,
    order_cd_len=7,
    order_cd_jitter=0.3,
    stock_lo_ratio=0.2,
    stock_hi_ratio=0.6,
    price_up_factor=1.08,
    price_down_factor=0.92,
    trend_threshold=0.05,
    cross_price_adj=0.05,
    inactive_price_factor=0.5,
)


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
        severity=Constant(1.2),
        duration=Constant(5),
    ),
    item_lifecycle=ItemLifecycleParams(
        stages=_LIFECYCLE_STAGES,
        init_stage="maturity",
        # Non-zero transitions so the PLC actually moves products
        # across stages during the run.
        default_stage_change_probs={
            "introduction": 0.04,
            "growth": 0.03,
            "maturity": 0.005,
            "decline": 0.05,
            "dead": 0.001,
        },
    ),
    stores=make_stores(
        [(_template, 1 + i, _policy) for i in range(_N_STORES)]
    ),
    n_steps=_N_STEPS,
    start_date=datetime(2024, 1, 1),
    world_seed=42,
)


def main() -> None:
    """Run the scenario and dump artifacts to ``data/llm_world_1000``."""
    run_log = Runner(scenario).run()
    output = _PROJECT_ROOT / "data" / "llm_world_1000"
    DataExporter(scenario, run_log).export_all(str(output))
    print(f"Archetype: {_ARCHETYPE}")
    print(f"Catalog: {len(_world.catalog)} items")
    print(f"Stores: {len(scenario.stores)} "
          f"(template={_template.id}, capacity={_template.capacity})")
    print(f"Regions: {_world.market.regions}")
    print(f"Wrote run artifacts to {output}")


if __name__ == "__main__":
    main()
