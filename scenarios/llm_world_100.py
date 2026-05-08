"""Load a saved Scenario JSON and re-attach a policy before running.

``Scenario.to_json`` deliberately omits each store's policy (see
``src/sim/scenario.py``). This wrapper loads the JSON authored in
``notebooks/01-openai_world_builder.ipynb``, attaches a ``BaselinePolicy``
to every store, and exposes the resulting ``Scenario`` as the top-level
``scenario`` symbol expected by ``main.py``.

Run it directly::

    uv run python scenarios/llm_world_100.py

or via the CLI shim::

    uv run python main.py scenarios/llm_world_100.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.sim.data_exporter import DataExporter
from src.sim.policy import BaselinePolicy
from src.sim.runner import Runner
from src.sim.scenario import Scenario


_JSON_PATH = _PROJECT_ROOT / "data" / "llm_world_100" / "scenario.json"

scenario = Scenario.from_json(_JSON_PATH.read_text())

_policy = BaselinePolicy(
    policy_seed=1000,
    promo_threshold=0.4,
    promo_discount=0.7,
    review_interval=10,
    target_active_count=4,
)
for store in scenario.stores:
    store.policy = _policy


def main() -> None:
    run_log = Runner(scenario).run()
    output = _PROJECT_ROOT / "data" / "llm_world_100"
    DataExporter(scenario, run_log).export_all(str(output))
    print(f"Loaded scenario from {_JSON_PATH}")
    print(f"Catalog ({len(scenario.catalog)} items)")
    print(f"Stores: {len(scenario.stores)}")
    print(f"Regions: {scenario.market.regions}")
    print(f"Wrote run artifacts to {output}")


if __name__ == "__main__":
    main()
