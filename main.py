"""CLI shim — load a ``scenarios/<file>.py`` and run it.

The scenario file owns all experiment configuration; this module owns
nothing more than argument parsing, dynamic import, and dispatching the
result to ``Runner`` + ``DataExporter``.

Usage::

    uv run python main.py scenarios/example_homogeneous.py
    uv run python main.py scenarios/example_paired_comparison.py --output /tmp/run
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from src.sim.data_exporter import DataExporter
from src.sim.runner import Runner
from src.sim.scenario import Scenario


def _load_scenario(path: Path) -> Scenario:
    """Import ``path`` as a module and return its ``scenario`` symbol."""
    if not path.is_file():
        raise SystemExit(f"main.py: scenario file not found: {path}")

    spec = importlib.util.spec_from_file_location(f"_scenario_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"main.py: could not build module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "scenario"):
        raise SystemExit(
            f"main.py: {path} does not expose a top-level `scenario` symbol"
        )
    obj = module.scenario
    if not isinstance(obj, Scenario):
        raise SystemExit(
            f"main.py: {path}.scenario is {type(obj).__name__}, expected Scenario"
        )
    return obj


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a sim Scenario from a Python file."
    )
    parser.add_argument(
        "scenario_path",
        help="Path to a Python file exposing a top-level `scenario` symbol.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output folder for parquet/JSON/PNG artifacts. Default: data/<stem>.",
    )
    args = parser.parse_args(argv)

    path = Path(args.scenario_path).resolve()
    scenario = _load_scenario(path)

    run_log = Runner(scenario).run()

    output = Path(args.output) if args.output else Path("data") / path.stem
    DataExporter(scenario, run_log).export_all(str(output))
    print(f"Wrote run artifacts to {output}")


if __name__ == "__main__":
    main()
