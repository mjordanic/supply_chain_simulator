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
from pathlib import Path

from src.sim.data_exporter import DataExporter
from src.sim.runner import Runner
from src.sim.scenario import Scenario, load_scenario_from_path


def _load_scenario(path: Path) -> Scenario:
    """Thin CLI wrapper around ``load_scenario_from_path``."""
    try:
        return load_scenario_from_path(path)
    except (FileNotFoundError, ImportError, AttributeError, TypeError) as exc:
        raise SystemExit(f"main.py: {exc}") from exc


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
