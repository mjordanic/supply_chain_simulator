"""CLI shim — load a ``scenarios/<file>.py`` and run it.

The scenario file owns all experiment configuration; this module owns
nothing more than argument parsing, dynamic import, and dispatching the
result to ``Runner`` + ``DataExporter``.

Usage::

    uv run python main.py scenarios/example_homogeneous.py
    uv run python main.py scenarios/example_paired_comparison.py --output /tmp/run

Errors emitted on a bad scenario path:

- ``main.py: scenario file not found: <path>`` — file does not exist
- ``main.py: <path> does not expose a top-level `scenario` symbol`` —
  module loaded but no top-level ``scenario =`` binding
- ``main.py: <path>.scenario is <type>, expected Scenario`` — wrong type

This entry point is intentionally tiny — every behavioural decision
lives in the scenario file itself, which is just a regular Python
module exposing a top-level ``scenario`` symbol of type ``Scenario``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.sim.data_exporter import DataExporter
from src.sim.runner import Runner
from src.sim.scenario import Scenario, load_scenario_from_path


def _load_scenario(path: Path) -> Scenario:
    """Thin CLI wrapper around ``load_scenario_from_path``.

    Maps the rich exceptions raised by ``load_scenario_from_path``
    (FileNotFoundError / ImportError / AttributeError / TypeError) into
    a single ``SystemExit`` with a ``main.py:`` prefix so the user sees
    a uniform error format.
    """
    try:
        return load_scenario_from_path(path)
    except (FileNotFoundError, ImportError, AttributeError, TypeError) as exc:
        raise SystemExit(f"main.py: {exc}") from exc


def main(argv: list[str] | None = None) -> None:
    """Parse args, load the scenario module, run it, and write artifacts."""
    # Argparse driver — argument list intentionally minimal.
    parser = argparse.ArgumentParser(
        description="Run a sim Scenario from a Python file."
    )
    # Required positional: path to the scenario module.
    parser.add_argument(
        "scenario_path",
        help="Path to a Python file exposing a top-level `scenario` symbol.",
    )
    # Optional output folder; default is ``data/<stem>/``.
    parser.add_argument(
        "--output",
        default=None,
        help="Output folder for parquet/JSON/PNG artifacts. Default: data/<stem>.",
    )
    args = parser.parse_args(argv)

    # Absolute path so the loader can build a stable module spec.
    path = Path(args.scenario_path).resolve()
    # Imports the file as a Python module and returns its ``scenario`` symbol.
    scenario = _load_scenario(path)

    # Execute the simulation. ``Runner.run`` returns the accumulated run log.
    run_log = Runner(scenario).run()

    # Resolve the output folder: explicit flag wins; otherwise default
    # to ``data/<file_stem>`` so two scenarios don't collide.
    output = Path(args.output) if args.output else Path("data") / path.stem
    DataExporter(scenario, run_log).export_all(str(output))
    # Surface the actual output folder so users (and tests) can find it.
    print(f"Wrote run artifacts to {output}")


if __name__ == "__main__":
    main()
