"""CLI entry point.

Two subcommands:

    uv run python main.py run <setup-dir> [--output DIR]
        Load a setup directory (``catalog.csv`` + ``setup.yaml``), run the
        simulation, and write outputs to DIR (default: ``data/<setup-dir-name>/``).
        Outputs include ``nodes.parquet``, timeseries, run log, overview PNG,
        and a verbatim ``config/`` snapshot of the input files.

    uv run python main.py scaffold ...
        (Reserved for issue 03 — topology scaffolder.)

The old "pass a ``scenarios/*.py`` path" entry point is removed as of issue 02.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.sim.data_exporter import DataExporter
from src.sim.runner import Runner
from src.sim.setup_io import load_setup


def _cmd_run(args: argparse.Namespace) -> None:
    """Execute the ``run`` subcommand."""
    setup_dir = Path(args.setup_dir)
    if not setup_dir.is_dir():
        raise SystemExit(f"main.py: setup directory not found: {setup_dir}")

    try:
        scenario = load_setup(setup_dir)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"main.py: {exc}") from exc

    # Resolve output folder: explicit flag wins; otherwise data/<dir-name>/
    output = Path(args.output) if args.output else Path("data") / setup_dir.name

    run_log = Runner(scenario).run()
    exporter = DataExporter(scenario, run_log)
    exporter.export_all(str(output))

    # Copy input files verbatim into config/ as the reproducibility snapshot.
    config_dir = output / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(setup_dir / "catalog.csv", config_dir / "catalog.csv")
    shutil.copy2(setup_dir / "setup.yaml", config_dir / "setup.yaml")

    # Write nodes.parquet (replaces stores.parquet from the old model).
    exporter.save_nodes_parquet(str(output))

    print(f"Wrote run artifacts to {output}")


def main(argv: list[str] | None = None) -> None:
    """Parse args and dispatch to the right subcommand."""
    parser = argparse.ArgumentParser(
        description="Supply-chain simulator CLI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # --- run subcommand ---------------------------------------------------
    run_parser = subparsers.add_parser(
        "run",
        help="Load a setup directory and run the simulation.",
    )
    run_parser.add_argument(
        "setup_dir",
        help="Path to a directory containing catalog.csv and setup.yaml.",
    )
    run_parser.add_argument(
        "--output",
        default=None,
        help="Output folder for artifacts. Default: data/<setup-dir-name>/.",
    )

    # --- scaffold subcommand (placeholder for issue 03) ------------------
    subparsers.add_parser(
        "scaffold",
        help="(Coming in issue 03) Generate a starter topology from a catalog.",
    )

    args = parser.parse_args(argv)

    if args.command == "run":
        _cmd_run(args)
    elif args.command == "scaffold":
        raise SystemExit("main.py: scaffold is not yet implemented (issue 03).")
    else:
        parser.print_help()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
