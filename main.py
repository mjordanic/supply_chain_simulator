"""CLI entry point.

Two subcommands:

    uv run python main.py run <setup-dir> [--output DIR]
        Load a setup directory (``catalog.csv`` + ``setup.yaml``), run the
        simulation, and write outputs to DIR (default: ``data/<setup-dir-name>/``).
        Outputs include ``nodes.parquet``, timeseries, run log, overview PNG,
        and a verbatim ``config/`` snapshot of the input files.

    uv run python main.py scaffold <catalog.csv> [options] --out <setup.yaml>
        Generate a starter ``nodes:``/``edges:`` block from a catalog CSV and
        write (or merge) it into a target ``setup.yaml``.

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


def _cmd_scaffold(args: argparse.Namespace) -> None:
    """Execute the ``scaffold`` subcommand."""
    import csv

    import yaml

    from src.sim.scenario import Ware
    from src.sim.topology_scaffolder import ScaffoldSpec, scaffold_topology

    catalog_path = Path(args.catalog_csv)
    if not catalog_path.is_file():
        raise SystemExit(f"main.py scaffold: catalog CSV not found: {catalog_path}")

    # Parse catalog CSV into Ware list
    wares: list[Ware] = []
    with catalog_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"main.py scaffold: empty CSV: {catalog_path}")
        for i, row in enumerate(reader):
            pid = row.get("product_id", f"P{i:04d}").strip()
            wares.append(
                Ware(
                    product_id=pid,
                    name=row.get("name", pid).strip(),
                    category=row.get("category", "").strip(),
                    related_products=[],
                    base_price=float(row.get("base_price", 1.0)),
                    unit_cost=float(row.get("unit_cost", 1.0)),
                    seasonality=row.get("seasonality", "").strip(),
                    init_stage=None,
                    stage_change_probs=None,
                    freshness_alpha=None,
                    freshness_decay=None,
                    init_stock_share=None,
                )
            )

    if not wares:
        raise SystemExit(f"main.py scaffold: no products found in {catalog_path}")

    spec = ScaffoldSpec(
        shop_count=args.shop_count,
        sink_density=args.sink_density,
        region=args.region,
        factory_capacity=args.factory_capacity,
        shop_capacity=args.shop_capacity,
        factory_lead_time=args.factory_lead_time,
        shop_lead_time=args.shop_lead_time,
    )

    topology = scaffold_topology(wares, spec)

    out_path = Path(args.out)
    if out_path.is_file():
        # Merge: load existing YAML, overwrite nodes/edges keys only.
        with out_path.open(encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        doc["nodes"] = topology["nodes"]
        doc["edges"] = topology["edges"]
        with out_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(doc, f, default_flow_style=False, sort_keys=False)
        print(
            f"Merged nodes/edges into {out_path} "
            f"({len(topology['nodes'])} nodes, {len(topology['edges'])} edges)"
        )
    else:
        # Emit: write a standalone nodes/edges YAML snippet.
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(topology, f, default_flow_style=False, sort_keys=False)
        print(
            f"Wrote {out_path} "
            f"({len(topology['nodes'])} nodes, {len(topology['edges'])} edges)"
        )


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

    # --- scaffold subcommand ---------------------------------------------
    scaffold_parser = subparsers.add_parser(
        "scaffold",
        help="Generate a starter nodes/edges block from a catalog CSV.",
    )
    scaffold_parser.add_argument(
        "catalog_csv",
        help="Path to a catalog.csv file.",
    )
    scaffold_parser.add_argument(
        "--out",
        required=True,
        help="Target setup.yaml: written fresh (nodes/edges only) or merged if it exists.",
    )
    scaffold_parser.add_argument(
        "--shop-count",
        type=int,
        default=1,
        dest="shop_count",
        help="Number of intermediate shop nodes. Default: 1.",
    )
    scaffold_parser.add_argument(
        "--sink-density",
        type=float,
        default=1.0,
        dest="sink_density",
        help="Fraction of (product × shop) pairs that get a sink. Default: 1.0.",
    )
    scaffold_parser.add_argument(
        "--region",
        default="US",
        help="Region applied to all nodes. Default: US.",
    )
    scaffold_parser.add_argument(
        "--factory-capacity",
        type=int,
        default=100,
        dest="factory_capacity",
        help="capacity_per_tick for each factory node. Default: 100.",
    )
    scaffold_parser.add_argument(
        "--shop-capacity",
        type=int,
        default=500,
        dest="shop_capacity",
        help="capacity for each shop node. Default: 500.",
    )
    scaffold_parser.add_argument(
        "--factory-lead-time",
        type=int,
        default=2,
        dest="factory_lead_time",
        help="lead_time on factory→shop edges. Default: 2.",
    )
    scaffold_parser.add_argument(
        "--shop-lead-time",
        type=int,
        default=1,
        dest="shop_lead_time",
        help="lead_time on shop→sink edges. Default: 1.",
    )

    args = parser.parse_args(argv)

    if args.command == "run":
        _cmd_run(args)
    elif args.command == "scaffold":
        _cmd_scaffold(args)
    else:
        parser.print_help()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
