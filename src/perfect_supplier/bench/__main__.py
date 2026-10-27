"""Command line for the evaluation harness. See ``src/perfect_supplier/bench/README.md``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.perfect_supplier.bench import log
from src.perfect_supplier.bench.protocol import PROTOCOL_PATH, Protocol, config_sha256, freeze
from src.perfect_supplier.world import FAMILIES, VARIANTS


def _configs(paths: list[Path]) -> list[dict]:
    out = []
    for p in paths:
        c = json.loads(Path(p).read_text())
        out.extend(c if isinstance(c, list) else [c])
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.perfect_supplier.bench")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--track", default="H0-harness", help="log file experiments/log/<track>.jsonl")
    parser.add_argument("--world-family", choices=FAMILIES, default="perfect_supplier",
                        help="world family for eval, tune, calibrate (its parameters come from the protocol)")
    sub = parser.add_subparsers(dest="command", required=True)

    li = sub.add_parser("import-legacy", help="arm configs from a runs/perfect-supplier* root")
    li.add_argument("--root", type=Path, required=True)
    li.add_argument("--model-root", type=Path, default=None)
    li.add_argument("--variant", choices=VARIANTS, default="base")
    li.add_argument("--out", type=Path, required=True)

    e = sub.add_parser("eval", help="paired evaluation of arm configs on a seed block")
    e.add_argument("--arms", type=Path, nargs="+", required=True)
    e.add_argument("--split", required=True)
    e.add_argument("--n-worlds", type=int, required=True)
    e.add_argument("--offset", type=int, default=0)
    e.add_argument("--variant", choices=VARIANTS, default="base")
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--rerun-reason", default=None)
    e.add_argument("--note", default="")

    t = sub.add_parser("tune", help="sequential TPE with a process pool, budget from the protocol")
    t.add_argument("--kind", required=True)
    t.add_argument("--name", required=True)
    t.add_argument("--variant", choices=VARIANTS, default="base")
    t.add_argument("--cell-keys", nargs="+", default=None, help="tune per cell of these world fields")
    t.add_argument("--base-config", type=Path, default=None, help="fixed fields (model, forecaster) for the arm")
    t.add_argument("--out", type=Path, required=True)

    sl = sub.add_parser("select", help="score tuned textbook arms on common tune worlds; writes select_score")
    sl.add_argument("--arms", type=Path, nargs="+", required=True, help="one config per file, updated in place")
    sl.add_argument("--variant", choices=VARIANTS, default="base")
    sl.add_argument("--offset", type=int, required=True, help="into the tune block, past every tuner's seeds")
    sl.add_argument("--n-worlds", type=int, required=True)

    ft = sub.add_parser("fit", help="train one learned forecaster for one training seed")
    ft.add_argument("--root", type=Path, required=True, help="behaviour log of the protocol's train block")
    ft.add_argument("--arm", choices=("jepa", "direct"), required=True)
    ft.add_argument("--seed", type=int, required=True)
    ft.add_argument("--steps", type=int, default=20_000)
    ft.add_argument("--no-tail-quantiles", action="store_true", help="train on QUANTILES only")
    ft.add_argument("--out", type=Path, required=True)

    f = sub.add_parser("freeze", help="list arm configs in experiments/frozen.json")
    f.add_argument("--arms", type=Path, nargs="+", required=True)
    f.add_argument("--note", required=True)

    c = sub.add_parser("calibrate", help="lead-time coverage of a planner arm's imagined demand")
    c.add_argument("--arm", type=Path, required=True)
    c.add_argument("--split", default="dev")
    c.add_argument("--n-worlds", type=int, default=60)
    c.add_argument("--offset", type=int, default=0)
    c.add_argument("--variant", choices=VARIANTS, default="base")
    c.add_argument("--out", type=Path, required=True)

    r = sub.add_parser("report", help="markdown report from an eval directory")
    r.add_argument("--scores", type=Path, required=True, help="directory with scores.parquet and arms.json")
    r.add_argument("--reference", default=None,
                   help="default: the textbook arm with the best tune score; setting it makes the report exploratory")
    r.add_argument("--primary", nargs="+", default=None,
                   help="default: the protocol's primary family; setting it makes the report exploratory")
    r.add_argument("--secondary", nargs="+", default=None,
                   help="pairs a:b; default: the protocol's secondary family; setting it makes the report exploratory")
    r.add_argument("--strata", nargs="+", default=None, help="BH-adjusted, exploratory; default: the protocol's")
    r.add_argument("--title", default="Paired evaluation")
    r.add_argument("--out", type=Path, required=True)

    pp = sub.add_parser("pooled-primary", help="Holm once over the protocol's primary cells' test evals")
    pp.add_argument("--out", type=Path, required=True)

    b = sub.add_parser("backfill", help="write the historical results as kind=historical log lines")
    b.add_argument("--snapshot", type=Path, required=True, help="copy of the captain's runs/ (read-only)")

    args = parser.parse_args(argv)
    protocol = Protocol.load(args.protocol)

    if args.command == "import-legacy":
        from src.perfect_supplier.bench.arms import legacy_configs

        args.out.mkdir(parents=True, exist_ok=True)
        for config in legacy_configs(args.root, args.variant, args.model_root):
            (args.out / f"{config['name']}.json").write_text(json.dumps(config, indent=1))
            print(args.out / f"{config['name']}.json", config_sha256(config)[:12])
    elif args.command == "eval":
        from src.perfect_supplier.bench.evaluate import evaluate

        frame = evaluate(_configs(args.arms), args.split, args.n_worlds, args.variant, args.out, protocol,
                         args.track, offset=args.offset, rerun_reason=args.rerun_reason, note=args.note,
                         family=args.world_family)
        print(frame.groupby("arm")["score"].mean().round(4).to_string())
    elif args.command == "tune":
        from src.perfect_supplier.bench.tuner import tune

        base = json.loads(args.base_config.read_text()) if args.base_config else None
        config = tune(args.kind, args.name, args.variant, protocol, args.track, args.out,
                      cell_keys=args.cell_keys, base_config=base, family=args.world_family)
        print(json.dumps(config, indent=1))
    elif args.command == "select":
        from src.perfect_supplier.bench.tuner import select

        configs = [json.loads(p.read_text()) for p in args.arms]
        for path, config in zip(args.arms, select(configs, protocol, args.track, args.variant, offset=args.offset,
                                                  n_worlds=args.n_worlds)):
            path.write_text(json.dumps(config, indent=1))
            print(f"{config['name']}: select {config['select_score']:.5f} tune {config.get('tune_score')}")
    elif args.command == "fit":
        from src.perfect_supplier.bench.fit import fit_model

        config = fit_model(args.root, args.arm, args.seed, protocol, args.track, args.out, steps=args.steps,
                           tail_quantiles=not args.no_tail_quantiles)
        print(json.dumps(config, indent=1))
    elif args.command == "freeze":
        for config in _configs(args.arms):
            sha = freeze(config, args.note, protocol)
            log.append(args.track, {"kind": "freeze", "arm": config["name"], "variant": config.get("variant"),
                                    "config_sha256": sha, "config": config, "note": args.note, "status": "kept"},
                       protocol)
            print(config["name"], sha)
    elif args.command == "calibrate":
        from src.perfect_supplier.bench.calibration import calibrate

        (config,) = _configs([args.arm])
        table = calibrate(config, args.split, args.n_worlds, args.variant, protocol, args.track, args.out,
                          offset=args.offset, family=args.world_family)
        print(table.to_string())
    elif args.command == "report":
        from src.perfect_supplier.bench.report import write_report

        secondary = [tuple(p.split(":")) for p in args.secondary] if args.secondary else None
        text = write_report(args.scores, protocol, args.out, args.track, reference=args.reference,
                            primary=args.primary, secondary=secondary, strata=args.strata, title=args.title)
        print(text)
    elif args.command == "pooled-primary":
        from src.perfect_supplier.bench.report import write_pooled_primary

        print(write_pooled_primary(protocol, args.out, args.track))
    elif args.command == "backfill":
        from src.perfect_supplier.bench.backfill import backfill

        for row in backfill(args.snapshot, protocol, args.track):
            print(row["run_id"], row["note"][:90])


if __name__ == "__main__":
    main()
