"""Published commands for the latent replenishment study.

The results notebook and the root README quote these strings. Importing
this module does not load the simulator; ``main`` imports it per subcommand
so the notebook's missing-artifact path stays cheap.
"""

from __future__ import annotations

import argparse
from pathlib import Path

COLLECT_COMMAND = (
    "uv run python -m src.jepa collect "
    "--episodes-each 64 "
    "--output runs/jepa-replenishment/log/transitions.parquet"
)

FIT_COMMAND = (
    "uv run python -m src.jepa fit "
    "--log runs/jepa-replenishment/log/transitions.parquet "
    "--output runs/jepa-replenishment/checkpoints"
)

TUNE_HOLDING_001_COMMAND = (
    "uv run python -m src.jepa tune "
    "--holding-rate 0.01 "
    "--trials 150 "
    "--output-dir runs/jepa-replenishment/tuning"
)

TUNE_HOLDING_005_COMMAND = (
    "uv run python -m src.jepa tune "
    "--holding-rate 0.05 "
    "--trials 150 "
    "--output-dir runs/jepa-replenishment/tuning"
)

EVAL_COMMAND = (
    "uv run python -m src.jepa eval "
    "--checkpoints runs/jepa-replenishment/checkpoints "
    "--anchor runs/jepa-replenishment/tuning/order_up_to_holding_0_01 "
    "--ceiling runs/jepa-replenishment/tuning/order_up_to_holding_0_05 "
    "--output runs/jepa-replenishment/synthetic"
)

M5_COMMAND = (
    "uv run jupyter nbconvert --to notebook --execute "
    "--output-dir runs/jepa-replenishment "
    "notebooks/11-jepa-m5.ipynb"
)


def _parser() -> argparse.ArgumentParser:
    from inspect import signature

    from src.jepa.log import EPISODES_EACH
    from src.jepa.plan import DEFAULT_BUDGET, fit_latent
    from src.tuning.config import TuningConfig

    fit_params = signature(fit_latent).parameters
    tuner = TuningConfig()
    parser = argparse.ArgumentParser(prog="python -m src.jepa")
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="Write the transition log.")
    collect.add_argument("--episodes-each", type=int, default=EPISODES_EACH)
    collect.add_argument("--episode-length", type=int, default=None)
    collect.add_argument("--k-active", type=int, default=None)
    collect.add_argument("--delivery-lag", type=int, default=None)
    collect.add_argument(
        "--output",
        type=Path,
        default=Path("runs/jepa-replenishment/log/transitions.parquet"),
    )

    fit = sub.add_parser("fit", help="Fit both predictors and save checkpoints.")
    fit.add_argument(
        "--log",
        type=Path,
        default=Path("runs/jepa-replenishment/log/transitions.parquet"),
    )
    fit.add_argument(
        "--output",
        type=Path,
        default=Path("runs/jepa-replenishment/checkpoints"),
    )
    fit.add_argument("--epochs", type=int, default=fit_params["epochs"].default)
    fit.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    fit.add_argument("--delivery-lag", type=int, default=fit_params["delivery_lag"].default)

    tune = sub.add_parser("tune", help="Tune OrderUpToPolicy at one holding rate.")
    tune.add_argument("--holding-rate", type=float, required=True)
    tune.add_argument("--trials", type=int, default=tuner.n_trials)
    tune.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/jepa-replenishment/tuning"),
    )
    tune.add_argument("--episode-length", type=int, default=None)
    tune.add_argument("--k-active", type=int, default=None)
    tune.add_argument("--n-search-seeds", type=int, default=None)
    tune.add_argument("--n-holdout-seeds", type=int, default=None)

    evaluate = sub.add_parser("eval", help="Score the tuner holdout and write tables.")
    evaluate.add_argument(
        "--checkpoints",
        type=Path,
        default=Path("runs/jepa-replenishment/checkpoints"),
    )
    evaluate.add_argument("--anchor", type=Path, default=None)
    evaluate.add_argument("--ceiling", type=Path, default=None)
    evaluate.add_argument(
        "--output",
        type=Path,
        default=Path("runs/jepa-replenishment/synthetic"),
    )
    evaluate.add_argument("--n-seeds", type=int, default=None)
    evaluate.add_argument("--episode-length", type=int, default=None)
    evaluate.add_argument("--k-active", type=int, default=None)
    evaluate.add_argument("--delivery-lag", type=int, default=None)
    return parser


def _overrides(args: argparse.Namespace, names: tuple[str, ...]) -> dict[str, int]:
    """Flags the user set. ``None`` keeps the published default of the callee."""
    return {
        name: getattr(args, name)
        for name in names
        if getattr(args, name) is not None
    }


def _tuning_shape(args: argparse.Namespace, names: tuple[str, ...]) -> dict[str, int]:
    shape = _overrides(args, names)
    if "k_active" in shape:
        shape["K_active"] = shape.pop("k_active")
    return shape


def _policy_factory(directory: Path | None):
    if directory is None:
        return None
    import json

    path = directory / "study.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    params = meta["best_params"]
    from src.tuning.search_spaces import order_up_to_from_params

    def factory():
        return order_up_to_from_params(params)

    return factory


def _collect(args: argparse.Namespace) -> None:
    from src.jepa.log import collect_transition_log

    collect_transition_log(
        n_episodes_each=args.episodes_each,
        path=args.output,
        **_overrides(args, ("episode_length", "k_active", "delivery_lag")),
    )


def _fit(args: argparse.Namespace) -> None:
    import pandas as pd

    from src.jepa.eval import IN_DISTRIBUTION_HOLDING_RATE
    from src.jepa.plan import fit_direct, fit_latent, save_planners

    frame = pd.read_parquet(args.log)
    shared = {
        "delivery_lag": args.delivery_lag,
        "epochs": args.epochs,
        "budget": args.budget,
        "holding_rate": IN_DISTRIBUTION_HOLDING_RATE,
    }
    latent = fit_latent(frame, seed=0, policy_seed=0, **shared)
    direct = fit_direct(frame, seed=1, policy_seed=1, **shared)
    save_planners(latent, direct, args.output)


def _tune(args: argparse.Namespace) -> None:
    from src.tuning.config import TuningConfig
    from src.tuning.episode import make_synthetic_catalog
    from src.tuning.search_spaces import order_up_to_space
    from src.tuning.study import confirm_top_k, run_study

    config = TuningConfig(
        holding_rate=args.holding_rate,
        n_trials=args.trials,
        **_tuning_shape(
            args,
            ("episode_length", "k_active", "n_search_seeds", "n_holdout_seeds"),
        ),
    )
    catalog = make_synthetic_catalog(config.active_catalog_size())
    name = f"order_up_to_holding_{config.holding_rate:.2f}".replace(".", "_")
    output_dir = str(args.output_dir)
    study = run_study(
        order_up_to_space,
        catalog=catalog,
        tuning_config=config,
        study_name=name,
        output_dir=output_dir,
    )
    confirm_top_k(
        study,
        order_up_to_space,
        catalog=catalog,
        tuning_config=config,
        study_dir=str(Path(output_dir) / name),
    )


def _eval(args: argparse.Namespace) -> None:
    from src.jepa.eval import compare_synthetic, write_synthetic_comparison
    from src.jepa.plan import load_planners
    from src.tuning.config import TuningConfig

    latent, direct = load_planners(args.checkpoints)
    shape = _tuning_shape(args, ("episode_length", "k_active", "delivery_lag"))
    config = TuningConfig(**shape) if shape else None
    seed_count = {} if args.n_seeds is None else {"n_seeds": args.n_seeds}
    result = compare_synthetic(
        latent,
        direct,
        anchor_factory=_policy_factory(args.anchor),
        ceiling_factory=_policy_factory(args.ceiling),
        config=config,
        **seed_count,
    )
    write_synthetic_comparison(result, args.output)


def main(argv: list[str] | None = None) -> None:
    """Run one published subcommand. ``argv`` is the argument list without the program name."""
    args = _parser().parse_args(argv)
    if args.command == "collect":
        _collect(args)
    elif args.command == "fit":
        _fit(args)
    elif args.command == "tune":
        _tune(args)
    elif args.command == "eval":
        _eval(args)
