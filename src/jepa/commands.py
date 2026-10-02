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

# Rates the holding-rate transfer sweep compares. One tuned OrderUpToPolicy each.
SWEEP_HOLDING_RATES = (0.005, 0.01, 0.02, 0.05, 0.1)


def order_up_to_study_name(holding_rate: float) -> str:
    """Directory name for one holding rate.

    Three decimal places, then trailing zeros stripped, so ``0.005`` does
    not collapse into ``0.01`` and the published ``0_01`` / ``0_05`` names stay.
    """
    text = f"{holding_rate:.3f}".rstrip("0").rstrip(".")
    return "order_up_to_holding_" + text.replace(".", "_")


def tune_command(holding_rate: float) -> str:
    """Published 150-trial command for one holding rate."""
    return (
        "uv run python -m src.jepa tune "
        f"--holding-rate {holding_rate:g} "
        "--trials 150 "
        "--output-dir runs/jepa-replenishment/tuning"
    )


TUNE_COMMANDS = tuple(tune_command(rate) for rate in SWEEP_HOLDING_RATES)

# Reduced width, one size below the published band, then three sizes inside it.
CAPACITY_GRID = ((32, 2), (128, 3), (256, 4), (384, 3), (512, 3))

SCALE_COMMAND = (
    "uv run python -m src.jepa scale "
    "--log runs/jepa-replenishment/log/transitions.parquet "
    "--output runs/jepa-replenishment/scaling"
)

EVAL_COMMAND = (
    "uv run python -m src.jepa eval "
    "--checkpoints runs/jepa-replenishment/checkpoints "
    "--log runs/jepa-replenishment/log/transitions.parquet "
    "--tuning-dir runs/jepa-replenishment/tuning "
    "--output runs/jepa-replenishment/synthetic"
)

M5_COMMAND = (
    "uv run jupyter nbconvert --to notebook --execute "
    "--output-dir runs/jepa-replenishment "
    "notebooks/11-jepa-m5.ipynb"
)


def _parser() -> argparse.ArgumentParser:
    from inspect import signature

    from src.jepa.dynamics import PUBLISHED_LR
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
    fit.add_argument("--lr", type=float, default=PUBLISHED_LR)
    fit.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    fit.add_argument("--delivery-lag", type=int, default=fit_params["delivery_lag"].default)
    fit.add_argument("--model-unroll-steps", type=int, default=None)

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
    evaluate.add_argument(
        "--tuning-dir",
        type=Path,
        default=Path("runs/jepa-replenishment/tuning"),
    )
    evaluate.add_argument(
        "--output",
        type=Path,
        default=Path("runs/jepa-replenishment/synthetic"),
    )
    evaluate.add_argument("--n-seeds", type=int, default=None)
    evaluate.add_argument("--episode-length", type=int, default=None)
    evaluate.add_argument("--k-active", type=int, default=None)
    evaluate.add_argument("--delivery-lag", type=int, default=None)
    evaluate.add_argument(
        "--log",
        type=Path,
        default=Path("runs/jepa-replenishment/log/transitions.parquet"),
    )

    scale = sub.add_parser("scale", help="Write the capacity and data-scaling curves.")
    scale.add_argument(
        "--log",
        type=Path,
        default=Path("runs/jepa-replenishment/log/transitions.parquet"),
    )
    scale.add_argument(
        "--output",
        type=Path,
        default=Path("runs/jepa-replenishment/scaling"),
    )
    scale.add_argument("--epochs", type=int, default=fit_params["epochs"].default)
    scale.add_argument("--lr", type=float, default=PUBLISHED_LR)
    scale.add_argument("--delivery-lag", type=int, default=fit_params["delivery_lag"].default)
    scale.add_argument("--sizes", type=str, default=None)
    scale.add_argument("--episodes", type=str, default=None)
    scale.add_argument("--model-unroll-steps", type=int, default=None)
    scale.add_argument("--embed-dim", type=int, default=None)
    scale.add_argument("--depth", type=int, default=None)
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
    return _order_up_to_factory(directory, holding_rate=None)


def _order_up_to_factory(directory: Path, holding_rate: float | None):
    import json

    path = directory / "study.json"
    if not path.is_file():
        rate = "" if holding_rate is None else f" for holding rate {holding_rate}"
        raise FileNotFoundError(f"missing tuned OrderUpToPolicy{rate}: {path}")
    meta = json.loads(path.read_text(encoding="utf-8"))
    recorded = (meta.get("episode_distribution") or {}).get("holding_rate")
    if holding_rate is not None:
        if recorded is None:
            raise ValueError(
                f"{directory.name} does not record the holding rate it was tuned at"
            )
        if abs(float(recorded) - float(holding_rate)) > 1e-9:
            raise ValueError(
                f"{directory.name} was tuned at holding rate {recorded}, not {holding_rate}"
            )
    params = meta["best_params"]
    from src.tuning.search_spaces import order_up_to_from_params

    def factory():
        return order_up_to_from_params(params)

    return factory


def sweep_factories(tuning_dir: Path) -> dict[float, object]:
    """One ``OrderUpToPolicy`` factory per sweep rate, from existing studies.

    Does not launch a tuning study. A missing directory, a study that does
    not record its holding rate, or a study recorded at a different rate
    fails before any episode is scored.
    """
    return {
        rate: _order_up_to_factory(Path(tuning_dir) / order_up_to_study_name(rate), rate)
        for rate in SWEEP_HOLDING_RATES
    }


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
        "lr": args.lr,
        "budget": args.budget,
        "holding_rate": IN_DISTRIBUTION_HOLDING_RATE,
    }
    if args.model_unroll_steps is not None:
        shared["model_unroll_steps"] = args.model_unroll_steps
    from src.jepa.dynamics import (
        PUBLISHED_DEPTH,
        PUBLISHED_DIRECT_HIDDEN,
        PUBLISHED_EMBED_DIM,
    )

    latent = fit_latent(
        frame,
        seed=0,
        policy_seed=0,
        embed_dim=PUBLISHED_EMBED_DIM,
        depth=PUBLISHED_DEPTH,
        **shared,
    )
    direct = fit_direct(
        frame,
        seed=0,
        policy_seed=1,
        hidden_dim=PUBLISHED_DIRECT_HIDDEN,
        depth=PUBLISHED_DEPTH,
        **shared,
    )
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
    name = order_up_to_study_name(config.holding_rate)
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
    import gc

    import pandas as pd

    from src.jepa.eval import (
        IN_DISTRIBUTION_HOLDING_RATE,
        finish_published_comparison,
        score_rollout_gate,
    )
    from src.jepa.plan import load_planners
    from src.tuning.config import TuningConfig

    latent, direct = load_planners(args.checkpoints)
    # The log is only an input to the rollout gate and the least-squares
    # floor. Drop it before the holdout episodes so those episodes are
    # not scored while the frame is still resident.
    frame = pd.read_parquet(args.log)
    accuracy, floor = score_rollout_gate(latent, direct, frame, args.output)
    del frame
    gc.collect()
    shape = _tuning_shape(args, ("episode_length", "k_active", "delivery_lag"))
    config = TuningConfig(**shape) if shape else None
    seed_count = {} if args.n_seeds is None else {"n_seeds": args.n_seeds}
    factories = sweep_factories(args.tuning_dir)
    anchor = (
        _policy_factory(args.anchor)
        if args.anchor is not None
        else factories[IN_DISTRIBUTION_HOLDING_RATE]
    )
    finish_published_comparison(
        latent,
        direct,
        accuracy,
        floor,
        args.output,
        anchor_factory=anchor,
        ceiling_factories=factories,
        config=config,
        **seed_count,
    )


def _parse_sizes(text: str | None) -> tuple[tuple[int, int], ...]:
    if text is None:
        return CAPACITY_GRID
    sizes: list[tuple[int, int]] = []
    for piece in text.split(","):
        width, layers = piece.split("x")
        sizes.append((int(width), int(layers)))
    return tuple(sizes)


def _episode_counts(text: str | None, frame) -> tuple[int, ...]:
    if text is not None:
        return tuple(int(piece) for piece in text.split(",") if piece)
    total = int(frame["episode_seed"].nunique())
    raw = (max(4, total // 4), max(4, total // 2), max(4, (3 * total) // 4), total)
    counts: list[int] = []
    for count in raw:
        if count <= total and count not in counts:
            counts.append(count)
    return tuple(counts)


def _scale(args: argparse.Namespace) -> None:
    import torch

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _scale_curves(args)
    finally:
        torch.set_num_threads(previous_threads)


def _scale_curves(args: argparse.Namespace) -> None:
    import pandas as pd

    from src.jepa.dynamics import (
        PLANNING_HORIZON_MULTIPLE,
        PUBLISHED_DEPTH,
        PUBLISHED_DIRECT_HIDDEN,
        PUBLISHED_EMBED_DIM,
        DirectDynamics,
        LatentDynamics,
        direct_hidden_matching,
        parameter_count,
    )
    from src.jepa.plan import (
        data_scaling_curve,
        model_capacity_curve,
        write_capacity_curve,
        write_data_curve,
    )

    frame = pd.read_parquet(args.log)
    if args.embed_dim is None and args.depth is None:
        embed_dim = PUBLISHED_EMBED_DIM
        depth = PUBLISHED_DEPTH
        hidden = PUBLISHED_DIRECT_HIDDEN
    else:
        embed_dim = PUBLISHED_EMBED_DIM if args.embed_dim is None else args.embed_dim
        depth = PUBLISHED_DEPTH if args.depth is None else args.depth
        hidden = direct_hidden_matching(embed_dim, depth)
    unroll_steps = (
        int(args.model_unroll_steps)
        if args.model_unroll_steps is not None
        else PLANNING_HORIZON_MULTIPLE * int(args.delivery_lag)
    )
    unroll = {"model_unroll_steps": unroll_steps}
    output = Path(args.output)
    meta = {
        "epochs": int(args.epochs),
        "delivery_lag": int(args.delivery_lag),
        "model_unroll_steps": unroll_steps,
        "log_rows": int(len(frame)),
        "log_episodes": int(frame["episode_seed"].nunique()),
        "seed": 0,
        "lr": float(args.lr),
    }
    capacity = model_capacity_curve(
        frame,
        _parse_sizes(args.sizes),
        delivery_lag=args.delivery_lag,
        epochs=args.epochs,
        lr=args.lr,
        artifact_dir=output / "fits",
        **unroll,
    )
    write_capacity_curve(capacity, output / "capacity.json", **meta)
    data = data_scaling_curve(
        frame,
        _episode_counts(args.episodes, frame),
        embed_dim=embed_dim,
        depth=depth,
        hidden_dim=hidden,
        delivery_lag=args.delivery_lag,
        epochs=args.epochs,
        lr=args.lr,
        artifact_dir=output / "fits",
        **unroll,
    )
    write_data_curve(
        data,
        output / "data.json",
        embed_dim=embed_dim,
        depth=depth,
        parameters=parameter_count(
            LatentDynamics(delivery_lag=args.delivery_lag, embed_dim=embed_dim, depth=depth)
        ),
        direct_hidden=hidden,
        direct_parameters=parameter_count(
            DirectDynamics(
                delivery_lag=args.delivery_lag,
                hidden_dim=hidden,
                depth=depth,
            )
        ),
        **meta,
    )


def main(argv: list[str] | None = None) -> None:
    """Run one published subcommand. ``argv`` is the argument list without the program name."""
    args = _parser().parse_args(argv)
    if args.command == "collect":
        _collect(args)
    elif args.command == "fit":
        _fit(args)
    elif args.command == "scale":
        _scale(args)
    elif args.command == "tune":
        _tune(args)
    elif args.command == "eval":
        _eval(args)
