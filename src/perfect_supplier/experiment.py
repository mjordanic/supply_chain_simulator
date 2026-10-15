"""Commands for the perfect-supplier experiment.

    uv run python -m src.perfect_supplier collect --n-worlds 2000
    uv run python -m src.perfect_supplier fit
    uv run python -m src.perfect_supplier tune --n-trials 200
    uv run python -m src.perfect_supplier eval --n-worlds 300
    uv run python -m src.perfect_supplier shadow --n-worlds 20

Every command takes ``--root`` (default ``runs/perfect-supplier``) and
``--variant`` (``base``, ``one_product``, ``no_fee``; a non-base variant
defaults to ``runs/perfect-supplier-<variant>``). ``eval`` and ``shadow`` take
``--model-root`` so a variant can reuse the forecasters fitted on the base
log: they learn demand only, and demand does not depend on the variant.
``collect`` writes the behaviour log (split ``train``). ``fit`` trains the
JEPA forecaster and the direct ablation on it. ``tune`` fits one parameter
set per textbook family across ``tune`` worlds. ``eval`` runs every arm
closed-loop on its own copy of each ``eval`` world. ``shadow`` drives each
``eval`` world with one arm and logs what every other arm would have ordered.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from src.perfect_supplier import data
from src.perfect_supplier.behavior import sample_behavior
from src.perfect_supplier.forecast import MovingAverageForecaster, OracleForecaster
from src.perfect_supplier.planner import Planner
from src.perfect_supplier.sim import Decision, simulate
from src.perfect_supplier.textbook import FAMILIES, EoqParams, EoqPolicy, TextbookParams, TextbookPolicy
from src.perfect_supplier.world import VARIANTS, World, demand_path, sample_world, world_seed

PART_SIZE = 50


def _workers() -> int:
    return int(os.environ.get("PS_WORKERS", os.cpu_count() or 1))


class NeverOrder:
    name = "never_order"

    def decide(self, obs) -> Decision:
        return Decision(np.zeros(obs.n_products, dtype=np.int64))


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


def tuned_params(root: Path) -> dict[str, TextbookParams | EoqParams]:
    path = root / "tuning" / "best_params.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {family: (EoqParams if family.startswith("eoq") else TextbookParams)(**p) for family, p in raw.items()}


def arm_factories(root: Path, arms: list[str], model_root: Path | None = None) -> dict[str, Callable[[World], object]]:
    """Map arm name to ``world -> policy``. Model weights load once per worker."""
    tuned = tuned_params(root)
    factories: dict[str, Callable[[World], object]] = {}
    for arm in arms:
        if arm == "never_order":
            factories[arm] = lambda w: NeverOrder()
        elif arm.startswith("tuned_") and arm.removeprefix("tuned_") not in tuned:
            raise ValueError(f"{arm}: no tuned parameters in {root / 'tuning'}; run tune first")
        elif arm.startswith("tuned_eoq"):
            factories[arm] = lambda w, f=arm.removeprefix("tuned_"), a=arm: EoqPolicy(tuned[f], name=a)
        elif arm.startswith("tuned_"):
            family = arm.removeprefix("tuned_")
            factories[arm] = lambda w, f=family, a=arm: TextbookPolicy(tuned[f], name=a)
        elif arm.startswith("default_"):
            family = arm.removeprefix("default_")
            factories[arm] = lambda w, f=family, a=arm: TextbookPolicy(TextbookParams(family=f), name=a)
        elif arm == "planner_oracle":
            factories[arm] = lambda w: Planner(OracleForecaster(w), name="planner_oracle")
        elif arm == "planner_moving_average":
            factories[arm] = lambda w: Planner(MovingAverageForecaster(), name="planner_moving_average")
        elif arm in ("jepa", "direct"):
            factories[arm] = _model_factory(model_root or root, arm)
        else:
            raise ValueError(f"unknown arm {arm!r}")
    return factories


def _model_factory(root: Path, arm: str):
    from src.perfect_supplier import jepa

    cache: dict[str, object] = {}

    def make(world: World):
        if "model" not in cache:
            import torch

            torch.set_num_threads(1)
            cache["model"] = jepa.load(root / "models" / f"{arm}.pt")
        return Planner(jepa.JepaForecaster(cache["model"], name=arm), name=arm)

    return make


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


def _run_part(root: Path, split: str, method: str, part: int, seeds: list[int], make, shadow_makers, variant: str):
    frames, summaries = [], []
    for seed in seeds:
        world = sample_world(seed, split, variant=variant)
        policy = make(world)
        shadow = {n: m(world) for n, m in (shadow_makers or {}).items()}
        result = simulate(world, policy, shadow=shadow)
        result.frame["method"] = method
        result.summary["method"] = method
        result.summary["behavior"] = policy.name
        if hasattr(policy, "describe"):
            result.summary["behavior_params"] = json.dumps(policy.describe())
        frames.append(result.frame.assign(behavior=policy.name))
        summaries.append(result.summary)
    data.write_part(root, split, method, part, frames, summaries)
    return [s["score"] for s in summaries]


def run_split(
    root: Path,
    split: str,
    method: str,
    seeds: list[int],
    make,
    shadow_makers=None,
    variant: str = "base",
    first_part: int = 0,
) -> None:
    """Run ``seeds`` and write parts ``first_part, first_part + 1, ...``.

    ``first_part == 0`` clears the slice first; a later chunk (``eval --offset``)
    adds to it.
    """
    if first_part == 0:
        data.clear(root, split, method)
    chunks = [seeds[i : i + PART_SIZE] for i in range(0, len(seeds), PART_SIZE)]
    # Small parts so every worker stays busy on slow (planner) arms.
    if len(chunks) < 4 * _workers():
        size = max(1, len(seeds) // (4 * _workers()))
        chunks = [seeds[i : i + size] for i in range(0, len(seeds), size)]
    scores = Parallel(n_jobs=_workers(), verbose=5)(
        delayed(_run_part)(root, split, method, first_part + i, chunk, make, shadow_makers, variant)
        for i, chunk in enumerate(chunks)
    )
    flat = [s for part in scores for s in part]
    print(f"{variant} {split}/{method}: {len(flat)} worlds, mean score {np.mean(flat):.4f}", flush=True)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def collect(root: Path, n_worlds: int, variant: str) -> None:
    seeds = [world_seed("train", i) for i in range(n_worlds)]
    run_split(root, "train", "behavior", seeds, lambda w: sample_behavior(w.world_seed), variant=variant)


def _bank(root: Path, seeds: list[int], variant: str):
    from src.perfect_supplier.jepa import bank_from_timeseries
    from src.perfect_supplier.world import HISTORY_DAYS

    frame = data.load_timeseries(
        root, split="train", method="behavior", world_seed=seeds,
        columns=["world_seed", "product", "day", "sales", "censored", "demand"],
    )
    history = {s: demand_path(sample_world(s, "train", variant=variant))[:HISTORY_DAYS] for s in seeds}
    return bank_from_timeseries(frame, history)


def fit(root: Path, n_worlds: int | None, steps: int, variant: str, val_fraction: float = 0.1) -> None:
    import torch

    from src.perfect_supplier import jepa

    torch.set_num_threads(_workers())
    summary = data.load_summary(root)
    seeds = sorted(summary.loc[(summary["split"] == "train"), "world_seed"].unique().tolist())
    if n_worlds:
        seeds = seeds[:n_worlds]
    rng = np.random.default_rng(0)
    val = set(rng.choice(seeds, size=max(1, int(len(seeds) * val_fraction)), replace=False).tolist())
    train_bank = _bank(root, [s for s in seeds if s not in val], variant)
    val_bank = _bank(root, sorted(val), variant)
    (root / "models").mkdir(parents=True, exist_ok=True)
    (root / "models" / "val_worlds.json").write_text(json.dumps(sorted(int(s) for s in val)))
    for arm, use_jepa in (("jepa", True), ("direct", False)):
        print(f"fitting {arm}", flush=True)
        config = jepa.JepaConfig(jepa=use_jepa, steps=steps)
        model = jepa.fit(train_bank, val_bank, config, log_path=root / "models" / f"{arm}_fit.json")
        jepa.save(model, root / "models" / f"{arm}.pt")


def _tune_objective(policy, seeds: list[int], variant: str) -> float:
    return float(np.mean([simulate(sample_world(s, "tune", variant=variant), policy).summary["score"] for s in seeds]))


TUNE_FAMILIES = (*FAMILIES, "eoq", "eoq_joint")


def tune(root: Path, n_trials: int, n_worlds: int, variant: str, families: list[str]) -> None:
    """Tune each family once across ``n_worlds`` tune worlds; merge into ``best_params.json``."""
    import optuna

    seeds = [world_seed("tune", i) for i in range(n_worlds)]
    out = root / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    best_path = out / "best_params.json"
    scores_path = out / "best_scores.json"
    best: dict[str, dict] = json.loads(best_path.read_text()) if best_path.exists() else {}
    scores: dict[str, dict] = json.loads(scores_path.read_text()) if scores_path.exists() else {}
    for family in families:

        def objective(trial, family=family):
            if family.startswith("eoq"):
                params = EoqParams(
                    service_z=trial.suggest_float("service_z", 0.0, 6.0),
                    demand_window=trial.suggest_int("demand_window", 7, 56),
                    q_scale=trial.suggest_float("q_scale", 0.25, 4.0, log=True),
                    joint=family == "eoq_joint",
                    family=family,
                )
                return _tune_objective(EoqPolicy(params), seeds, variant)
            params = TextbookParams(
                family=family,
                cover_horizon_ticks=trial.suggest_int("cover_horizon_ticks", 1, 28),
                safety_lead_pct_of_lag=trial.suggest_float("safety_lead_pct_of_lag", 0.0, 2.0),
                stockout_safety_bonus_pct_of_lag=trial.suggest_float("stockout_safety_bonus_pct_of_lag", 0.0, 1.5),
                review_interval=(
                    trial.suggest_int("review_interval", 1, 14) if family.startswith("periodic") else None
                ),
                demand_window=trial.suggest_int("demand_window", 2, 42),
            )
            return _tune_objective(TextbookPolicy(params), seeds, variant)

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials, n_jobs=_workers())
        params = dict(study.best_params)
        if family.startswith("eoq"):
            params["joint"] = family == "eoq_joint"
        else:
            params.setdefault("review_interval", None)
        best[family] = {"family": family, **params}
        scores[family] = {"tune_score": study.best_value, "n_worlds": n_worlds, "n_trials": n_trials, "variant": variant}
        study.trials_dataframe().to_parquet(out / f"{family}_trials.parquet")
        print(f"{family}: best {study.best_value:.4f} {params}", flush=True)
        best_path.write_text(json.dumps(best, indent=1))
        scores_path.write_text(json.dumps(scores, indent=1))


DEFAULT_EVAL_ARMS = [
    "jepa",
    "direct",
    "planner_moving_average",
    "planner_oracle",
    "tuned_order_up_to",
    "tuned_reorder_point",
    "tuned_periodic_order_up_to",
    "tuned_periodic_reorder",
    "tuned_eoq",
    "tuned_eoq_joint",
    "default_order_up_to",
    "never_order",
]


def evaluate(root: Path, n_worlds: int, arms: list[str], variant: str, model_root: Path | None, offset: int = 0) -> None:
    seeds = [world_seed("eval", offset + i) for i in range(n_worlds)]
    for arm, make in arm_factories(root, arms, model_root).items():
        # A run writes at most one part per world, so numbering parts from the offset cannot collide.
        run_split(root, "eval", arm, seeds, make, variant=variant, first_part=offset)


def shadow(root: Path, n_worlds: int, driver: str, arms: list[str], variant: str, model_root: Path | None) -> None:
    seeds = [world_seed("eval", 100_000 + i) for i in range(n_worlds)]
    makers = arm_factories(root, [driver, *[a for a in arms if a != driver]], model_root)
    drive = makers.pop(driver)
    run_split(root, "shadow", driver, seeds, drive, shadow_makers=makers, variant=variant)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.perfect_supplier")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--variant", choices=VARIANTS, default="base")
    parser.add_argument("--model-root", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--n-worlds", type=int, default=2000)
    f = sub.add_parser("fit")
    f.add_argument("--n-worlds", type=int, default=None)
    f.add_argument("--steps", type=int, default=20_000)
    t = sub.add_parser("tune")
    t.add_argument("--n-trials", type=int, default=200)
    t.add_argument("--n-worlds", type=int, default=64)
    t.add_argument("--families", nargs="+", choices=TUNE_FAMILIES, default=list(TUNE_FAMILIES))
    e = sub.add_parser("eval")
    e.add_argument("--n-worlds", type=int, default=300)
    e.add_argument("--offset", type=int, default=0)
    e.add_argument("--arms", nargs="+", default=DEFAULT_EVAL_ARMS)
    s = sub.add_parser("shadow")
    s.add_argument("--n-worlds", type=int, default=20)
    s.add_argument("--driver", default="jepa")
    s.add_argument(
        "--arms", nargs="+",
        default=["jepa", "planner_oracle", "tuned_periodic_order_up_to", "tuned_eoq", "tuned_eoq_joint"],
    )
    args = parser.parse_args(argv)
    root = args.root or data.variant_root(args.variant)
    if args.command == "collect":
        collect(root, args.n_worlds, args.variant)
    elif args.command == "fit":
        fit(root, args.n_worlds, args.steps, args.variant)
    elif args.command == "tune":
        tune(root, args.n_trials, args.n_worlds, args.variant, args.families)
    elif args.command == "eval":
        evaluate(root, args.n_worlds, args.arms, args.variant, args.model_root, args.offset)
    elif args.command == "shadow":
        shadow(root, args.n_worlds, args.driver, args.arms, args.variant, args.model_root)
