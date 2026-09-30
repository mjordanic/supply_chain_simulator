"""In-distribution and holding-rate retarget comparison.

Reports paired ``net_profit / initial_cash`` on the tuner holdout. The
in-distribution rule passes when the paired interval against the anchor
covers zero, or the latent mean is at least 95% of the anchor mean. The
retarget rule passes when the frozen planner beats the frozen textbook
policy on that paired profit.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from src.jepa.dynamics import RolloutAccuracy, held_out_rollout_accuracy, rollout_beats_baseline
from src.jepa.plan import ReplenishmentPlanner
from src.sim.policy import (
    IntermediatePolicy,
    OrderUpToPolicy,
    PeriodicOrderUpToPolicy,
    PeriodicReorderPolicy,
    ReorderPointPolicy,
)
from src.tuning.config import TuningConfig
from src.tuning.episode import TuningEpisodeSpec, make_synthetic_catalog, sample_episode
from src.tuning.evaluator import evaluate_policy_normalised

_TUNER = TuningConfig()
N_HOLDOUT_SEEDS = _TUNER.n_holdout_seeds
HOLDOUT_SEED_OFFSET = _TUNER.holdout_seed_offset
IN_DISTRIBUTION_HOLDING_RATE = 0.01
RETARGET_HOLDING_RATE = 0.05
MEAN_RATIO = 0.95

IN_DISTRIBUTION_RULE = (
    "In-distribution rule (holding_rate 0.01, tuner holdout): pass if the "
    "paired interval of latent minus the anchor covers zero, or the latent "
    "mean is at least 95% of the anchor mean."
)
RETARGET_RULE = (
    "Retarget rule (holding_rate 0.05, same weights, no new log): pass if "
    "the frozen planner beats the frozen OrderUpToPolicy from the 0.01 study "
    "on paired net_profit / initial_cash. The OrderUpToPolicy tuned at 0.05 "
    "is a ceiling and is not a pass bar."
)

_IN_DISTRIBUTION_TEXTBOOK: dict[str, Callable[[], IntermediatePolicy]] = {
    "order_up_to": OrderUpToPolicy,
    "reorder_point": ReorderPointPolicy,
    "periodic_order_up_to": PeriodicOrderUpToPolicy,
    "periodic_reorder": PeriodicReorderPolicy,
}


def paired_interval(
    left: Sequence[float],
    right: Sequence[float],
    *,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of ``left - right``.

    Same index arithmetic as the tuner holdout summary, so a published
    table and a smoke table describe one interval.
    """
    diffs = [float(a) - float(b) for a, b in zip(left, right, strict=True)]
    rng = random.Random(seed)
    n = len(diffs)
    if n == 0:
        return (float("nan"), float("nan"))
    boot_means: list[float] = []
    for _ in range(n_resamples):
        sample = [rng.choice(diffs) for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    alpha = 1.0 - ci
    lo_idx = int(alpha / 2 * n_resamples)
    hi_idx = min(int((1.0 - alpha / 2) * n_resamples) - 1, n_resamples - 1)
    return (boot_means[lo_idx], boot_means[hi_idx])


def in_distribution_passes(latent: Sequence[float], anchor: Sequence[float]) -> bool:
    """Interval covers zero, or the latent mean is at least 95% of the anchor."""
    low, high = paired_interval(latent, anchor)
    if low <= 0.0 <= high:
        return True
    latent_mean = sum(latent) / len(latent)
    anchor_mean = sum(anchor) / len(anchor)
    return latent_mean >= MEAN_RATIO * anchor_mean


def retarget_passes(latent: Sequence[float], frozen: Sequence[float]) -> bool:
    """Frozen planner beats the frozen textbook policy on paired profit."""
    diffs = [float(a) - float(b) for a, b in zip(latent, frozen, strict=True)]
    return sum(diffs) / len(diffs) > 0.0


@dataclass(frozen=True)
class ComparisonTable:
    """Paired ``net_profit / initial_cash`` for one protocol, aligned by seed."""

    holding_rate: float
    seeds: tuple[int, ...]
    per_seed: dict[str, tuple[float, ...]]


@dataclass(frozen=True)
class SyntheticComparison:
    """In-distribution table, retarget table, and the two pass booleans."""

    in_distribution: ComparisonTable
    retarget: ComparisonTable
    in_distribution_passes: bool
    retarget_passes: bool
    in_distribution_paired_interval: tuple[float, float]


def _specs(config: TuningConfig, n_seeds: int, seed_offset: int) -> list[TuningEpisodeSpec]:
    catalog = make_synthetic_catalog(config.K_catalog)
    return [sample_episode(catalog, config, seed_offset + i) for i in range(n_seeds)]


def _normalised(factory: Callable[[], IntermediatePolicy], specs: list[TuningEpisodeSpec]) -> tuple[float, ...]:
    kpis = evaluate_policy_normalised(factory, specs)
    return tuple(
        float(profit) / max(1e-9, float(cash))
        for profit, cash in zip(
            kpis["per_seed_net_profit"],
            kpis["per_seed_initial_cash"],
            strict=True,
        )
    )


def _planner_factory(
    planner: ReplenishmentPlanner,
    *,
    holding_rate: float,
    order_fee: float,
    ration: str | None = None,
) -> Callable[[], ReplenishmentPlanner]:
    """Fresh buyer per seed. Dynamics stay the fitted object."""
    chosen = planner.ration if ration is None else ration

    def factory() -> ReplenishmentPlanner:
        return ReplenishmentPlanner(
            planner.dynamics,
            policy_seed=planner.policy_seed,
            delivery_lag=planner.delivery_lag,
            budget=planner.budget,
            holding_rate=holding_rate,
            order_fee=order_fee,
            ration=chosen,
        )

    return factory


def compare_synthetic(
    latent: ReplenishmentPlanner,
    direct: ReplenishmentPlanner,
    *,
    anchor_factory: Callable[[], IntermediatePolicy] | None = None,
    ceiling_factory: Callable[[], IntermediatePolicy] | None = None,
    n_seeds: int = N_HOLDOUT_SEEDS,
    seed_offset: int = HOLDOUT_SEED_OFFSET,
    config: TuningConfig | None = None,
) -> SyntheticComparison:
    """Score both protocols on the tuner holdout.

    ``n_seeds`` and ``seed_offset`` default to the tuner holdout (32 seeds
    from ``13_000_000``). ``config`` supplies the episode shape; smoke
    tests pass a short one. Holding rate is forced to 0.01 on the
    in-distribution specs and 0.05 on the retarget specs. ``anchor_factory``
    is the ``OrderUpToPolicy`` scored at 0.01 and again, unchanged, at
    0.05. When it is omitted the anchor is published-default
    ``OrderUpToPolicy``. ``ceiling_factory`` is scored only on the retarget
    and is not a pass bar. Each seed builds a new planner on the fitted
    dynamics, so the weights are not refit. This does not run a tuning study.
    """
    base = config if config is not None else TuningConfig()
    seeds = tuple(seed_offset + i for i in range(n_seeds))
    in_config = replace(base, holding_rate=IN_DISTRIBUTION_HOLDING_RATE)
    retarget_config = replace(base, holding_rate=RETARGET_HOLDING_RATE)
    in_specs = _specs(in_config, n_seeds, seed_offset)
    retarget_specs = _specs(retarget_config, n_seeds, seed_offset)
    order_fee = float(base.order_fee)

    in_dist: dict[str, tuple[float, ...]] = {
        "latent": _normalised(
            _planner_factory(latent, holding_rate=IN_DISTRIBUTION_HOLDING_RATE, order_fee=order_fee),
            in_specs,
        ),
        "direct": _normalised(
            _planner_factory(direct, holding_rate=IN_DISTRIBUTION_HOLDING_RATE, order_fee=order_fee),
            in_specs,
        ),
        "fair_share": _normalised(
            _planner_factory(
                latent,
                holding_rate=IN_DISTRIBUTION_HOLDING_RATE,
                order_fee=order_fee,
                ration="fair_share",
            ),
            in_specs,
        ),
    }
    for label, factory in _IN_DISTRIBUTION_TEXTBOOK.items():
        in_dist[label] = _normalised(factory, in_specs)
    anchor = anchor_factory if anchor_factory is not None else OrderUpToPolicy
    in_dist["anchor"] = (
        in_dist["order_up_to"]
        if anchor_factory is None
        else _normalised(anchor, in_specs)
    )

    retarget = {
        "latent": _normalised(
            _planner_factory(latent, holding_rate=RETARGET_HOLDING_RATE, order_fee=order_fee),
            retarget_specs,
        ),
        "anchor": _normalised(anchor, retarget_specs),
    }
    if ceiling_factory is not None:
        retarget["ceiling"] = _normalised(ceiling_factory, retarget_specs)
    low, high = paired_interval(in_dist["latent"], in_dist["anchor"])
    return SyntheticComparison(
        in_distribution=ComparisonTable(
            holding_rate=IN_DISTRIBUTION_HOLDING_RATE,
            seeds=seeds,
            per_seed=in_dist,
        ),
        retarget=ComparisonTable(
            holding_rate=RETARGET_HOLDING_RATE,
            seeds=seeds,
            per_seed=retarget,
        ),
        in_distribution_passes=in_distribution_passes(in_dist["latent"], in_dist["anchor"]),
        retarget_passes=retarget_passes(retarget["latent"], retarget["anchor"]),
        in_distribution_paired_interval=(low, high),
    )


def _comparison_frame(table: ComparisonTable) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for index, seed in enumerate(table.seeds):
        row: dict[str, float | int] = {"seed": int(seed)}
        for arm, values in table.per_seed.items():
            row[arm] = float(values[index])
        rows.append(row)
    return pd.DataFrame(rows)


def write_synthetic_comparison(result: SyntheticComparison, directory: str | Path) -> Path:
    """Write the in-distribution and retarget tables. Does not invent a cell."""
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    _comparison_frame(result.in_distribution).to_parquet(
        destination / "in_distribution.parquet",
        index=False,
    )
    _comparison_frame(result.retarget).to_parquet(
        destination / "retarget.parquet",
        index=False,
    )
    return destination


class RolloutGateError(RuntimeError):
    """Held-out rollout accuracy did not beat ordinary least squares."""

    def __init__(self, accuracy: RolloutAccuracy) -> None:
        self.accuracy = accuracy
        super().__init__(
            "held-out rollout accuracy did not beat the ordinary-least-squares baseline"
        )


def _json_number(value: float) -> float | None:
    if math.isnan(value) or math.isinf(value):
        return None
    return float(value)


def _write_rollout_accuracy(accuracy: RolloutAccuracy, path: Path) -> None:
    payload = {
        "steps": list(accuracy.steps),
        "latent": [_json_number(value) for value in accuracy.latent],
        "direct": [_json_number(value) for value in accuracy.direct],
        "ols": [_json_number(value) for value in accuracy.ols],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_gated_profit_table(
    result: SyntheticComparison,
    directory: str | Path,
    accuracy: RolloutAccuracy,
) -> Path:
    """Write the profit tables when rollout accuracy beats least squares.

    The per-step curve is written either way. The profit parquets are not.
    """
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    _write_rollout_accuracy(accuracy, destination / "rollout_accuracy.json")
    if not rollout_beats_baseline(accuracy):
        raise RolloutGateError(accuracy)
    return write_synthetic_comparison(result, destination)


def publish_synthetic_comparison(
    latent: ReplenishmentPlanner,
    direct: ReplenishmentPlanner,
    frame: pd.DataFrame,
    directory: str | Path,
    **compare_kwargs: object,
) -> SyntheticComparison:
    """Score rollout accuracy, then the profit table if the gate opens."""
    accuracy = held_out_rollout_accuracy(latent.dynamics, direct.dynamics, frame)
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    _write_rollout_accuracy(accuracy, destination / "rollout_accuracy.json")
    if not rollout_beats_baseline(accuracy):
        raise RolloutGateError(accuracy)
    result = compare_synthetic(latent, direct, **compare_kwargs)
    write_synthetic_comparison(result, destination)
    return result
