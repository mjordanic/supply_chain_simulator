"""In-distribution and holding-rate retarget comparison.

Reports paired ``net_profit / initial_cash`` on the tuner holdout, with
order volume, service level, stockout rate, and realised fill rate on
every arm. The in-distribution rule passes when the learned arm beats
the tuned anchor and the bootstrap interval of that difference excludes
zero. The retarget rule passes when the frozen planner beats the frozen
textbook policy on that paired profit.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

from src.jepa.dynamics import (
    OracleDynamics,
    RolloutAccuracy,
    fit_least_squares_dynamics,
    held_out_rollout_accuracy,
    rollout_beats_baseline,
)
from src.jepa.plan import ReplenishmentPlanner
from src.jepa.policy import ExploratoryPolicy, NeverOrderPolicy
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
COLLINEAR_THRESHOLD = 0.99

IN_DISTRIBUTION_RULE = (
    "In-distribution rule (holding_rate 0.01, tuner holdout): pass if the "
    "learned arm beats the tuned anchor on paired net_profit / initial_cash "
    "and the bootstrap interval of latent minus that anchor excludes zero."
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
    """Learned arm beats the anchor and the interval excludes zero.

    The interval is latent minus anchor. Entirely above zero is a win.
    A tie, or a loss whose interval happens to exclude zero, fails.
    """
    low, _high = paired_interval(latent, anchor)
    return low > 0.0


def retarget_passes(latent: Sequence[float], frozen: Sequence[float]) -> bool:
    """Frozen planner beats the frozen textbook policy on paired profit."""
    diffs = [float(a) - float(b) for a, b in zip(latent, frozen, strict=True)]
    return sum(diffs) / len(diffs) > 0.0


# Reference rows stay when a learned arm is not distinct from them.
# Among learned arms, the earlier name is the one that stays in the file.
ARM_PREFERENCE: tuple[str, ...] = (
    "never_order",
    "random_cover",
    "least_squares",
    "oracle",
    "anchor",
    "order_up_to",
    "reorder_point",
    "periodic_order_up_to",
    "periodic_reorder",
    "ceiling",
    "latent",
    "direct",
    "permutation",
    "fair_share",
)

LEARNED_ARMS: tuple[str, ...] = ("latent", "direct", "permutation", "fair_share")

ROW_FIELDS: tuple[str, ...] = ("tightness", "cash_tightness", "latent_minus_anchor")
METRIC_SUFFIXES: tuple[str, ...] = (
    "_order_volume",
    "_service_level",
    "_stockout_rate",
    "_fill_rate",
)


def is_profit_column(name: str) -> bool:
    """True for an arm's paired profit, not a per-seed field or operational column."""
    if name == "seed" or name in ROW_FIELDS or name.startswith("rollout_r2_"):
        return False
    return not name.endswith(METRIC_SUFFIXES)


def pearson(left: Sequence[float], right: Sequence[float]) -> float:
    """Correlation of two per-seed series. Identical series score 1."""
    n = len(left)
    if n != len(right) or n < 2:
        return float("nan")
    mean_left = sum(left) / n
    mean_right = sum(right) / n
    var_left = sum((value - mean_left) ** 2 for value in left)
    var_right = sum((value - mean_right) ** 2 for value in right)
    if var_left <= 1e-18 or var_right <= 1e-18:
        if all(abs(a - b) <= 1e-9 for a, b in zip(left, right, strict=True)):
            return 1.0
        return float("nan")
    cov = sum(
        (a - mean_left) * (b - mean_right) for a, b in zip(left, right, strict=True)
    )
    return cov / math.sqrt(var_left * var_right)


def collinear_report(
    profits: dict[str, tuple[float, ...]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Warnings and the arms omitted from the written table.

    A pair whose per-seed profit correlates above 0.99 is not a distinct
    comparison. The later name in ``ARM_PREFERENCE`` is omitted, and only
    when the earlier name itself stays. A chain does not drop an arm that
    is still distinct from every arm the file keeps.
    """
    names = [name for name in ARM_PREFERENCE if name in profits]
    names.extend(name for name in profits if name not in names)
    warnings: list[str] = []
    dropped: set[str] = set()
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            score = pearson(profits[left], profits[right])
            if not score > COLLINEAR_THRESHOLD:
                continue
            warnings.append(
                f"{left} and {right} correlate at {score:.4f} across seeds "
                "and are not a distinct comparison"
            )
            if left not in dropped:
                dropped.add(right)
    return tuple(warnings), tuple(name for name in names if name in dropped)


def learned_beats_never_order(table: ComparisonTable) -> bool:
    """Every learned arm's mean profit is strictly above the never-order floor."""
    floor = table.per_seed["never_order"]
    floor_mean = sum(floor) / len(floor)
    for name in LEARNED_ARMS:
        series = table.per_seed[name]
        if sum(series) / len(series) <= floor_mean:
            return False
    return True


@dataclass(frozen=True)
class ComparisonTable:
    """One protocol. ``per_seed`` is paired ``net_profit / initial_cash``."""

    holding_rate: float
    seeds: tuple[int, ...]
    per_seed: dict[str, tuple[float, ...]]
    tightness: tuple[float, ...] = ()
    cash_tightness: tuple[float, ...] = ()
    order_volume: dict[str, tuple[float, ...]] = field(default_factory=dict)
    service_level: dict[str, tuple[float, ...]] = field(default_factory=dict)
    stockout_rate: dict[str, tuple[float, ...]] = field(default_factory=dict)
    fill_rate: dict[str, tuple[float, ...]] = field(default_factory=dict)
    paired_difference: tuple[float, ...] = ()
    paired_interval: tuple[float, float] = (float("nan"), float("nan"))
    collinear_warnings: tuple[str, ...] = ()
    dropped_arms: tuple[str, ...] = ()


@dataclass(frozen=True)
class SyntheticComparison:
    """In-distribution table, retarget table, and the two pass booleans."""

    in_distribution: ComparisonTable
    retarget: ComparisonTable
    in_distribution_passes: bool
    retarget_passes: bool
    in_distribution_paired_interval: tuple[float, float]
    rollout_accuracy: RolloutAccuracy | None = None


def _specs(config: TuningConfig, n_seeds: int, seed_offset: int) -> list[TuningEpisodeSpec]:
    catalog = make_synthetic_catalog(config.K_catalog)
    return [sample_episode(catalog, config, seed_offset + i) for i in range(n_seeds)]


def _column(kpis: dict[str, object], name: str) -> tuple[float, ...]:
    return tuple(float(value) for value in kpis[name])  # type: ignore[index]


def _score_arm(
    factory: Callable[[], IntermediatePolicy],
    specs: list[TuningEpisodeSpec],
) -> dict[str, tuple[float, ...]]:
    """Profit and the operational columns the evaluator already returns."""
    kpis = evaluate_policy_normalised(factory, specs)
    profit = tuple(
        float(value) / max(1e-9, float(cash))
        for value, cash in zip(
            kpis["per_seed_net_profit"],
            kpis["per_seed_initial_cash"],
            strict=True,
        )
    )
    return {
        "profit": profit,
        "order_volume": _column(kpis, "per_seed_order_volume"),
        "service_level": _column(kpis, "per_seed_service_level"),
        "stockout_rate": _column(kpis, "per_seed_stockout_rate"),
        "fill_rate": _column(kpis, "per_seed_fill_rate"),
    }


def _tightness(specs: list[TuningEpisodeSpec]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    def _num(value: float | None) -> float:
        return float("nan") if value is None else float(value)

    return (
        tuple(_num(spec.tightness) for spec in specs),
        tuple(_num(spec.cash_tightness) for spec in specs),
    )


def _planner_factory(
    planner: ReplenishmentPlanner,
    *,
    holding_rate: float,
    order_fee: float,
    ration: str | None = None,
    dynamics: object | None = None,
) -> Callable[[], ReplenishmentPlanner]:
    """Fresh buyer per seed. Dynamics stay the fitted object."""
    chosen = planner.ration if ration is None else ration
    model = planner.dynamics if dynamics is None else dynamics

    def factory() -> ReplenishmentPlanner:
        return ReplenishmentPlanner(
            model,
            policy_seed=planner.policy_seed,
            delivery_lag=planner.delivery_lag,
            budget=planner.budget,
            holding_rate=holding_rate,
            order_fee=order_fee,
            ration=chosen,
            horizon_multiple=planner.horizon_multiple,
            search=planner.search,
        )

    return factory


def _comparison_table(
    *,
    holding_rate: float,
    seeds: tuple[int, ...],
    specs: list[TuningEpisodeSpec],
    arms: dict[str, Callable[[], IntermediatePolicy]],
    alias: tuple[str, str] | None = None,
) -> ComparisonTable:
    scored = {name: _score_arm(factory, specs) for name, factory in arms.items()}
    profits = {name: values["profit"] for name, values in scored.items()}
    order_volume = {name: values["order_volume"] for name, values in scored.items()}
    service_level = {name: values["service_level"] for name, values in scored.items()}
    stockout_rate = {name: values["stockout_rate"] for name, values in scored.items()}
    fill_rate = {name: values["fill_rate"] for name, values in scored.items()}
    if alias is not None:
        dest, source = alias
        profits[dest] = profits[source]
        order_volume[dest] = order_volume[source]
        service_level[dest] = service_level[source]
        stockout_rate[dest] = stockout_rate[source]
        fill_rate[dest] = fill_rate[source]
    warnings, dropped = collinear_report(profits)
    tightness, cash_tightness = _tightness(specs)
    low, high = paired_interval(profits["latent"], profits["anchor"])
    difference = tuple(
        float(left) - float(right)
        for left, right in zip(profits["latent"], profits["anchor"], strict=True)
    )
    return ComparisonTable(
        holding_rate=holding_rate,
        seeds=seeds,
        per_seed=profits,
        tightness=tightness,
        cash_tightness=cash_tightness,
        order_volume=order_volume,
        service_level=service_level,
        stockout_rate=stockout_rate,
        fill_rate=fill_rate,
        paired_difference=difference,
        paired_interval=(low, high),
        collinear_warnings=warnings,
        dropped_arms=dropped,
    )


def compare_synthetic(
    latent: ReplenishmentPlanner,
    direct: ReplenishmentPlanner,
    *,
    anchor_factory: Callable[[], IntermediatePolicy] | None = None,
    ceiling_factory: Callable[[], IntermediatePolicy] | None = None,
    n_seeds: int = N_HOLDOUT_SEEDS,
    seed_offset: int = HOLDOUT_SEED_OFFSET,
    config: TuningConfig | None = None,
    frame: pd.DataFrame | None = None,
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

    ``frame`` is the transition log. The least-squares floor is fit on
    the latent model's training episodes from that log. Without a log the
    floor predicts zeros. The oracle ceiling does not fit.
    """
    base = config if config is not None else TuningConfig()
    seeds = tuple(seed_offset + i for i in range(n_seeds))
    in_config = replace(base, holding_rate=IN_DISTRIBUTION_HOLDING_RATE)
    retarget_config = replace(base, holding_rate=RETARGET_HOLDING_RATE)
    in_specs = _specs(in_config, n_seeds, seed_offset)
    retarget_specs = _specs(retarget_config, n_seeds, seed_offset)
    order_fee = float(base.order_fee)
    log = frame if frame is not None else pd.DataFrame()
    train_episodes = set(latent.dynamics.train_episodes) or None
    least_squares = fit_least_squares_dynamics(
        log,
        delivery_lag=latent.delivery_lag,
        train_episodes=train_episodes,
    )
    oracle = OracleDynamics(latent.delivery_lag)
    cover_lag = int(base.delivery_lag) if base.delivery_lag is not None else int(latent.delivery_lag)

    def _shared(holding_rate: float) -> dict[str, Callable[[], IntermediatePolicy]]:
        return {
            "never_order": lambda: NeverOrderPolicy(),
            "random_cover": lambda: ExploratoryPolicy(policy_seed=0, delivery_lag=cover_lag),
            "least_squares": _planner_factory(
                latent,
                holding_rate=holding_rate,
                order_fee=order_fee,
                dynamics=least_squares,
            ),
            "oracle": _planner_factory(
                latent,
                holding_rate=holding_rate,
                order_fee=order_fee,
                dynamics=oracle,
            ),
            "latent": _planner_factory(
                latent, holding_rate=holding_rate, order_fee=order_fee
            ),
            "direct": _planner_factory(
                direct, holding_rate=holding_rate, order_fee=order_fee
            ),
            "permutation": _planner_factory(
                latent,
                holding_rate=holding_rate,
                order_fee=order_fee,
                ration="permutation",
            ),
            "fair_share": _planner_factory(
                latent,
                holding_rate=holding_rate,
                order_fee=order_fee,
                ration="fair_share",
            ),
        }

    anchor = anchor_factory if anchor_factory is not None else OrderUpToPolicy
    in_arms = _shared(IN_DISTRIBUTION_HOLDING_RATE)
    for label, factory in _IN_DISTRIBUTION_TEXTBOOK.items():
        in_arms[label] = factory
    published_anchor = anchor_factory is None
    if not published_anchor:
        in_arms["anchor"] = anchor
    in_table = _comparison_table(
        holding_rate=IN_DISTRIBUTION_HOLDING_RATE,
        seeds=seeds,
        specs=in_specs,
        arms=in_arms,
        alias=("anchor", "order_up_to") if published_anchor else None,
    )
    retarget_arms = _shared(RETARGET_HOLDING_RATE)
    retarget_arms["anchor"] = anchor
    if ceiling_factory is not None:
        retarget_arms["ceiling"] = ceiling_factory
    retarget_table = _comparison_table(
        holding_rate=RETARGET_HOLDING_RATE,
        seeds=seeds,
        specs=retarget_specs,
        arms=retarget_arms,
    )
    return SyntheticComparison(
        in_distribution=in_table,
        retarget=retarget_table,
        in_distribution_passes=in_distribution_passes(
            in_table.per_seed["latent"],
            in_table.per_seed["anchor"],
        ),
        retarget_passes=retarget_passes(
            retarget_table.per_seed["latent"],
            retarget_table.per_seed["anchor"],
        ),
        in_distribution_paired_interval=in_table.paired_interval,
    )


def _mean_score(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if not math.isnan(float(value))]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


def _kept_arms(table: ComparisonTable) -> list[str]:
    dropped = set(table.dropped_arms)
    names = [name for name in ARM_PREFERENCE if name in table.per_seed and name not in dropped]
    names.extend(name for name in table.per_seed if name not in names and name not in dropped)
    return names


def _comparison_frame(
    table: ComparisonTable,
    accuracy: RolloutAccuracy | None,
) -> pd.DataFrame:
    """One row per seed. Collinear arms are omitted. Does not invent a cell."""
    arms = _kept_arms(table)
    operational = (
        ("order_volume", table.order_volume),
        ("service_level", table.service_level),
        ("stockout_rate", table.stockout_rate),
        ("fill_rate", table.fill_rate),
    )
    rows: list[dict[str, float | int]] = []
    for index, seed in enumerate(table.seeds):
        row: dict[str, float | int] = {"seed": int(seed)}
        if table.tightness:
            row["tightness"] = float(table.tightness[index])
        if table.cash_tightness:
            row["cash_tightness"] = float(table.cash_tightness[index])
        for name in arms:
            row[name] = float(table.per_seed[name][index])
            for suffix, series in operational:
                values = series.get(name)
                if values:
                    row[f"{name}_{suffix}"] = float(values[index])
        if table.paired_difference:
            row["latent_minus_anchor"] = float(table.paired_difference[index])
        if accuracy is not None:
            row["rollout_r2_latent"] = _mean_score(accuracy.latent)
            row["rollout_r2_direct"] = _mean_score(accuracy.direct)
            row["rollout_r2_ols"] = _mean_score(accuracy.ols)
        rows.append(row)
    return pd.DataFrame(rows)


def write_synthetic_comparison(result: SyntheticComparison, directory: str | Path) -> Path:
    """Write the in-distribution and retarget tables. Does not invent a cell."""
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    accuracy = result.rollout_accuracy
    _comparison_frame(result.in_distribution, accuracy).to_parquet(
        destination / "in_distribution.parquet",
        index=False,
    )
    _comparison_frame(result.retarget, accuracy).to_parquet(
        destination / "retarget.parquet",
        index=False,
    )
    warnings = {
        "in_distribution": list(result.in_distribution.collinear_warnings),
        "retarget": list(result.retarget.collinear_warnings),
    }
    (destination / "comparison_warnings.json").write_text(
        json.dumps(warnings, indent=2) + "\n",
        encoding="utf-8",
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
    return write_synthetic_comparison(replace(result, rollout_accuracy=accuracy), destination)


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
    result = compare_synthetic(latent, direct, frame=frame, **compare_kwargs)
    result = replace(result, rollout_accuracy=accuracy)
    write_synthetic_comparison(result, destination)
    return result
