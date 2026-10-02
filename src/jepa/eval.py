"""In-distribution, unseen-SKU, unseen-disruption, and holding-rate sweep.

Reports paired ``net_profit / initial_cash`` for each protocol, with
order volume, service level, stockout rate, and realised fill rate on
every arm. The three generalisation protocols are separate tables.
The in-distribution rule passes when the learned arm beats
the tuned anchor and the bootstrap interval of that difference excludes
zero. The holding-rate sweep scores the frozen planner against the
textbook policy tuned at each rate, and refuses the curve when that
planner's order sequence does not change.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import random
import sys
from collections.abc import Callable, Mapping, Sequence
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
from src.sim.distributions import Constant
from src.sim.policy import (
    IntermediatePolicy,
    OrderUpToPolicy,
    PeriodicOrderUpToPolicy,
    PeriodicReorderPolicy,
    ReorderPointPolicy,
)
from src.sim.scenario import DisruptionParams, Ware
from src.tuning.config import TuningConfig
from src.jepa.commands import SWEEP_HOLDING_RATES
from src.tuning.episode import TuningEpisodeSpec, make_synthetic_catalog, partition_catalog, sample_episode
from src.tuning.evaluator import evaluate_policy_normalised
from src.tuning.rollout import run_policy_episode
from src.tuning.rollout import run_policy_episode

_TUNER = TuningConfig()
N_HOLDOUT_SEEDS = _TUNER.n_holdout_seeds
HOLDOUT_SEED_OFFSET = _TUNER.holdout_seed_offset
UNSEEN_SKU_SEED_OFFSET = 15_000_000
UNSEEN_DISRUPTION_SEED_OFFSET = 16_000_000
IN_DISTRIBUTION_HOLDING_RATE = 0.01
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
HOLDING_RATE_SWEEP_RULE = (
    "Holding-rate sweep (0.005, 0.01, 0.02, 0.05, 0.1; same weights, no new "
    "log): at each rate the frozen planner is scored against the "
    "OrderUpToPolicy tuned at that rate. The run reports no curve when the "
    "frozen planner's order sequence is identical across those rates on the "
    "tuner holdout."
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


class UnfalsifiableTransfer(RuntimeError):
    """The frozen planner placed the same orders at every holding rate.

    The sweep then has no profit number to report. The world's holding
    charge on an unchanged order sequence is not evidence of transfer.
    """


def assert_order_sequences_differ(sequences: Mapping[float, Sequence[object]]) -> None:
    """Refuse the transfer curve when every rate placed the same orders.

    One differing rate is enough. The check compares the order sequence,
    not the profit the world charges for it.
    """
    if len(sequences) < 2:
        raise UnfalsifiableTransfer(
            "holding-rate sweep needs two rates before it can report a curve"
        )
    ordered = [sequences[rate] for rate in sequences]
    if all(seq == ordered[0] for seq in ordered[1:]):
        raise UnfalsifiableTransfer(
            "frozen planner order sequence is identical across holding rates; "
            "refusing to report the transfer curve"
        )


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
    if (
        name in {"seed", "holding_rate"}
        or name in ROW_FIELDS
        or name.startswith("rollout_r2_")
        or name.startswith("latent_minus_")
    ):
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
    baseline: str = "anchor"


@dataclass(frozen=True)
class SyntheticComparison:
    """In-distribution, unseen-SKU, unseen-disruption, and the holding-rate sweep."""

    in_distribution: ComparisonTable
    holding_rate_sweep: tuple[ComparisonTable, ...]
    in_distribution_passes: bool
    in_distribution_paired_interval: tuple[float, float]
    unseen_sku: ComparisonTable
    unseen_disruption: ComparisonTable
    rollout_accuracy: RolloutAccuracy | None = None
    holding_rate_sweep_refusal: str | None = None


def harsh_disruption_params() -> DisruptionParams:
    """Disruption regime held out of every training episode.

    Harsher than ``default_disruption_params`` on frequency, severity, and
    duration, and it adds pandemic and political unrest. The log and the
    in-distribution and unseen-SKU protocols keep the default regime.
    """
    return DisruptionParams(
        event_prob=0.2,
        types=[
            "natural_disaster",
            "economic_crisis",
            "pandemic",
            "political_unrest",
        ],
        regions=["US"],
        severity=Constant(0.2),
        duration=Constant(21),
    )


def _specs(
    config: TuningConfig,
    n_seeds: int,
    seed_offset: int,
    catalog: list[Ware],
    disruption_params: DisruptionParams | None = None,
) -> list[TuningEpisodeSpec]:
    return [
        sample_episode(
            catalog,
            config,
            seed_offset + i,
            disruption_params=disruption_params,
        )
        for i in range(n_seeds)
    ]


def protocol_episodes(
    config: TuningConfig,
    n_seeds: int,
    *,
    seed_offset: int | None = None,
) -> dict[str, list[TuningEpisodeSpec]]:
    """Episodes for the three generalisation protocols.

    In-distribution and unseen-disruption draw the training partition.
    Unseen-SKU draws the evaluation partition. Only unseen-disruption
    uses ``harsh_disruption_params``. Holding rate is the in-distribution
    rate. ``seed_offset`` moves the in-distribution block; the other two
    protocols stay on their published offsets.
    """
    offset = HOLDOUT_SEED_OFFSET if seed_offset is None else int(seed_offset)
    shaped = replace(config, holding_rate=IN_DISTRIBUTION_HOLDING_RATE)
    full = make_synthetic_catalog(config.K_catalog)
    training, evaluation = partition_catalog(full)
    return {
        "in_distribution": _specs(shaped, n_seeds, offset, training),
        "unseen_sku": _specs(shaped, n_seeds, UNSEEN_SKU_SEED_OFFSET, evaluation),
        "unseen_disruption": _specs(
            shaped,
            n_seeds,
            UNSEEN_DISRUPTION_SEED_OFFSET,
            training,
            harsh_disruption_params(),
        ),
    }


def published_protocol_ranges() -> dict[str, dict[str, int | str]]:
    """Published seed block of each generalisation protocol.

    The training block is not included: its length is the log that was
    collected. These three blocks are disjoint from each other and from
    the published log, which starts at ``14_000_000``.
    """
    rows = (
        ("in_distribution", HOLDOUT_SEED_OFFSET, "training", "default"),
        ("unseen_sku", UNSEEN_SKU_SEED_OFFSET, "evaluation", "default"),
        ("unseen_disruption", UNSEEN_DISRUPTION_SEED_OFFSET, "training", "harsh"),
    )
    return {
        name: {
            "seed_min": offset,
            "seed_max": offset + N_HOLDOUT_SEEDS - 1,
            "n_seeds": N_HOLDOUT_SEEDS,
            "catalog": catalog,
            "disruption": disruption,
        }
        for name, offset, catalog, disruption in rows
    }


def _column(kpis: dict[str, object], name: str) -> tuple[float, ...]:
    return tuple(float(value) for value in kpis[name])  # type: ignore[index]


def _eval_workers() -> int:
    raw = os.environ.get("JEPA_EVAL_WORKERS", "1")
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


_FACTORY: Callable[[], IntermediatePolicy] | None = None
_INCLUDE_ORDERS = False


def _episode_from_spec(spec: TuningEpisodeSpec) -> dict:
    if _FACTORY is None:
        raise RuntimeError("episode worker has no policy factory")
    return run_policy_episode(_FACTORY(), spec, include_orders=_INCLUDE_ORDERS)


def _map_episodes(
    factory: Callable[[], IntermediatePolicy],
    specs: list[TuningEpisodeSpec],
    *,
    include_orders: bool = False,
) -> list[dict]:
    """One fresh policy per seed. Workers above 1 fork; seed order is kept."""
    global _FACTORY, _INCLUDE_ORDERS
    workers = _eval_workers()
    if workers <= 1 or len(specs) <= 1:
        return [
            run_policy_episode(factory(), spec, include_orders=include_orders)
            for spec in specs
        ]
    _FACTORY = factory
    _INCLUDE_ORDERS = include_orders
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(min(workers, len(specs))) as pool:
        return pool.map(_episode_from_spec, specs)


def _score_from_metrics(
    metrics: list[dict],
    specs: list[TuningEpisodeSpec],
) -> dict[str, tuple[float, ...]]:
    profit = tuple(
        float(row["net_profit"]) / max(1e-9, float(spec.balance))
        for row, spec in zip(metrics, specs, strict=True)
    )
    return {
        "profit": profit,
        "order_volume": tuple(float(row["order_volume"]) for row in metrics),
        "service_level": tuple(float(row["service_level"]) for row in metrics),
        "stockout_rate": tuple(float(row["stockout_rate"]) for row in metrics),
        "fill_rate": tuple(float(row["fill_rate"]) for row in metrics),
    }


def _score_arm(
    factory: Callable[[], IntermediatePolicy],
    specs: list[TuningEpisodeSpec],
) -> dict[str, tuple[float, ...]]:
    """Profit and the operational columns the evaluator already returns."""
    if _eval_workers() > 1 and len(specs) > 1:
        return _score_from_metrics(_map_episodes(factory, specs), specs)
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


def _table_from_scored(
    *,
    holding_rate: float,
    seeds: tuple[int, ...],
    specs: list[TuningEpisodeSpec],
    scored: dict[str, dict[str, tuple[float, ...]]],
    baseline: str,
) -> ComparisonTable:
    profits = {name: values["profit"] for name, values in scored.items()}
    warnings, dropped = collinear_report(profits)
    tightness, cash_tightness = _tightness(specs)
    low, high = paired_interval(profits["latent"], profits[baseline])
    difference = tuple(
        float(left) - float(right)
        for left, right in zip(profits["latent"], profits[baseline], strict=True)
    )
    return ComparisonTable(
        holding_rate=holding_rate,
        seeds=seeds,
        per_seed=profits,
        tightness=tightness,
        cash_tightness=cash_tightness,
        order_volume={name: values["order_volume"] for name, values in scored.items()},
        service_level={name: values["service_level"] for name, values in scored.items()},
        stockout_rate={name: values["stockout_rate"] for name, values in scored.items()},
        fill_rate={name: values["fill_rate"] for name, values in scored.items()},
        paired_difference=difference,
        paired_interval=(low, high),
        collinear_warnings=warnings,
        dropped_arms=dropped,
        baseline=baseline,
    )


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _comparison_table(
    *,
    holding_rate: float,
    seeds: tuple[int, ...],
    specs: list[TuningEpisodeSpec],
    arms: dict[str, Callable[[], IntermediatePolicy]],
    alias: tuple[str, str] | None = None,
    baseline: str = "anchor",
) -> ComparisonTable:
    scored: dict[str, dict[str, tuple[float, ...]]] = {}
    for name, factory in arms.items():
        _progress(f"jepa eval arm {name} seeds {len(specs)} holding {holding_rate:g}")
        scored[name] = _score_arm(factory, specs)
    if alias is not None:
        dest, source = alias
        scored[dest] = scored[source]
    return _table_from_scored(
        holding_rate=holding_rate,
        seeds=seeds,
        specs=specs,
        scored=scored,
        baseline=baseline,
    )


def _sweep_must_check_orders(config: TuningConfig, assert_orders: bool | None) -> bool:
    """The published episode distribution is the real holdout.

    A caller-supplied cash or capacity pin, or a shortened episode, is
    not that holdout. ``publish_synthetic_comparison`` forces the check.
    """
    if assert_orders is not None:
        return assert_orders
    published = TuningConfig()
    return (
        config.episode_length == published.episode_length
        and config.K_active is None
        and config.K_catalog == published.K_catalog
        and config.capacity_dist is None
        and config.balance_dist is None
        and config.delivery_lag is None
        and config.order_fee == published.order_fee
    )


def _roll_frozen_planner(
    planner: ReplenishmentPlanner,
    holding_rate: float,
    specs: list[TuningEpisodeSpec],
    order_fee: float,
) -> tuple[dict[str, tuple[float, ...]], tuple[tuple[tuple[tuple[str, int], ...], ...], ...]]:
    """One fresh buyer per seed. Returns the arm score and each seed's orders."""
    profits: list[float] = []
    volumes: list[float] = []
    service: list[float] = []
    stockout: list[float] = []
    fill: list[float] = []
    sequences: list[tuple[tuple[tuple[str, int], ...], ...]] = []

    def factory() -> ReplenishmentPlanner:
        return planner.retarget(holding_rate, order_fee=order_fee)

    for spec, metrics in zip(
        specs,
        _map_episodes(factory, specs, include_orders=True),
        strict=True,
    ):
        profits.append(float(metrics["net_profit"]) / max(1e-9, float(spec.balance)))
        volumes.append(float(metrics["order_volume"]))
        service.append(float(metrics["service_level"]))
        stockout.append(float(metrics["stockout_rate"]))
        fill.append(float(metrics["fill_rate"]))
        sequences.append(metrics["order_sequence"])  # type: ignore[arg-type]
    scored = {
        "profit": tuple(profits),
        "order_volume": tuple(volumes),
        "service_level": tuple(service),
        "stockout_rate": tuple(stockout),
        "fill_rate": tuple(fill),
    }
    return scored, tuple(sequences)


def holding_rate_sweep(
    planner: ReplenishmentPlanner,
    ceiling_factories: Mapping[float, Callable[[], IntermediatePolicy]] | None = None,
    *,
    n_seeds: int = N_HOLDOUT_SEEDS,
    seed_offset: int = HOLDOUT_SEED_OFFSET,
    config: TuningConfig | None = None,
    assert_orders: bool = True,
) -> tuple[ComparisonTable, ...]:
    """Score the frozen planner against the textbook policy tuned at each rate.

    Episodes are the tuner holdout on ``config`` (the published episode
    distribution when ``config`` is omitted). The planner is retargeted.
    Its weights are not refit and no transition log is collected. When
    ``assert_orders`` is set, identical order sequences raise
    ``UnfalsifiableTransfer`` before any table is returned.

    ``ceiling_factories`` maps each sweep rate to the ``OrderUpToPolicy``
    tuned at that rate. Omitted factories use the published default.
    A partial map is an error. This does not run a tuning study.
    """
    if ceiling_factories is not None:
        missing = [rate for rate in SWEEP_HOLDING_RATES if rate not in ceiling_factories]
        if missing:
            raise KeyError(f"textbook factory missing for holding rates {missing}")
    base = config if config is not None else TuningConfig()
    training, _evaluation = partition_catalog(make_synthetic_catalog(base.K_catalog))
    order_fee = float(base.order_fee)
    specs_by_rate = {
        rate: _specs(replace(base, holding_rate=rate), n_seeds, seed_offset, training)
        for rate in SWEEP_HOLDING_RATES
    }
    latent_scored: dict[float, dict[str, tuple[float, ...]]] = {}
    sequences: dict[float, tuple] = {}
    for rate, specs in specs_by_rate.items():
        _progress(f"jepa eval sweep latent holding {rate:g} seeds {len(specs)}")
        scored, sequence = _roll_frozen_planner(planner, rate, specs, order_fee)
        latent_scored[rate] = scored
        sequences[rate] = sequence
    if assert_orders:
        assert_order_sequences_differ(sequences)
    tables: list[ComparisonTable] = []
    for rate, specs in specs_by_rate.items():
        factory = OrderUpToPolicy if ceiling_factories is None else ceiling_factories[rate]
        _progress(f"jepa eval sweep ceiling holding {rate:g} seeds {len(specs)}")
        tables.append(
            _table_from_scored(
                holding_rate=rate,
                seeds=tuple(seed_offset + index for index in range(n_seeds)),
                specs=specs,
                scored={
                    "latent": latent_scored[rate],
                    "ceiling": _score_arm(factory, specs),
                },
                baseline="ceiling",
            )
        )
    return tuple(tables)


def compare_synthetic(
    latent: ReplenishmentPlanner,
    direct: ReplenishmentPlanner,
    *,
    anchor_factory: Callable[[], IntermediatePolicy] | None = None,
    ceiling_factories: Mapping[float, Callable[[], IntermediatePolicy]] | None = None,
    n_seeds: int = N_HOLDOUT_SEEDS,
    seed_offset: int = HOLDOUT_SEED_OFFSET,
    config: TuningConfig | None = None,
    frame: pd.DataFrame | None = None,
    assert_orders: bool | None = None,
    least_squares: object | None = None,
) -> SyntheticComparison:
    """Score the generalisation protocols and the holding-rate sweep.

    In-distribution uses the training SKUs and, by default, 32 seeds from
    ``13_000_000``. Unseen-SKU uses the evaluation partition and seeds from
    ``15_000_000``. Unseen-disruption uses the training SKUs, seeds from
    ``16_000_000``, and ``harsh_disruption_params``. Those three tables are
    separate. ``n_seeds`` shortens every protocol. ``seed_offset`` moves
    only the in-distribution block and the sweep, which rescores that same
    block at each rate in ``SWEEP_HOLDING_RATES``. ``config`` supplies the
    episode shape; smoke tests pass a short one. ``anchor_factory`` is the
    ``OrderUpToPolicy`` scored at 0.01 on the three generalisation tables.
    When it is omitted the anchor is published-default ``OrderUpToPolicy``.
    ``ceiling_factories`` maps each sweep rate to the textbook policy tuned
    at that rate. When it is omitted every rate uses the published default.
    Each seed builds a new planner on the fitted dynamics, so the weights
    are not refit and no new log is collected. This does not run a tuning
    study.

    On the published episode distribution the sweep refuses to return when
    the frozen planner's order sequence is identical across rates.
    ``assert_orders=False`` skips that check for a shortened smoke world.
    ``publish_synthetic_comparison`` always checks.

    ``frame`` is the transition log. The least-squares floor is fit on
    the latent model's training episodes from that log. Without a log the
    floor predicts zeros. The oracle ceiling does not fit.
    """
    base = config if config is not None else TuningConfig()
    episodes = protocol_episodes(base, n_seeds, seed_offset=seed_offset)
    seeds = {
        "in_distribution": tuple(seed_offset + i for i in range(n_seeds)),
        "unseen_sku": tuple(UNSEEN_SKU_SEED_OFFSET + i for i in range(n_seeds)),
        "unseen_disruption": tuple(
            UNSEEN_DISRUPTION_SEED_OFFSET + i for i in range(n_seeds)
        ),
    }
    order_fee = float(base.order_fee)
    if least_squares is None:
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
    alias = ("anchor", "order_up_to") if published_anchor else None
    generalisation = {
        name: _comparison_table(
            holding_rate=IN_DISTRIBUTION_HOLDING_RATE,
            seeds=seeds[name],
            specs=episodes[name],
            arms=in_arms,
            alias=alias,
        )
        for name in ("in_distribution", "unseen_sku", "unseen_disruption")
    }
    in_table = generalisation["in_distribution"]
    sweep_refusal: str | None = None
    try:
        sweep = holding_rate_sweep(
            latent,
            ceiling_factories,
            n_seeds=n_seeds,
            seed_offset=seed_offset,
            config=base,
            assert_orders=_sweep_must_check_orders(base, assert_orders),
        )
    except UnfalsifiableTransfer as exc:
        # Keep the three generalisation tables. The sweep curve is withheld.
        sweep = ()
        sweep_refusal = str(exc)
    return SyntheticComparison(
        in_distribution=in_table,
        holding_rate_sweep=sweep,
        in_distribution_passes=in_distribution_passes(
            in_table.per_seed["latent"],
            in_table.per_seed["anchor"],
        ),
        in_distribution_paired_interval=in_table.paired_interval,
        unseen_sku=generalisation["unseen_sku"],
        unseen_disruption=generalisation["unseen_disruption"],
        holding_rate_sweep_refusal=sweep_refusal,
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
            row[f"latent_minus_{table.baseline}"] = float(table.paired_difference[index])
        if accuracy is not None:
            row["rollout_r2_latent"] = _mean_score(accuracy.latent)
            row["rollout_r2_direct"] = _mean_score(accuracy.direct)
            row["rollout_r2_ols"] = _mean_score(accuracy.ols)
        rows.append(row)
    return pd.DataFrame(rows)


def write_synthetic_comparison(result: SyntheticComparison, directory: str | Path) -> Path:
    """Write each protocol to its own parquet. Does not invent a cell."""
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    accuracy = result.rollout_accuracy
    tables = {
        "in_distribution": result.in_distribution,
        "unseen_sku": result.unseen_sku,
        "unseen_disruption": result.unseen_disruption,
    }
    for name, table in tables.items():
        _comparison_frame(table, accuracy).to_parquet(destination / f"{name}.parquet", index=False)
    if result.holding_rate_sweep:
        frames = []
        for table in result.holding_rate_sweep:
            frame = _comparison_frame(table, accuracy)
            frame.insert(0, "holding_rate", table.holding_rate)
            frames.append(frame)
        pd.concat(frames, ignore_index=True).to_parquet(
            destination / "holding_rate_sweep.parquet",
            index=False,
        )
    else:
        sweep = destination / "holding_rate_sweep.parquet"
        if sweep.exists():
            sweep.unlink()
    stale = destination / "retarget.parquet"
    if stale.exists():
        stale.unlink()
    warnings = {name: list(table.collinear_warnings) for name, table in tables.items()}
    warnings["holding_rate_sweep"] = [
        f"holding_rate {table.holding_rate:g}: {line}"
        for table in result.holding_rate_sweep
        for line in table.collinear_warnings
    ]
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


_PROFIT_ARTIFACTS = (
    "in_distribution.parquet",
    "unseen_sku.parquet",
    "unseen_disruption.parquet",
    "holding_rate_sweep.parquet",
    "retarget.parquet",
    "comparison_warnings.json",
)


def _record_profit_refusal(destination: Path) -> None:
    """Leave a record that the publish gate withheld the profit tables."""
    payload = {
        "refused": True,
        "reason": "held-out rollout accuracy did not beat the ordinary-least-squares baseline",
    }
    (destination / "profit_table_refused.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _clear_profit_refusal(destination: Path) -> None:
    refused = destination / "profit_table_refused.json"
    if refused.exists():
        refused.unlink()


def _record_sweep_refusal(destination: Path, reason: str) -> None:
    """The order-sequence check withheld the curve. The reason is the exception text."""
    payload = {"refused": True, "reason": reason}
    (destination / "holding_rate_sweep_refused.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _clear_sweep_refusal(destination: Path) -> None:
    refused = destination / "holding_rate_sweep_refused.json"
    if refused.exists():
        refused.unlink()


def _remove_profit_tables(destination: Path) -> None:
    """Drop profit tables so a refused re-run cannot leave a stale one."""
    for name in _PROFIT_ARTIFACTS:
        path = destination / name
        if path.exists():
            path.unlink()


def write_gated_profit_table(
    result: SyntheticComparison,
    directory: str | Path,
    accuracy: RolloutAccuracy,
) -> Path:
    """Write the profit tables when rollout accuracy beats least squares.

    The per-step curve is written either way. When the gate refuses, any
    protocol parquet already in the directory is removed and
    ``profit_table_refused.json`` records the refusal. The gate is not
    bypassed.
    """
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    _write_rollout_accuracy(accuracy, destination / "rollout_accuracy.json")
    if not rollout_beats_baseline(accuracy):
        _remove_profit_tables(destination)
        _clear_sweep_refusal(destination)
        _record_profit_refusal(destination)
        raise RolloutGateError(accuracy)
    _clear_profit_refusal(destination)
    return write_synthetic_comparison(replace(result, rollout_accuracy=accuracy), destination)


def score_rollout_gate(
    latent: ReplenishmentPlanner,
    direct: ReplenishmentPlanner,
    frame: pd.DataFrame,
    directory: str | Path,
) -> tuple[RolloutAccuracy, object]:
    """Write rollout accuracy and fit the least-squares floor.

    The floor does not keep ``frame``. A caller that drops ``frame``
    before the profit episodes does not hold the log for that scoring.
    """
    accuracy = held_out_rollout_accuracy(latent.dynamics, direct.dynamics, frame)
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    _write_rollout_accuracy(accuracy, destination / "rollout_accuracy.json")
    if not rollout_beats_baseline(accuracy):
        _remove_profit_tables(destination)
        _clear_sweep_refusal(destination)
        _record_profit_refusal(destination)
        raise RolloutGateError(accuracy)
    train_episodes = set(latent.dynamics.train_episodes) or None
    floor = fit_least_squares_dynamics(
        frame,
        delivery_lag=latent.delivery_lag,
        train_episodes=train_episodes,
    )
    return accuracy, floor


def finish_published_comparison(
    latent: ReplenishmentPlanner,
    direct: ReplenishmentPlanner,
    accuracy: RolloutAccuracy,
    least_squares: object,
    directory: str | Path,
    **compare_kwargs: object,
) -> SyntheticComparison:
    """Score the profit tables after the rollout gate has opened.

    An identical-order sweep refusal still writes the three generalisation
    tables and ``holding_rate_sweep_refused.json``. It does not invent a
    sweep curve.
    """
    compare_kwargs["assert_orders"] = True
    compare_kwargs["least_squares"] = least_squares
    destination = Path(directory)
    result = compare_synthetic(latent, direct, frame=None, **compare_kwargs)
    result = replace(result, rollout_accuracy=accuracy)
    _clear_profit_refusal(destination)
    if result.holding_rate_sweep_refusal is not None:
        sweep = destination / "holding_rate_sweep.parquet"
        if sweep.exists():
            sweep.unlink()
        _record_sweep_refusal(destination, result.holding_rate_sweep_refusal)
    else:
        _clear_sweep_refusal(destination)
    write_synthetic_comparison(result, destination)
    return result


def publish_synthetic_comparison(
    latent: ReplenishmentPlanner,
    direct: ReplenishmentPlanner,
    frame: pd.DataFrame,
    directory: str | Path,
    **compare_kwargs: object,
) -> SyntheticComparison:
    """Score rollout accuracy, then the profit tables if the gate opens.

    A rollout refusal writes ``profit_table_refused.json`` and no protocol
    parquet. The order-sequence check runs only after that gate opens. If
    the frozen planner's orders do not differ across holding rates, the
    three generalisation tables are still written,
    ``holding_rate_sweep.parquet`` is withheld, and
    ``holding_rate_sweep_refused.json`` records that reason.
    """
    accuracy, floor = score_rollout_gate(latent, direct, frame, directory)
    return finish_published_comparison(
        latent,
        direct,
        accuracy,
        floor,
        directory,
        **compare_kwargs,
    )
