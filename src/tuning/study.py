"""Optuna study orchestration and artifact serialisation for policy tuning.

``run_study`` is the top-level entry point: it builds a shared ``EpisodeSpec``
list (CRN at the trial level), runs ``n_trials`` Optuna trials calling
``evaluate_policy_normalised`` for each, and writes three artifact files to
``{output_dir}/{study_name}/``:

  - ``trials.parquet``   — one row per trial, all documented columns.
  - ``per_seed.parquet`` — long format, one row per (trial, seed).
  - ``study.json``       — self-describing study metadata.

``confirm_top_k`` re-evaluates the top-K search trials plus the published
default on a disjoint held-out seed set, and writes:

  - ``holdout.parquet``        — one row per (label, seed).
  - ``holdout_summary.json``   — bootstrap confidence intervals + headline uplift.

CRN guarantee
-------------
The same ``eval_specs`` list is built once before ``study.optimize`` is called
and is closed over by the trial objective.  Every trial therefore evaluates on
bit-identical world trajectories for each seed; differences in aggregate KPIs
between trials are attributable to the policy kwargs alone.

The holdout specs are built once in ``confirm_top_k`` and reused for all K+1
policy evaluations (same CRN paired comparison on the held-out seeds).

Artifact portability
--------------------
The Parquet files are written with ``pandas``.  They load cleanly with only
``pandas`` + ``pyarrow`` — no Optuna import required at read time.  The JSON
file is written with the standard library ``json`` module.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Any, Callable

import optuna

from src.rl.configs.default import RLConfig
from src.rl.episode_sampler import EpisodeSpec, sample_episode
from src.sim.policy import Policy
from src.sim.scenario import StoreTemplate, Ware
from src.tuning.config import TuningConfig
from src.tuning.evaluator import evaluate_policy_normalised


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_eval_specs(
    catalog: list[Ware],
    base_template: StoreTemplate,
    rl_config: RLConfig,
    tuning_config: TuningConfig,
) -> list[EpisodeSpec]:
    """Build the shared CRN eval-spec list for the search phase.

    Seeds are taken from ``[tuning_config.seed_offset, seed_offset +
    n_search_seeds)``.  The list is built once at study start and reused for
    every trial objective call.
    """
    import dataclasses

    # Override episode_length in the config so specs reflect the tuning horizon.
    rl_config_with_length = dataclasses.replace(
        rl_config,
        episode_length=tuning_config.episode_length,
    )
    specs: list[EpisodeSpec] = []
    for i in range(tuning_config.n_search_seeds):
        seed = tuning_config.seed_offset + i
        spec = sample_episode(
            catalog=catalog,
            base_template=base_template,
            config=rl_config_with_length,
            episode_seed=seed,
        )
        specs.append(spec)
    return specs


def _get_git_sha() -> str:
    """Return the current HEAD SHA (40 hex chars) or empty string if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ""


def _introspect_policy_class_name(
    policy_space: Callable[[optuna.Trial], Policy],
) -> str:
    """Return a best-effort class name for the policy produced by policy_space.

    Probes the factory by creating a FixedTrial with a minimal parameter set
    that is common across all bundled factories.  If introspection fails
    (custom factory with non-standard params), returns the factory's
    ``__name__`` instead.
    """
    # We probe with a small set of known-common params.  The probe may fail
    # for unusual factories; we catch that and fall back.
    try:
        # Use a FixedTrial with a generous default parameter set that covers
        # all four bundled factories.
        probe_params = {
            "cover_horizon_ticks": 10,
            "safety_lead_pct_of_lag": 0.667,
            "opening_budget_pct": 0.5,
            "stockout_safety_bonus_pct_of_lag": 0.0,
            "Q": 5,
            "review_interval": 7,
        }
        trial = optuna.trial.FixedTrial(probe_params)
        policy = policy_space(trial)
        return type(policy).__name__
    except Exception:
        return getattr(policy_space, "__name__", repr(policy_space))


def _extract_search_space_from_trial(trial: optuna.Trial) -> dict[str, dict[str, Any]]:
    """Return a ``{param_name: {type, low, high}}`` dict from a completed trial.

    Reads ``trial.distributions`` (always populated after the trial completes).
    """
    import optuna.distributions as od

    result: dict[str, dict[str, Any]] = {}
    for name, dist in trial.distributions.items():
        if isinstance(dist, od.IntDistribution):
            result[name] = {"type": "int", "low": dist.low, "high": dist.high}
        elif isinstance(dist, od.FloatDistribution):
            result[name] = {"type": "float", "low": dist.low, "high": dist.high}
        elif isinstance(dist, od.CategoricalDistribution):
            result[name] = {"type": "categorical", "choices": list(dist.choices)}
        else:
            result[name] = {"type": type(dist).__name__}
    return result


# ---------------------------------------------------------------------------
# Public: run_study
# ---------------------------------------------------------------------------


_TRIALS_PARQUET_COLUMNS = [
    "trial_id",
    "state",
    "datetime_start",
    "datetime_complete",
    "duration_s",
    "mean_normalised_return",
    "mean_net_profit",
    "mean_service_level",
    "mean_stockout_rate",
    "mean_inventory_turnover",
    "mean_revenue",
    "mean_mean_price_pct_of_msrp",
]

_PER_SEED_PARQUET_COLUMNS = [
    "trial_id",
    "seed",
    "capacity",
    "initial_cash",
    "net_profit",
    "normalised_return",
    "service_level",
    "stockout_rate",
    "inventory_turnover",
    "revenue",
    "mean_price_pct_of_msrp",
]


def run_study(
    policy_space: Callable[[optuna.Trial], Policy],
    *,
    catalog: list[Ware],
    base_template: StoreTemplate,
    rl_config: RLConfig,
    tuning_config: TuningConfig,
    study_name: str,
    output_dir: str = "runs/tuning",
) -> optuna.Study:
    """Run an Optuna study and write artifact files to disk.

    Parameters
    ----------
    policy_space:
        Trial-callback factory: ``f(trial) -> Policy``.  Called once per
        trial inside the Optuna objective.  All four bundled factories in
        ``src.tuning.search_spaces`` are valid here; custom factories
        following the same signature are equally valid.
    catalog:
        Full product universe passed to ``sample_episode``.
    base_template:
        Non-episodic ``StoreTemplate`` knobs (region, delivery lag, …).
    rl_config:
        ``RLConfig`` instance; ``episode_length`` is overridden by
        ``tuning_config.episode_length`` when building eval specs.
    tuning_config:
        Immutable study configuration; see ``TuningConfig`` for all fields.
    study_name:
        Human-readable name for this study.  Used as the output sub-directory
        name under ``output_dir``.
    output_dir:
        Root directory for artifact output.  Artifacts land at
        ``{output_dir}/{study_name}/``.

    Returns
    -------
    optuna.Study
        The completed in-memory Optuna study.  Callers can inspect
        ``study.best_trial``, ``study.trials``, etc. without re-reading
        from disk.

    Side effects
    ------------
    Writes three files to ``{output_dir}/{study_name}/``:

    - ``trials.parquet``
    - ``per_seed.parquet``
    - ``study.json``
    """
    import pandas as pd

    started_at = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # 1. Build shared eval specs (CRN at the trial level).
    # ------------------------------------------------------------------
    eval_specs = _build_eval_specs(catalog, base_template, rl_config, tuning_config)

    # ------------------------------------------------------------------
    # 2. Create Optuna study (in-memory; no SQLite).
    # ------------------------------------------------------------------
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=tuning_config.sampler_seed),
        storage=None,
    )

    # ------------------------------------------------------------------
    # 3. Define the trial objective (closure over eval_specs).
    # ------------------------------------------------------------------
    # Accumulate per-seed rows across all trials into a shared list so
    # per_seed.parquet can be built in one pass after optimize().
    per_seed_rows: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        policy = policy_space(trial)
        kpis = evaluate_policy_normalised(
            lambda p=policy: p,  # return the already-constructed instance
            eval_specs,
            config=rl_config,
        )
        # Stash the full KPI dict in trial user_attrs for the artifact writer.
        trial.set_user_attr("kpis", kpis)
        return kpis["mean_normalised_return"]

    # ------------------------------------------------------------------
    # 4. Run the study (sequential; n_jobs=1; exceptions surface loudly).
    # ------------------------------------------------------------------
    study.optimize(
        objective,
        n_trials=tuning_config.n_trials,
        n_jobs=1,
        catch=(),
    )

    finished_at = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # 5. Prepare output directory.
    # ------------------------------------------------------------------
    study_dir = os.path.join(output_dir, study_name)
    os.makedirs(study_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 6. Build trials DataFrame.
    # ------------------------------------------------------------------
    # Collect per-seed rows while iterating trials (one pass).
    trials_rows: list[dict[str, Any]] = []

    for t in study.trials:
        kpis: dict[str, Any] = t.user_attrs.get("kpis", {})

        # Duration
        if t.datetime_start is not None and t.datetime_complete is not None:
            duration_s = (t.datetime_complete - t.datetime_start).total_seconds()
        else:
            duration_s = None

        row: dict[str, Any] = {
            "trial_id": t.number,
            "state": t.state.name,
            "datetime_start": t.datetime_start,
            "datetime_complete": t.datetime_complete,
            "duration_s": duration_s,
            "mean_normalised_return": kpis.get("mean_normalised_return"),
            "mean_net_profit": kpis.get("mean_net_profit"),
            "mean_service_level": kpis.get("mean_service_level"),
            "mean_stockout_rate": kpis.get("mean_stockout_rate"),
            "mean_inventory_turnover": kpis.get("mean_inventory_turnover"),
            "mean_revenue": kpis.get("mean_revenue"),
            "mean_mean_price_pct_of_msrp": kpis.get("mean_mean_price_pct_of_msrp"),
        }
        # Add one column per search-space parameter.
        for param_name, param_value in t.params.items():
            row[param_name] = param_value

        trials_rows.append(row)

        # Accumulate per-seed rows.
        n_seeds = tuning_config.n_search_seeds
        ps_net_profit = kpis.get("per_seed_net_profit", [None] * n_seeds)
        ps_initial_cash = kpis.get("per_seed_initial_cash", [None] * n_seeds)
        ps_capacity = kpis.get("per_seed_capacity", [None] * n_seeds)
        ps_service_level = kpis.get("per_seed_service_level", [None] * n_seeds)
        ps_stockout_rate = kpis.get("per_seed_stockout_rate", [None] * n_seeds)
        ps_inv_turnover = kpis.get("per_seed_inventory_turnover", [None] * n_seeds)
        ps_revenue = kpis.get("per_seed_revenue", [None] * n_seeds)
        ps_price_pct = kpis.get("per_seed_mean_price_pct_of_msrp", [None] * n_seeds)

        for i, spec in enumerate(eval_specs):
            seed = tuning_config.seed_offset + i
            net_profit_i = ps_net_profit[i] if i < len(ps_net_profit) else None
            initial_cash_i = ps_initial_cash[i] if i < len(ps_initial_cash) else None
            normalised_return_i = (
                net_profit_i / max(1e-9, initial_cash_i)
                if net_profit_i is not None and initial_cash_i is not None
                else None
            )
            per_seed_rows.append({
                "trial_id": t.number,
                "seed": seed,
                "capacity": ps_capacity[i] if i < len(ps_capacity) else None,
                "initial_cash": initial_cash_i,
                "net_profit": net_profit_i,
                "normalised_return": normalised_return_i,
                "service_level": ps_service_level[i] if i < len(ps_service_level) else None,
                "stockout_rate": ps_stockout_rate[i] if i < len(ps_stockout_rate) else None,
                "inventory_turnover": ps_inv_turnover[i] if i < len(ps_inv_turnover) else None,
                "revenue": ps_revenue[i] if i < len(ps_revenue) else None,
                "mean_price_pct_of_msrp": ps_price_pct[i] if i < len(ps_price_pct) else None,
            })

    trials_df = pd.DataFrame(trials_rows)
    per_seed_df = pd.DataFrame(per_seed_rows, columns=_PER_SEED_PARQUET_COLUMNS)

    # ------------------------------------------------------------------
    # 7. Write Parquet artifacts.
    # ------------------------------------------------------------------
    trials_df.to_parquet(os.path.join(study_dir, "trials.parquet"), index=False)
    per_seed_df.to_parquet(os.path.join(study_dir, "per_seed.parquet"), index=False)

    # ------------------------------------------------------------------
    # 8. Build and write study.json.
    # ------------------------------------------------------------------
    # Introspect the search space from the best trial's distributions
    # (or the first completed trial if best_trial is not defined).
    search_space: dict[str, dict[str, Any]] = {}
    for t in study.trials:
        if t.distributions:
            search_space = _extract_search_space_from_trial(t)
            break

    best_trial = study.best_trial
    study_meta: dict[str, Any] = {
        "study_name": study_name,
        "n_trials": tuning_config.n_trials,
        "n_search_seeds": tuning_config.n_search_seeds,
        "n_holdout_seeds": tuning_config.n_holdout_seeds,
        "episode_length": tuning_config.episode_length,
        "seed_offset": tuning_config.seed_offset,
        "holdout_seed_offset": tuning_config.holdout_seed_offset,
        "sampler_seed": tuning_config.sampler_seed,
        "search_space": search_space,
        "best_trial_id": best_trial.number,
        "best_params": best_trial.params,
        "best_value": best_trial.value,
        "started_at": started_at,
        "finished_at": finished_at,
        "git_sha": _get_git_sha(),
        "policy_class_name": _introspect_policy_class_name(policy_space),
    }

    with open(os.path.join(study_dir, "study.json"), "w", encoding="utf-8") as fh:
        json.dump(study_meta, fh, indent=2, default=str)

    return study


# ---------------------------------------------------------------------------
# Public: confirm_top_k
# ---------------------------------------------------------------------------


_HOLDOUT_PARQUET_COLUMNS = [
    "label",
    "trial_id",
    "seed",
    "capacity",
    "initial_cash",
    "net_profit",
    "normalised_return",
    "service_level",
    "stockout_rate",
    "inventory_turnover",
    "revenue",
    "mean_price_pct_of_msrp",
]

_HOLDOUT_SUMMARY_FIELDS = [
    "tuned_best_label",
    "tuned_best_mean_net_profit",
    "tuned_best_ci_low",
    "tuned_best_ci_high",
    "default_mean_net_profit",
    "default_ci_low",
    "default_ci_high",
    "headline_uplift",
    "headline_uplift_ci_low",
    "headline_uplift_ci_high",
    "n_holdout_seeds",
]


def _bootstrap_ci(
    values: list[float],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Return (ci_low, ci_high) via percentile bootstrap.

    Parameters
    ----------
    values:
        Per-seed numeric values.
    n_resamples:
        Number of bootstrap resamples.
    ci:
        Coverage probability (default 0.95 → 95% CI).
    seed:
        RNG seed for reproducibility.
    """
    import random

    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))

    boot_means: list[float] = []
    for _ in range(n_resamples):
        sample = [rng.choice(values) for _ in range(n)]
        boot_means.append(sum(sample) / n)

    boot_means.sort()
    alpha = 1.0 - ci
    lo_idx = int(alpha / 2 * n_resamples)
    hi_idx = int((1.0 - alpha / 2) * n_resamples) - 1
    hi_idx = min(hi_idx, n_resamples - 1)
    return (boot_means[lo_idx], boot_means[hi_idx])


def confirm_top_k(
    study: "optuna.Study",
    policy_space: Callable[["optuna.Trial"], Policy],
    *,
    catalog: list[Ware],
    base_template: StoreTemplate,
    rl_config: RLConfig,
    tuning_config: TuningConfig,
    study_dir: str,
) -> dict[str, Any]:
    """Re-evaluate the top-K search trials plus the published default on held-out seeds.

    Selects the top ``tuning_config.top_k_for_holdout`` completed trials by
    ``mean_normalised_return`` (descending), re-instantiates each policy via
    ``optuna.trial.FixedTrial(trial.params)`` passed to ``policy_space``, and
    evaluates them on ``tuning_config.n_holdout_seeds`` CRN seeds starting at
    ``tuning_config.holdout_seed_offset`` — disjoint from the search seeds.

    The published default (``OrderUpToPolicy()`` with no kwargs) is evaluated
    on the *same* holdout seeds for CRN paired comparison.

    Writes two artifact files:

    - ``holdout.parquet``       — one row per (label, seed).
      Columns: same as ``per_seed.parquet`` + ``label``.
      Row count: ``(top_k_for_holdout + 1) × n_holdout_seeds``.
    - ``holdout_summary.json``  — bootstrap CIs + headline uplift.

    Parameters
    ----------
    study:
        The completed ``optuna.Study`` returned by ``run_study``.
    policy_space:
        The same trial-callback factory used to run the original study.
        Re-invoked via ``FixedTrial`` to reconstruct the K best policies.
    catalog, base_template, rl_config, tuning_config:
        Same arguments as ``run_study``.
    study_dir:
        Absolute path to the study artifact directory (typically
        ``{output_dir}/{study_name}/``).  Artefacts are written here.

    Returns
    -------
    dict
        The ``holdout_summary.json`` payload (all documented fields).
    """
    import pandas as pd
    from src.sim.policy import OrderUpToPolicy

    # ------------------------------------------------------------------
    # 1. Build holdout eval specs (disjoint from search seeds).
    # ------------------------------------------------------------------
    import dataclasses as _dc

    rl_config_with_length = _dc.replace(
        rl_config,
        episode_length=tuning_config.episode_length,
    )
    holdout_specs: list[EpisodeSpec] = []
    from src.rl.episode_sampler import sample_episode as _sample_episode
    for i in range(tuning_config.n_holdout_seeds):
        seed = tuning_config.holdout_seed_offset + i
        spec = _sample_episode(
            catalog=catalog,
            base_template=base_template,
            config=rl_config_with_length,
            episode_seed=seed,
        )
        holdout_specs.append(spec)

    # ------------------------------------------------------------------
    # 2. Select top-K completed trials by mean_normalised_return.
    # ------------------------------------------------------------------
    completed = [
        t for t in study.trials
        if t.value is not None
    ]
    if not completed:
        raise ValueError("confirm_top_k: no completed trials found in study.")

    top_k = min(tuning_config.top_k_for_holdout, len(completed))
    sorted_trials = sorted(completed, key=lambda t: t.value, reverse=True)
    top_k_trials = sorted_trials[:top_k]

    # ------------------------------------------------------------------
    # 3. Evaluate each top-K trial + the published default on holdout specs.
    # ------------------------------------------------------------------
    holdout_rows: list[dict[str, Any]] = []
    per_label_net_profit: dict[str, list[float]] = {}

    def _evaluate_and_collect(
        label: str,
        policy_factory: Callable[[], Policy],
        trial_id: int | None,
    ) -> None:
        kpis = evaluate_policy_normalised(
            policy_factory,
            holdout_specs,
            config=rl_config,
        )
        net_profits: list[float] = kpis["per_seed_net_profit"]
        per_label_net_profit[label] = net_profits

        for i, spec in enumerate(holdout_specs):
            seed = tuning_config.holdout_seed_offset + i
            net_profit_i = kpis["per_seed_net_profit"][i]
            initial_cash_i = kpis["per_seed_initial_cash"][i]
            normalised_return_i = net_profit_i / max(1e-9, initial_cash_i)
            holdout_rows.append({
                "label": label,
                "trial_id": trial_id if trial_id is not None else -1,
                "seed": seed,
                "capacity": kpis["per_seed_capacity"][i],
                "initial_cash": initial_cash_i,
                "net_profit": net_profit_i,
                "normalised_return": normalised_return_i,
                "service_level": kpis["per_seed_service_level"][i],
                "stockout_rate": kpis["per_seed_stockout_rate"][i],
                "inventory_turnover": kpis["per_seed_inventory_turnover"][i],
                "revenue": kpis["per_seed_revenue"][i],
                "mean_price_pct_of_msrp": kpis["per_seed_mean_price_pct_of_msrp"][i],
            })

    # Evaluate top-K trials.
    for rank, trial in enumerate(top_k_trials, start=1):
        label = f"tuned_rank_{rank}"
        fixed_params = dict(trial.params)

        def _make_factory(params: dict[str, Any]) -> Callable[[], Policy]:
            def _factory() -> Policy:
                fixed_trial = optuna.trial.FixedTrial(params)
                return policy_space(fixed_trial)
            return _factory

        _evaluate_and_collect(label, _make_factory(fixed_params), trial.number)

    # Evaluate published default (OrderUpToPolicy with no kwargs).
    _evaluate_and_collect(
        "default",
        lambda: OrderUpToPolicy(),
        None,
    )

    # ------------------------------------------------------------------
    # 4. Write holdout.parquet.
    # ------------------------------------------------------------------
    os.makedirs(study_dir, exist_ok=True)
    holdout_df = pd.DataFrame(holdout_rows, columns=_HOLDOUT_PARQUET_COLUMNS)
    holdout_df.to_parquet(os.path.join(study_dir, "holdout.parquet"), index=False)

    # ------------------------------------------------------------------
    # 5. Compute bootstrap CIs.
    # ------------------------------------------------------------------
    # Find the label with the best mean net profit among the tuned ranks.
    tuned_labels = [f"tuned_rank_{r}" for r in range(1, top_k + 1)]
    best_label = max(
        tuned_labels,
        key=lambda lbl: sum(per_label_net_profit.get(lbl, [0.0])) / max(1, len(per_label_net_profit.get(lbl, [0.0]))),
    )
    tuned_best_profits = per_label_net_profit[best_label]
    default_profits = per_label_net_profit["default"]

    tuned_mean = sum(tuned_best_profits) / max(1, len(tuned_best_profits))
    default_mean = sum(default_profits) / max(1, len(default_profits))

    tuned_ci_low, tuned_ci_high = _bootstrap_ci(tuned_best_profits, seed=0)
    default_ci_low, default_ci_high = _bootstrap_ci(default_profits, seed=1)

    # Paired uplift: per-seed differences.
    paired_uplift = [
        t - d
        for t, d in zip(tuned_best_profits, default_profits)
    ]
    headline_uplift = sum(paired_uplift) / max(1, len(paired_uplift))
    uplift_ci_low, uplift_ci_high = _bootstrap_ci(paired_uplift, seed=2)

    # ------------------------------------------------------------------
    # 6. Write holdout_summary.json.
    # ------------------------------------------------------------------
    summary: dict[str, Any] = {
        "tuned_best_label": best_label,
        "tuned_best_mean_net_profit": tuned_mean,
        "tuned_best_ci_low": tuned_ci_low,
        "tuned_best_ci_high": tuned_ci_high,
        "default_mean_net_profit": default_mean,
        "default_ci_low": default_ci_low,
        "default_ci_high": default_ci_high,
        "headline_uplift": headline_uplift,
        "headline_uplift_ci_low": uplift_ci_low,
        "headline_uplift_ci_high": uplift_ci_high,
        "n_holdout_seeds": tuning_config.n_holdout_seeds,
    }

    with open(os.path.join(study_dir, "holdout_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    return summary


__all__ = ["run_study", "confirm_top_k"]


# ---------------------------------------------------------------------------
# CLI entry point: python -m src.tuning.study
# ---------------------------------------------------------------------------


def _cli_main(argv: list[str] | None = None) -> None:
    """Parse CLI flags and run a tuning study end-to-end.

    Canonical invocation::

        uv run python -m src.tuning.study \\
            --policy order_up_to \\
            --trials 150 \\
            --study-name order_up_to_v1

    See ``--help`` for all flags.
    """
    import argparse
    import dataclasses as _dc
    import sys

    from src.tuning.config import TuningConfig
    from src.tuning.search_spaces import (
        order_up_to_space,
        periodic_order_up_to_space,
        periodic_reorder_space,
        reorder_point_space,
    )

    _POLICY_DISPATCH = {
        "order_up_to": order_up_to_space,
        "reorder_point": reorder_point_space,
        "periodic_order_up_to": periodic_order_up_to_space,
        "periodic_reorder": periodic_reorder_space,
    }

    _defaults = TuningConfig()

    parser = argparse.ArgumentParser(
        prog="python -m src.tuning.study",
        description=(
            "Run an Optuna policy-hyperparameter study and write artifacts to disk.\n\n"
            "Artifacts land under <output-dir>/<study-name>/:\n"
            "  trials.parquet, per_seed.parquet, study.json\n"
            "  holdout.parquet, holdout_summary.json  (unless --skip-holdout)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--policy",
        required=True,
        choices=list(_POLICY_DISPATCH),
        help="Bundled trial-callback factory to use.",
    )
    parser.add_argument(
        "--study-name",
        required=True,
        help="Human-readable study name; also used as the output sub-directory.",
    )

    # Optional overrides for TuningConfig fields
    parser.add_argument(
        "--trials",
        type=int,
        default=_defaults.n_trials,
        metavar="N",
        help=f"Number of Optuna trials (default: {_defaults.n_trials}).",
    )
    parser.add_argument(
        "--n-search-seeds",
        type=int,
        default=_defaults.n_search_seeds,
        metavar="N",
        help=f"CRN seeds for the search phase (default: {_defaults.n_search_seeds}).",
    )
    parser.add_argument(
        "--n-holdout-seeds",
        type=int,
        default=_defaults.n_holdout_seeds,
        metavar="N",
        help=f"CRN seeds for the holdout phase (default: {_defaults.n_holdout_seeds}).",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=_defaults.seed_offset,
        metavar="N",
        help=f"First search-phase seed (default: {_defaults.seed_offset}).",
    )
    parser.add_argument(
        "--sampler-seed",
        type=int,
        default=_defaults.sampler_seed,
        metavar="N",
        help=f"TPESampler seed for reproducibility (default: {_defaults.sampler_seed}).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=_defaults.top_k_for_holdout,
        metavar="K",
        help=f"Number of top trials to re-evaluate on holdout (default: {_defaults.top_k_for_holdout}).",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=_defaults.episode_length,
        metavar="T",
        help=f"Episode length in ticks (default: {_defaults.episode_length}).",
    )

    # World selection
    parser.add_argument(
        "--world",
        default="rl_train",
        metavar="ARCHETYPE",
        help=(
            "World archetype; selects catalog + base template via the same "
            "resolution logic as the RL training pipeline (default: rl_train)."
        ),
    )

    # Output
    parser.add_argument(
        "--output-dir",
        default="runs/tuning",
        metavar="PATH",
        help="Root output directory (default: runs/tuning).",
    )

    # Skip holdout
    parser.add_argument(
        "--skip-holdout",
        action="store_true",
        default=False,
        help="Skip confirm_top_k re-evaluation (write only the three search artifacts).",
    )

    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Build TuningConfig with flag overrides.
    # ------------------------------------------------------------------
    tuning_config = TuningConfig(
        n_trials=args.trials,
        n_search_seeds=args.n_search_seeds,
        n_holdout_seeds=args.n_holdout_seeds,
        episode_length=args.episode_length,
        seed_offset=args.seed_offset,
        sampler_seed=args.sampler_seed,
        top_k_for_holdout=args.top_k,
    )

    # ------------------------------------------------------------------
    # Load world (catalog + base template) using the same resolution logic
    # as the RL training pipeline.
    # ------------------------------------------------------------------
    from src.rl.configs.default import RLConfig

    rl_config = _dc.replace(RLConfig(), world_archetype=args.world)

    # Reuse the world-loading helper from the training module.
    from src.rl.train import _load_world_catalog_and_template

    catalog, base_template, market_params, disruption_params = (
        _load_world_catalog_and_template(rl_config)
    )

    # ------------------------------------------------------------------
    # Select the policy factory.
    # ------------------------------------------------------------------
    policy_space = _POLICY_DISPATCH[args.policy]

    # ------------------------------------------------------------------
    # Run the study.
    # ------------------------------------------------------------------
    print(
        f"[tuning] Starting study '{args.study_name}' | policy={args.policy} "
        f"| trials={tuning_config.n_trials} | search_seeds={tuning_config.n_search_seeds} "
        f"| output={args.output_dir}/{args.study_name}",
        file=sys.stderr,
    )

    study = run_study(
        policy_space,
        catalog=catalog,
        base_template=base_template,
        rl_config=rl_config,
        tuning_config=tuning_config,
        study_name=args.study_name,
        output_dir=args.output_dir,
    )

    # ------------------------------------------------------------------
    # Optionally run confirm_top_k.
    # ------------------------------------------------------------------
    study_dir = os.path.join(args.output_dir, args.study_name)

    if not args.skip_holdout:
        confirm_top_k(
            study,
            policy_space,
            catalog=catalog,
            base_template=base_template,
            rl_config=rl_config,
            tuning_config=tuning_config,
            study_dir=study_dir,
        )

    # ------------------------------------------------------------------
    # Print one-line summary to stdout.
    # ------------------------------------------------------------------
    best = study.best_trial
    print(
        f"study={args.study_name} "
        f"n_trials={tuning_config.n_trials} "
        f"best_mean_normalised_return={best.value:.6f} "
        f"best_params={best.params} "
        f"output={study_dir}"
    )


if __name__ == "__main__":
    _cli_main()
