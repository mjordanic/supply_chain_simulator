"""Optuna study orchestration and artifact serialisation for policy tuning.

``run_study`` is the top-level entry point: it builds a shared ``EpisodeSpec``
list (CRN at the trial level), runs ``n_trials`` Optuna trials calling
``evaluate_policy_normalised`` for each, and writes three artifact files to
``{output_dir}/{study_name}/``:

  - ``trials.parquet``   — one row per trial, all documented columns.
  - ``per_seed.parquet`` — long format, one row per (trial, seed).
  - ``study.json``       — self-describing study metadata.

CRN guarantee
-------------
The same ``eval_specs`` list is built once before ``study.optimize`` is called
and is closed over by the trial objective.  Every trial therefore evaluates on
bit-identical world trajectories for each seed; differences in aggregate KPIs
between trials are attributable to the policy kwargs alone.

Artifact portability
--------------------
The Parquet files are written with ``pandas``.  They load cleanly with only
``pandas`` + ``pyarrow`` — no Optuna import required at read time.  The JSON
file is written with the standard library ``json`` module.

``confirm_top_k`` is added in issue 05.
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


__all__ = ["run_study"]
