"""TuningConfig — immutable configuration dataclass for hyperparameter tuning studies.

All defaults are documented in ADR 0009.  ``TuningConfig()`` (no args)
produces a valid configuration for a full production study.  For smoke
tests and CI, override ``n_trials``, ``n_search_seeds``, and
``episode_length`` to shrink the run.

Seed offset design
------------------
The default offsets are chosen to be disjoint from all other seed ranges
in the project:

- Training: seeds ``[0, total_env_steps)``.
- CRN paired-eval: ``eval_seed_offset = 10_000_000`` (default in ``RLConfig``).
- Two-scale eval: ``eval_seed_offset + 1_000_000 = 11_000_000``.
- Tuning search: ``seed_offset = 12_000_000``.
- Tuning holdout: ``holdout_seed_offset = 13_000_000``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TuningConfig:
    """Immutable configuration for one Optuna-based policy tuning study.

    Fields
    ------
    n_trials:
        Number of Optuna trials (TPE proposals) to run. Default 150.
    n_search_seeds:
        Number of CRN seeds used to evaluate each trial during the search
        phase. Mean normalised return over these seeds is the objective.
        Default 16.
    n_holdout_seeds:
        Number of CRN seeds used in the post-search confirmation phase
        (``confirm_top_k``).  Disjoint from ``n_search_seeds`` by
        construction (different ``holdout_seed_offset``). Default 32.
    episode_length:
        Number of ticks per episode. Default 365 (one calendar year at
        daily resolution).
    seed_offset:
        First seed for the search-phase eval set.  Must be disjoint from
        the training seed space (``[0, total_env_steps)``) and from the
        CRN eval ranges (10_000_000 and 11_000_000). Default 12_000_000.
    holdout_seed_offset:
        First seed for the holdout-phase eval set.  Must be disjoint from
        ``seed_offset`` and all other ranges. Default 13_000_000.
    sampler_seed:
        Seed for Optuna's TPESampler.  Fixed so the proposal sequence is
        reproducible across study restarts with the same config. Default 42.
    top_k_for_holdout:
        How many top-ranked search trials to re-evaluate on the holdout
        set in ``confirm_top_k``.  Default 5.
    """

    n_trials: int = 150
    n_search_seeds: int = 16
    n_holdout_seeds: int = 32
    episode_length: int = 365
    seed_offset: int = 12_000_000       # disjoint from training (0) and from two-scale eval (10_000_000 + 1_000_000)
    holdout_seed_offset: int = 13_000_000
    sampler_seed: int = 42
    top_k_for_holdout: int = 5


__all__ = ["TuningConfig"]
