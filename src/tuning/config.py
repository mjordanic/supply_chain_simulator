"""TuningConfig — immutable configuration dataclass for hyperparameter tuning studies.

Fully self-contained: holds every knob the tuning module needs (episode shape,
randomisation distributions, simulator settings, world archetype, study
parameters). No dependency on any RL config.

Seed offset design
------------------
The default offsets are chosen to be disjoint from any other seed ranges used
elsewhere in the project:

- Tuning search:  ``seed_offset = 12_000_000``
- Tuning holdout: ``holdout_seed_offset = 13_000_000``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.sim.distributions import Distribution


def _default_capacity_dist() -> Distribution:
    """``LogUniform(100, 10_000)`` — per-episode capacity distribution.

    Log-uniform over two orders of magnitude so a single study spans
    small-store and flagship deployments.
    """
    from src.sim.distributions import LogUniform

    return LogUniform(100, 10_000)

def _default_init_stock_pct_dist() -> Distribution:
    """``Uniform(0.2, 1.0)`` — per-episode starting inventory fraction of capacity."""
    from src.sim.distributions import Uniform

    return Uniform(0.2, 1.0)


def _default_balance_dist() -> Distribution:
    """``LogUniform(10_000, 1_000_000)`` — per-episode opening balance distribution."""
    from src.sim.distributions import LogUniform

    return LogUniform(10_000, 1_000_000)


@dataclass(frozen=True)
class TuningConfig:
    """Immutable configuration for one Optuna-based policy tuning study."""

    # ------------------------------------------------------------------
    # Episode shape
    # ------------------------------------------------------------------
    episode_length: int = 365
    """Number of ticks per episode (one calendar year at daily resolution)."""

    K_active: int = 5
    """Number of active SKUs per episode (sampled from the catalog)."""

    K_catalog: int = 100
    """Catalog size used by the synthetic fallback when no world cache is found."""

    # ------------------------------------------------------------------
    # Episode randomisation
    # ------------------------------------------------------------------
    capacity_dist: Distribution = field(default_factory=_default_capacity_dist)
    balance_dist: Distribution = field(default_factory=_default_balance_dist)
    init_stock_pct_dist: Distribution = field(default_factory=_default_init_stock_pct_dist)

    # ------------------------------------------------------------------
    # Simulator knobs (forwarded into the StoreTemplate)
    # ------------------------------------------------------------------
    delivery_lag: int = 3
    holding_rate: float = 0.01
    order_fee: float = 50.0

    # ------------------------------------------------------------------
    # Study parameters
    # ------------------------------------------------------------------
    n_trials: int = 150
    """Number of Optuna trials (TPE proposals) to run."""

    n_search_seeds: int = 16
    """CRN seeds used to evaluate each trial during the search phase."""

    n_holdout_seeds: int = 32
    """CRN seeds used in the post-search confirmation phase."""

    seed_offset: int = 12_000_000
    """First seed for the search-phase eval set."""

    holdout_seed_offset: int = 13_000_000
    """First seed for the holdout-phase eval set (disjoint from search)."""

    sampler_seed: int = 42
    """Seed for Optuna's TPESampler."""

    top_k_for_holdout: int = 5
    """Number of top-ranked search trials to re-evaluate on the holdout set."""

    # ------------------------------------------------------------------
    # World
    # ------------------------------------------------------------------
    setup_dir: Optional[str] = None
    """Optional path to a setup directory (catalog.csv + setup.yaml).
    When set, the tuning stack loads catalog + market from this directory.
    When unset, a synthetic catalog is used (no LLM calls required)."""


__all__ = ["TuningConfig"]
