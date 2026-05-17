"""Tuning-side Config-adapter for the sim episode sampler.

Unpacks ``TuningConfig`` and delegates to ``src.sim.episode_sampler.sample_episode``.
The ``TuningEpisodeSpec`` name is retired; callers use ``EpisodeSpec`` from sim.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from src.sim.episode_sampler import EpisodeSpec, sample_episode as _sim_sample_episode
from src.sim.scenario import (
    DisruptionParams,
    ItemLifecycleParams,
    MarketParams,
    StoreTemplate,
    Ware,
)

if TYPE_CHECKING:
    from src.tuning.config import TuningConfig


# Transitional alias — retained so any callers using the old name still work.
TuningEpisodeSpec = EpisodeSpec


def sample_episode(
    catalog: list[Ware],
    base_template: StoreTemplate,
    config: "TuningConfig",
    episode_seed: int,
    *,
    market_params: MarketParams | None = None,
    disruption_params: DisruptionParams | None = None,
    lifecycle_params: ItemLifecycleParams | None = None,
    start_date: datetime | None = None,
) -> EpisodeSpec:
    """Tuning Config-adapter: unpack ``TuningConfig`` and call sim's sampler."""
    return _sim_sample_episode(
        catalog,
        base_template,
        K_active=config.K_active,
        episode_length=config.episode_length,
        capacity_dist=config.capacity_dist,
        balance_dist=config.balance_dist,
        episode_seed=episode_seed,
        init_stock_pct_dist=config.init_stock_pct_dist,
        market_params=market_params,
        disruption_params=disruption_params,
        lifecycle_params=lifecycle_params,
        start_date=start_date,
    )


__all__ = ["EpisodeSpec", "TuningEpisodeSpec", "sample_episode"]
