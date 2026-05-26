"""Config-adapter: delegates world loading to ``src.sim.world_loader``.

``load_world(config)`` unpacks the relevant fields from ``TuningConfig``
and forwards them to ``src.sim.world_loader.load_world``.  All resolution
logic lives in the sim layer; this file is a thin shim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.sim.scenario import DisruptionParams, MarketParams, StoreTemplate, Ware
from src.sim.world_loader import load_world as _sim_load_world

if TYPE_CHECKING:
    from src.tuning.config import TuningConfig


def load_world(
    config: "TuningConfig",
) -> tuple[list[Ware], StoreTemplate, MarketParams | None, DisruptionParams | None]:
    """Resolve catalog + base StoreTemplate for a tuning study.

    Delegates to ``src.sim.world_loader.load_world``; see that module for
    the full three-tier resolution order.
    """
    return _sim_load_world(
        archetype=config.world_archetype,
        cache_path=config.world_cache_path,
        delivery_lag=config.delivery_lag,
        holding_rate=config.holding_rate,
        order_fee=config.order_fee,
        K_active=config.K_active,
        synthetic_fallback=True,
        synthetic_catalog_size=getattr(config, "K_catalog", 100),
    )


__all__ = ["load_world"]
