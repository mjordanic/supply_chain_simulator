"""Canonical episode seed-splitting and default-parameter helpers.

Seed splitting strategy
-----------------------
The single ``episode_seed`` is fanned into independent sub-seeds so
any subset of randomisations can be frozen for ablations without disturbing
the others:

  - ``assortment_seed`` — which K products are active this episode
  - ``capacity_seed``   — store capacity draw
  - ``balance_seed``    — opening cash draw
  - ``world_seed``      — ``Scenario.world_seed`` (drives all world-side RNG)

RL adds a fifth ``slot`` stream in ``src.rl.episode_sampler``; the slot
stream and ``slot_permutation`` are RL-observation-encoder concerns and
stay in the RL layer (see ADR 0004).

Sub-seeds are derived deterministically via a lightweight hash:
``sub = (episode_seed * PRIME + OFFSET) & 0xFFFF_FFFF`` where each
sub-purpose uses a different ``(PRIME, OFFSET)`` pair. No external
library required; the full derivation is unit-testable in one line.

No I/O, no module-level mutable state, no global RNG.

The three ``default_*_params()`` factories are public and are consumed by:
- ``src.rl.episode_sampler.sample_episode``'s ``is None`` fallbacks.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.sim.scenario import (
    DisruptionParams,
    ItemLifecycleParams,
    MarketParams,
    Scenario,
)


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeSpec:
    """Complete, self-contained description of one sim episode.

    Fields
    ------
    scenario
        A fully-initialised ``Scenario`` with ``n_steps == episode_length``
        and ``world_seed`` derived from ``episode_seed``.
    active_subset
        The K product ids selected for this episode, in catalog order.
        Retained so per-tick KPI traces can iterate the active SKUs in a
        stable order.

    Note: ``slot_permutation`` is deliberately absent — it is an
    RL-observation-encoder concern (ADR 0004). RL callers use
    ``RLEpisodeSpec`` defined in ``src.rl.episode_sampler``.
    """

    scenario: Scenario
    active_subset: tuple[str, ...]


# ---------------------------------------------------------------------------
# Sub-seed derivation
# ---------------------------------------------------------------------------

# Sub-purposes; RL adds "slot" as its own in its own module.
# "allocation" (issue 03) drives the per-phase buyer shuffle via
# allocation_rng = Random(_derive_seed(world_seed, "allocation")).
# Not yet consumed — wired up in issue 05 (GraphSimulation).
_SUB_SEED_PARAMS: dict[str, tuple[int, int]] = {
    "assortment": (0x9E37_79B9, 0x0000_0001),
    "capacity":   (0x6C62_272E, 0x0000_0002),
    "balance":    (0x517C_C1B7, 0x0000_0003),
    "world":      (0x27D4_EB2F, 0x0000_0004),
    "init_stock": (0x85EB_CA6B, 0x0000_0006),
    "allocation": (0xA24B_AED4, 0x0000_0007),
}
_MASK_32 = 0xFFFF_FFFF


def _derive_seed(episode_seed: int, purpose: str) -> int:
    """Return a deterministic 32-bit sub-seed for ``purpose``.

    The derivation is a single multiply-add-mask step; all four purposes
    produce different values for any non-degenerate ``episode_seed``.
    """
    prime, offset = _SUB_SEED_PARAMS[purpose]
    return (episode_seed * prime + offset) & _MASK_32


# ---------------------------------------------------------------------------
# Default Scenario parameters (synthetic-fallback defaults)
# ---------------------------------------------------------------------------


def default_market_params() -> MarketParams:
    """Return a sensible ``MarketParams`` for synthetic/fallback episodes.

    All fields are kept simple (scalar or low-variance distributions) so
    the training signal is dominated by the store's pricing / ordering
    decisions, not by exotic market dynamics.
    """
    from src.sim.distributions import Constant, Normal, Uniform

    return MarketParams(
        cycle_len=365,
        cycle_amp=0.3,
        init_demand=1.0,
        init_supply=1.0,
        peak_factor=1.2,
        off_factor=0.7,
        season_months={"all_season": list(range(1, 13))},
        regions=["US"],
        correlation=0.7,
        trend_update_interval=20,
        min_value=0.2,
        max_value=2.0,
        stage_multipliers={
            "introduction": 0.7,
            "growth": 1.5,
            "maturity": 1.0,
            "decline": 0.2,
            "dead": 0.05,
        },
        price_elasticity=-1.5,
        promo_multiplier=1.0,
        demand_factor_min=0.1,
        supply_factor_min=0.01,
        cross_inv_lo=0.3,
        cross_inv_hi=0.7,
        cross_factor_range=(0.3, 1.6),
        trend=Constant(1.0),
        demand_shock=Normal(0.0, 0.01),
        supply_shock=Normal(0.0, 0.01),
        base_demand=Uniform(2, 8),
    )


def default_disruption_params() -> DisruptionParams:
    """Return default ``DisruptionParams`` (event_prob=0.05)."""
    from src.sim.distributions import Constant

    return DisruptionParams(
        event_prob=0.05,
        types=["natural_disaster", "economic_crisis"],
        regions=["US"],
        severity=Constant(0.01),
        duration=Constant(3),
    )


def default_lifecycle_params() -> ItemLifecycleParams:
    """Return default ``ItemLifecycleParams`` (maturity, zero-transition)."""
    stages = ["introduction", "growth", "maturity", "decline", "dead"]
    return ItemLifecycleParams(
        stages=stages,
        init_stage="maturity",
        default_stage_change_probs={s: 0.0 for s in stages},
    )


__all__ = [
    "EpisodeSpec",
    "default_market_params",
    "default_disruption_params",
    "default_lifecycle_params",
]
