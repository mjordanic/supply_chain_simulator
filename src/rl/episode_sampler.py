"""RL-side episode sampler: wraps sim's sampler and adds the slot stream.

``sample_episode(catalog, base_template, config, episode_seed) → RLEpisodeSpec``
composes ``src.sim.episode_sampler.sample_episode`` for the
``(scenario, active_subset)`` pair, then derives a 5th ``slot_seed``
(RL-specific) and produces the slot permutation.

``RLEpisodeSpec`` is a composition of sim's ``EpisodeSpec`` plus
``slot_permutation`` — an RL-observation-encoder concern (ADR 0004).
The slot stream stays in this module; the four shared streams (assortment /
capacity / balance / world) live in sim.

Seed splitting strategy
-----------------------
The single ``episode_seed`` is fanned into five independent sub-seeds:

  - ``assortment_seed`` — which K products are active (in sim)
  - ``capacity_seed``   — store capacity draw (in sim)
  - ``balance_seed``    — opening cash draw (in sim)
  - ``world_seed``      — ``Scenario.world_seed`` (in sim)
  - ``slot_seed``       — slot-permutation of the active subset (RL-only)

Sub-seeds are derived deterministically via a lightweight hash:
``sub = (episode_seed * PRIME + OFFSET) & 0xFFFF_FFFF``.

No I/O, no module-level mutable state, no global RNG.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from random import Random
from typing import TYPE_CHECKING

from src.sim.episode_sampler import (
    EpisodeSpec as _SimEpisodeSpec,
    _derive_seed as _sim_derive_seed,
    sample_episode as _sim_sample_episode,
)
from src.sim.scenario import (
    DisruptionParams,
    ItemLifecycleParams,
    MarketParams,
    StoreTemplate,
    Ware,
)

if TYPE_CHECKING:
    from src.rl.configs.default import RLConfig


# ---------------------------------------------------------------------------
# Public dataclass: RLEpisodeSpec composes sim's EpisodeSpec + slot_permutation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RLEpisodeSpec:
    """Complete, self-contained description of one RL episode.

    Fields
    ------
    spec
        The underlying sim ``EpisodeSpec`` with ``(scenario, active_subset)``.
    slot_permutation
        A tuple of length K mapping *slot index* → *position in
        ``spec.active_subset``*.  ``slot_permutation[slot] = k`` means "slot
        ``slot`` should display the SKU at index ``k`` of ``active_subset``".
        The inverse permutation (``active_subset`` index → slot) is the
        one applied when building the observation tensor.
    """

    spec: _SimEpisodeSpec
    slot_permutation: tuple[int, ...]

    # Convenience pass-throughs so existing RL callsites continue to work
    # without touching .spec.scenario / .spec.active_subset.
    @property
    def scenario(self):
        return self.spec.scenario

    @property
    def active_subset(self):
        return self.spec.active_subset


# ---------------------------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------------------------

# The historical name in this module; RL internal code used ``EpisodeSpec``.
# Now that sim defines ``EpisodeSpec``, the RL type is ``RLEpisodeSpec``.
# This alias keeps existing RL imports that reference ``EpisodeSpec`` in
# this module working during the transition. New code should use ``RLEpisodeSpec``.
EpisodeSpec = RLEpisodeSpec


# ---------------------------------------------------------------------------
# RL-only sub-seed constant (5th stream)
# ---------------------------------------------------------------------------

_SLOT_PRIME = 0x165667B1
_SLOT_OFFSET = 0x0000_0005
_MASK_32 = 0xFFFF_FFFF

# Re-export the shared params so test imports like
#   ``from src.rl.episode_sampler import _SUB_SEED_PARAMS``
# continue to work. The RL module adds "slot" on top of sim's four.
from src.sim.episode_sampler import _SUB_SEED_PARAMS as _SIM_SUB_SEED_PARAMS

_SUB_SEED_PARAMS: dict[str, tuple[int, int]] = {
    **_SIM_SUB_SEED_PARAMS,
    "slot": (_SLOT_PRIME, _SLOT_OFFSET),
}


def _derive_seed(episode_seed: int, purpose: str) -> int:
    """Return a deterministic 32-bit sub-seed for ``purpose``.

    Delegates to sim's ``_derive_seed`` for the four shared purposes;
    handles "slot" locally.
    """
    if purpose in _SIM_SUB_SEED_PARAMS:
        return _sim_derive_seed(episode_seed, purpose)
    if purpose == "slot":
        return (episode_seed * _SLOT_PRIME + _SLOT_OFFSET) & _MASK_32
    raise KeyError(f"Unknown sub-seed purpose: {purpose!r}")


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def sample_episode(
    catalog: list[Ware],
    base_template: StoreTemplate,
    config: "RLConfig",
    episode_seed: int,
    *,
    market_params: MarketParams | None = None,
    disruption_params: DisruptionParams | None = None,
    lifecycle_params: ItemLifecycleParams | None = None,
    start_date: datetime | None = None,
) -> RLEpisodeSpec:
    """Deterministically sample a fully-specified ``RLEpisodeSpec``.

    Parameters
    ----------
    catalog:
        The full product universe (length >= ``config.K_active``).
    base_template:
        A ``StoreTemplate`` carrying non-episodic knobs.
    config:
        ``RLConfig`` instance; only ``K_active``, ``capacity_dist``,
        ``balance_dist``, and ``episode_length`` are consumed here.
    episode_seed:
        Single integer seed. Any two calls with the same seed and the same
        ``(catalog, base_template, config)`` produce identical ``RLEpisodeSpec``
        objects — the function is pure.
    market_params, disruption_params, lifecycle_params:
        Optional overrides; fall back to sensible defaults when ``None``.
    start_date:
        Episode start date; defaults to 2024-01-01.

    Returns
    -------
    RLEpisodeSpec
        Fully deterministic description of the episode.
    """
    # --- call sim's sampler for the four shared streams ---
    sim_spec = _sim_sample_episode(
        catalog,
        base_template,
        K_active=config.K_active,
        episode_length=config.episode_length,
        capacity_dist=config.capacity_dist,
        balance_dist=config.balance_dist,
        episode_seed=episode_seed,
        market_params=market_params,
        disruption_params=disruption_params,
        lifecycle_params=lifecycle_params,
        start_date=start_date,
    )

    # --- derive RL-only slot permutation (5th stream) ---
    slot_seed = _derive_seed(episode_seed, "slot")
    slot_rng = Random(slot_seed)
    K = config.K_active
    perm = list(range(K))
    slot_rng.shuffle(perm)
    slot_permutation: tuple[int, ...] = tuple(perm)

    return RLEpisodeSpec(
        spec=sim_spec,
        slot_permutation=slot_permutation,
    )


__all__ = ["RLEpisodeSpec", "EpisodeSpec", "sample_episode"]
