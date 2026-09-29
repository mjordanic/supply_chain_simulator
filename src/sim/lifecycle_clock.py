"""Per-SKU lifecycle stage clock (ADR 0002).

``advance_stage`` is a pure one-draw transition. ``LifecycleState`` owns
the current stage of every catalog SKU. Each SKU draws from its own RNG,
seeded from the episode's ``lifecycle`` sub-seed of ``world_seed``, so one
SKU's transition does not shift another's stream and the draws stay off
``world_rng``.
"""

from __future__ import annotations

from random import Random
from typing import Any

from src.sim.distributions import Distribution
from src.sim.episode_sampler import _derive_seed, _mix_seed
from src.sim.scenario import CANONICAL_STAGES, Scenario


def advance_stage(
    rng: Random,
    current_stage: str,
    stage_change_probs: dict[str, float],
) -> str:
    """Advance one step in ``CANONICAL_STAGES`` with probability ``p``.

    Always consumes one ``rng.random()`` draw. A missing stage is
    probability 0. From ``dead`` the next stage is ``introduction``.
    """
    prob = float(stage_change_probs.get(current_stage, 0.0))
    draw = rng.random()
    if draw < prob:
        idx = CANONICAL_STAGES.index(current_stage)
        return CANONICAL_STAGES[(idx + 1) % len(CANONICAL_STAGES)]
    return current_stage


def _resolve_scalar(value: Any, rng: Random) -> Any:
    if isinstance(value, Distribution):
        return value.sample(rng)
    return value


class LifecycleState:
    """Current stage and transition RNG for every catalog product."""

    def __init__(self, scenario: Scenario) -> None:
        params = scenario.item_lifecycle
        base = _derive_seed(scenario.world_seed, "lifecycle")
        self.stages: dict[str, str] = {}
        self.probs: dict[str, dict[str, float]] = {}
        self._rngs: dict[str, Random] = {}
        for ware in scenario.catalog:
            resolve = Random(_mix_seed(base, ware.product_id, 1))
            init_source = ware.init_stage if ware.init_stage is not None else params.init_stage
            init_stage = str(_resolve_scalar(init_source, resolve))
            raw_probs = (
                ware.stage_change_probs
                if ware.stage_change_probs is not None
                else params.default_stage_change_probs
            )
            probs = {
                stage: float(_resolve_scalar(prob, resolve))
                for stage, prob in raw_probs.items()
            }
            self.stages[ware.product_id] = init_stage
            self.probs[ware.product_id] = probs
            self._rngs[ware.product_id] = Random(_mix_seed(base, ware.product_id, 2))

    def tick(self) -> None:
        """Draw one transition per SKU."""
        for pid, stage in self.stages.items():
            self.stages[pid] = advance_stage(self._rngs[pid], stage, self.probs[pid])

    def demand_factors(self, stage_multipliers: dict[str, Any]) -> dict[str, float]:
        """Map each product to its current stage multiplier. Missing stages are 1."""
        factors: dict[str, float] = {}
        for pid, stage in self.stages.items():
            raw = stage_multipliers.get(stage, 1.0)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                factors[pid] = 1.0
            else:
                factors[pid] = float(raw)
        return factors


__all__ = ["LifecycleState", "advance_stage"]
