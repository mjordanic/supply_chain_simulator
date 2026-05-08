"""Pure-function lifecycle stage clock (issue 02).

Encapsulates the cyclic-with-terminal-default transition rule:

  draw once against ``stage_change_probs[current_stage]``; if it fires,
  advance to the next stage in the canonical list
  ``[introduction, growth, maturity, decline, dead]``; from ``dead`` the
  next stage is ``introduction`` (set ``dead → introduction = 0`` to
  recover strict-terminal behaviour).

Pure function, no internal state. ``ItemRegistry.tick`` calls
``advance_stage`` once per item per tick using the shared ``world_rng``;
the one-draw-per-call contract is what keeps CRN demand traces aligned
across paired stores even when their lifecycles diverge.
"""

from __future__ import annotations

from random import Random


CANONICAL_STAGES: list[str] = [
    "introduction",
    "growth",
    "maturity",
    "decline",
    "dead",
]


def advance_stage(
    rng: Random,
    current_stage: str,
    stage_change_probs: dict[str, float],
) -> str:
    """Probabilistically advance one cyclic step in ``CANONICAL_STAGES``.

    Always burns exactly one ``rng.random()`` draw, regardless of the
    probability or whether the transition fires — load-bearing for CRN.
    Stages absent from ``stage_change_probs`` are treated as probability
    ``0.0`` (no transition).
    """
    prob = stage_change_probs.get(current_stage, 0.0)
    draw = rng.random()
    if draw < prob:
        idx = CANONICAL_STAGES.index(current_stage)
        next_idx = (idx + 1) % len(CANONICAL_STAGES)
        return CANONICAL_STAGES[next_idx]
    return current_stage


__all__ = ["CANONICAL_STAGES", "advance_stage"]
