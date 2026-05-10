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

# ``Random`` is imported only for the type annotation — the function
# itself works with anything that exposes a ``.random()`` method.
from random import Random


# Canonical lifecycle stages, ordered. ``advance_stage`` reads this list
# to compute "the next stage" via modular arithmetic. Renaming entries
# silently breaks every downstream consumer that compares strings (e.g.
# ``MarketParams.stage_multipliers``) — coordinate updates if you must.
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
    probability or whether the transition fires — load-bearing for CRN
    (Common Random Numbers): if the draw count varied per item we
    couldn't keep paired stores aligned. Stages absent from
    ``stage_change_probs`` are treated as probability ``0.0`` (no
    transition).
    """
    # Probability of transition for this stage; missing key ⇒ stage is
    # treated as "no transition allowed" rather than raising.
    prob = stage_change_probs.get(current_stage, 0.0)
    # Single RNG draw — *must* happen unconditionally (see docstring).
    draw = rng.random()
    if draw < prob:
        # Index of the current stage in the canonical list. Will raise
        # ``ValueError`` if ``current_stage`` is not a canonical stage,
        # which is the right behaviour — we want loud failure not
        # silent freezing on a typo.
        idx = CANONICAL_STAGES.index(current_stage)
        # ``% len(...)`` makes the lifecycle cyclic: ``dead → introduction``.
        # Set the ``dead`` entry of ``stage_change_probs`` to ``0.0`` if
        # you want strict-terminal behaviour.
        next_idx = (idx + 1) % len(CANONICAL_STAGES)
        return CANONICAL_STAGES[next_idx]
    return current_stage


__all__ = ["CANONICAL_STAGES", "advance_stage"]
