"""Unit tests for ``LifecycleClock.advance_stage`` (issue 02).

Pure-function contract:

  draw once against ``stage_change_probs[current_stage]``; if it fires,
  advance to the next stage in the canonical list
  ``[introduction, growth, maturity, decline, dead]`` (cyclic — from
  ``dead`` the next stage is ``introduction``); otherwise stay.

Tests use ``random.Random`` with fixed seeds where the draw value
matters; ``prob = 1.0`` / ``prob = 0.0`` cases sidestep the draw value
entirely.
"""

from __future__ import annotations

from random import Random

from src.sim.lifecycle_clock import CANONICAL_STAGES, advance_stage


def _all_zero_probs() -> dict[str, float]:
    return {s: 0.0 for s in CANONICAL_STAGES}


def _all_one_probs() -> dict[str, float]:
    return {s: 1.0 for s in CANONICAL_STAGES}


def test_canonical_stages_is_intro_growth_maturity_decline_dead() -> None:
    assert CANONICAL_STAGES == [
        "introduction",
        "growth",
        "maturity",
        "decline",
        "dead",
    ]


def test_prob_one_advances_every_tick() -> None:
    rng = Random(0)
    probs = _all_one_probs()
    stage = "introduction"
    expected = ["growth", "maturity", "decline", "dead", "introduction"]
    for nxt in expected:
        stage = advance_stage(rng, stage, probs)
        assert stage == nxt


def test_prob_zero_never_advances() -> None:
    rng = Random(0)
    probs = _all_zero_probs()
    for stage in CANONICAL_STAGES:
        for _ in range(20):
            assert advance_stage(rng, stage, probs) == stage


def test_dead_with_prob_one_returns_introduction() -> None:
    rng = Random(0)
    probs = _all_one_probs()
    assert advance_stage(rng, "dead", probs) == "introduction"


def test_dead_with_prob_zero_stays_dead() -> None:
    rng = Random(0)
    probs = _all_zero_probs()
    for _ in range(50):
        assert advance_stage(rng, "dead", probs) == "dead"


def test_cyclic_ordering_through_dead_back_to_introduction() -> None:
    """One full lap matches the canonical cycle."""
    rng = Random(0)
    probs = _all_one_probs()
    stage = "introduction"
    history = [stage]
    for _ in range(len(CANONICAL_STAGES)):
        stage = advance_stage(rng, stage, probs)
        history.append(stage)
    assert history == [
        "introduction",
        "growth",
        "maturity",
        "decline",
        "dead",
        "introduction",
    ]


def test_consumes_exactly_one_rng_draw_per_call() -> None:
    """Caller-visible contract: one ``rng.random()`` per ``advance_stage`` call.

    ItemRegistry.tick depends on this for CRN: every catalog item burns
    exactly one world_rng draw per tick regardless of whether it advances.
    """

    class _CountingRandom:
        def __init__(self) -> None:
            self.count = 0

        def random(self) -> float:
            self.count += 1
            return 0.999  # always above any reasonable prob

    rng = _CountingRandom()
    probs = {s: 0.5 for s in CANONICAL_STAGES}
    for stage in CANONICAL_STAGES:
        advance_stage(rng, stage, probs)  # type: ignore[arg-type]
    assert rng.count == len(CANONICAL_STAGES)


def test_missing_stage_in_probs_treated_as_zero() -> None:
    """Looking up a stage absent from the dict yields no transition.

    Defensive: if a Ware override only sets some stages, the unset ones
    default to "no transition" rather than raising.
    """
    rng = Random(0)
    probs = {"introduction": 1.0}  # other stages absent
    assert advance_stage(rng, "growth", probs) == "growth"
    assert advance_stage(rng, "decline", probs) == "decline"
