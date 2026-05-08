"""Integration tests for per-``Ware`` lifecycle overrides on ``ItemRegistry``.

Exercise via ``tick()`` and observe the resulting ``stage(...)`` — don't
mock ``LifecycleClock`` (issue 02 testing notes).
"""

from __future__ import annotations

from random import Random

from src.sim.item_registry import ItemRegistry
from src.sim.scenario import ItemLifecycleParams, Ware, load_catalog


_CANONICAL_STAGES = ["introduction", "growth", "maturity", "decline", "dead"]


def _catalog() -> list[Ware]:
    return load_catalog(
        [
            {
                "name": "A",
                "category": "X",
                "related_products": [],
                "base_price": 1.0,
                "unit_cost": 0.5,
                "seasonality": "all_season",
            },
            {
                "name": "B",
                "category": "X",
                "related_products": [],
                "base_price": 2.0,
                "unit_cost": 1.0,
                "seasonality": "all_season",
            },
        ]
    )


def _params(default_prob: float = 0.0, init_stage: str = "maturity") -> ItemLifecycleParams:
    return ItemLifecycleParams(
        stages=_CANONICAL_STAGES,
        init_stage=init_stage,
        default_stage_change_probs={s: default_prob for s in _CANONICAL_STAGES},
    )


def test_default_init_stage_applied_when_ware_has_no_override() -> None:
    registry = ItemRegistry(_params(init_stage="growth"), _catalog(), Random(0))
    assert registry.stage("P0000") == "growth"
    assert registry.stage("P0001") == "growth"


def test_per_ware_init_stage_override_wins_over_default() -> None:
    cat = _catalog()
    cat[0] = cat[0]._replace(init_stage="introduction")
    registry = ItemRegistry(_params(init_stage="maturity"), cat, Random(0))
    assert registry.stage("P0000") == "introduction"
    assert registry.stage("P0001") == "maturity"


def test_per_ware_stage_change_probs_override_drives_transitions() -> None:
    """An override of ``intro: 1.0`` advances the overridden ware every tick;
    other wares (no override, default 0.0) stay put."""
    cat = _catalog()
    cat[0] = cat[0]._replace(
        init_stage="introduction",
        stage_change_probs={s: 1.0 for s in _CANONICAL_STAGES},
    )
    registry = ItemRegistry(_params(init_stage="maturity"), cat, Random(0))
    assert registry.stage("P0000") == "introduction"
    assert registry.stage("P0001") == "maturity"

    registry.tick()
    assert registry.stage("P0000") == "growth"
    assert registry.stage("P0001") == "maturity"

    registry.tick()
    assert registry.stage("P0000") == "maturity"
    assert registry.stage("P0001") == "maturity"


def test_default_probs_apply_when_ware_override_is_none() -> None:
    """Without an override, items follow ``default_stage_change_probs``."""
    cat = _catalog()
    params = ItemLifecycleParams(
        stages=_CANONICAL_STAGES,
        init_stage="introduction",
        default_stage_change_probs={s: 1.0 for s in _CANONICAL_STAGES},
    )
    registry = ItemRegistry(params, cat, Random(0))
    assert registry.stage("P0000") == "introduction"
    registry.tick()
    assert registry.stage("P0000") == "growth"
    assert registry.stage("P0001") == "growth"


def test_dead_to_introduction_cycle_via_tick() -> None:
    """End-to-end: an item starting in ``dead`` with prob=1 wraps to
    ``introduction`` on the next tick — the cyclic-with-terminal-default
    semantics, observed through ``ItemRegistry.tick()``."""
    cat = _catalog()
    cat[0] = cat[0]._replace(init_stage="dead")
    params = ItemLifecycleParams(
        stages=_CANONICAL_STAGES,
        init_stage="dead",
        default_stage_change_probs={s: 1.0 for s in _CANONICAL_STAGES},
    )
    registry = ItemRegistry(params, cat, Random(0))
    assert registry.stage("P0000") == "dead"
    registry.tick()
    assert registry.stage("P0000") == "introduction"


def test_tick_burns_one_world_rng_draw_per_item() -> None:
    """CRN contract: every catalog item consumes exactly one ``world_rng``
    draw per tick, independent of whether a transition fires."""

    class _CountingRng:
        def __init__(self) -> None:
            self.count = 0

        def random(self) -> float:
            self.count += 1
            return 0.999  # never below any reasonable prob

    rng = _CountingRng()
    registry = ItemRegistry(_params(default_prob=0.5), _catalog(), rng)  # type: ignore[arg-type]
    rng.count = 0  # reset after construction draws (none expected with float probs)
    registry.tick()
    assert rng.count == len(_catalog())
