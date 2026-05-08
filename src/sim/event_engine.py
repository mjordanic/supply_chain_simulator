"""Typed ``EventEngine`` (issue 04).

Replaces the previous ``src/events/event_manager.py``. The spawn / apply
math is preserved verbatim, with two changes vs. the old module:

1. Randomness flows through an injected ``world_rng`` instead of the
   global ``random`` module.
2. ``DisruptionParams`` (typed dataclass with ``Distribution`` fields)
   replaces the old ``init_params`` / ``live_params`` dict pair.

``tick(market)`` returns the *newly spawned* ``WorldEvent`` for logging,
or ``None`` when no event spawned this step. The previous module returned
nothing; surfacing the spawned event is what lets the runner build its
``run_log["global"]["events"]["occurrences"]`` series.
"""

from __future__ import annotations

from random import Random
from typing import Callable

from src.sim.scenario import DisruptionParams


class WorldEvent:
    """Active macro event; mutates affected regions' market state per tick."""

    def __init__(
        self,
        event_type: str,
        severity: float,
        affected_regions: list[str],
        duration: int,
    ) -> None:
        self.event_type = event_type
        self.severity = severity
        self.affected_regions = affected_regions
        self.duration = duration

    def apply(self, market) -> None:
        """Apply one tick of impact to each affected region. Decrements duration."""
        impact = self.severity
        for region in self.affected_regions:
            state = market.market_state[region]
            if self.event_type in ("natural_disaster", "pandemic"):
                state["market_demand"] = max(market.min_value, state["market_demand"] - impact)
                state["market_supply"] = max(market.min_value, state["market_supply"] - impact)
            elif self.event_type in ("economic_crisis", "political_unrest"):
                state["market_demand"] = max(market.min_value, state["market_demand"] - impact)
            elif self.event_type == "technological_breakthrough":
                state["market_supply"] = min(market.max_value, state["market_supply"] + impact)
        self.duration -= 1


class FutureEvent:
    """Delayed callback scheduled for a future step."""

    def __init__(self, event_type: str, delay: int, callback: Callable[[], None]) -> None:
        self.event_type = event_type
        self.delay = delay
        self.callback = callback


class EventEngine:
    """Schedules world events and queued callbacks against an injected ``world_rng``."""

    def __init__(self, params: DisruptionParams, world_rng: Random) -> None:
        self.params = params
        self.rng = world_rng
        self.active: list[WorldEvent] = []
        self.queued: list[FutureEvent] = []

    def spawn_event(self) -> WorldEvent:
        """Sample a fresh ``WorldEvent`` from ``world_rng``.

        RNG draw order (preserved verbatim): choice(types) → severity →
        randint(1, n_regions) → sample(regions, n_regions) → duration.
        """
        event_type = self.rng.choice(list(self.params.types))
        severity = float(self.params.severity.sample(self.rng))
        n_regions = self.rng.randint(1, len(self.params.regions))
        regions = self.rng.sample(list(self.params.regions), n_regions)
        duration = int(self.params.duration.sample(self.rng))
        return WorldEvent(event_type, severity, regions, duration)

    def tick(self, market) -> WorldEvent | None:
        """Advance one step.

        Order (preserved verbatim from ``event_manager.EventEngine.tick``):
        apply active events → drop expired → spawn check → fire queued
        callbacks at or before ``market.current_step()``.
        """
        for event in self.active:
            event.apply(market)
        self.active = [e for e in self.active if e.duration > 0]

        new_event: WorldEvent | None = None
        if self.rng.random() < self.params.event_prob:
            new_event = self.spawn_event()
            self.active.append(new_event)

        current = market.current_step()
        fired: list[FutureEvent] = []
        for event in self.queued:
            if event.delay <= current:
                event.callback()
                fired.append(event)
        self.queued = [e for e in self.queued if e not in fired]

        return new_event

    def schedule(self, event_type: str, delay: int, callback: Callable[[], None]) -> None:
        """Queue ``callback`` to fire at simulation step ``delay``."""
        self.queued.append(FutureEvent(event_type, delay, callback))
        self.queued.sort(key=lambda x: x.delay)


__all__ = ["EventEngine", "FutureEvent", "WorldEvent"]
