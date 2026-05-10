"""Typed ``EventEngine`` (issue 04).

Generates stochastic disruption events (natural disasters, economic
crises, pandemics, political unrest, technological breakthroughs) that
shift regional demand/supply, and dispatches *scheduled callbacks* used
for order arrivals.

Two construction-shape changes vs. the previous
``src/events/event_manager.py``:

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
    """Active macro event; mutates affected regions' market state per tick.

    Carries its own remaining ``duration`` so the engine can drop it
    when it expires. The exact demand/supply mutation depends on
    ``event_type`` — see ``apply``.
    """

    def __init__(
        self,
        event_type: str,
        severity: float,
        affected_regions: list[str],
        duration: int,
    ) -> None:
        # Discriminator selecting which mutation branch in ``apply`` to
        # run (natural_disaster, pandemic, economic_crisis, etc.).
        self.event_type = event_type
        # Magnitude of the per-tick demand/supply mutation.
        self.severity = severity
        # Region keys (must match keys in ``market.market_state``) that
        # the event impacts. Other regions are untouched.
        self.affected_regions = affected_regions
        # Remaining lifetime in ticks; decremented each ``apply``.
        self.duration = duration

    def apply(self, market) -> None:
        """Apply one tick of impact to each affected region. Decrements duration."""
        # Per-tick mutation magnitude is the configured severity scalar.
        impact = self.severity
        for region in self.affected_regions:
            # Direct reference into the market's mutable state dict —
            # changes propagate to subsequent ``market.tick`` reads.
            state = market.market_state[region]
            if self.event_type in ("natural_disaster", "pandemic"):
                # Hits BOTH demand and supply (people stay home, factories halt).
                state["market_demand"] = max(market.min_value, state["market_demand"] - impact)
                state["market_supply"] = max(market.min_value, state["market_supply"] - impact)
            elif self.event_type in ("economic_crisis", "political_unrest"):
                # Hits demand only (people defer purchases).
                state["market_demand"] = max(market.min_value, state["market_demand"] - impact)
            elif self.event_type == "technological_breakthrough":
                # Positive supply shock; doesn't affect demand.
                state["market_supply"] = min(market.max_value, state["market_supply"] + impact)
        # One tick of life consumed; ``EventEngine.tick`` drops the
        # event from ``active`` once this hits zero.
        self.duration -= 1


class FutureEvent:
    """Delayed callback scheduled for a future step.

    Used for order arrivals: ``Runner._dispatch_orders`` queues a
    ``FutureEvent`` whose ``callback`` calls ``store.deliver`` at the
    computed arrival tick.
    """

    def __init__(self, event_type: str, delay: int, callback: Callable[[], None]) -> None:
        # Label for the scheduled callback (mostly for logs/debugging).
        self.event_type = event_type
        # *Absolute* simulation step at which to fire — the field is
        # named ``delay`` for historical reasons but it's compared
        # against ``market.current_step()`` directly in ``EventEngine.tick``.
        self.delay = delay
        # Zero-arg side-effecting callable, e.g. ``lambda: store.deliver(pid, qty)``.
        self.callback = callback


class EventEngine:
    """Schedules world events and queued callbacks against an injected ``world_rng``."""

    def __init__(self, params: DisruptionParams, world_rng: Random) -> None:
        # Disruption parameters (event_prob, types, severity, duration,
        # region pool). All ``Distribution`` fields stay un-sampled until
        # ``spawn_event`` actually needs a number.
        self.params = params
        # Shared world RNG. Every random choice in this module pulls
        # from it; never the global ``random`` module.
        self.rng = world_rng
        # Live events impacting markets each tick. Length grows when an
        # event spawns and shrinks when one expires.
        self.active: list[WorldEvent] = []
        # Delayed callbacks (order arrivals). Sorted by ``delay`` after
        # every insertion so the per-tick scan can stop early in
        # principle (currently scans full list — fine for our sizes).
        self.queued: list[FutureEvent] = []

    def spawn_event(self) -> WorldEvent:
        """Sample a fresh ``WorldEvent`` from ``world_rng``.

        RNG draw order (preserved verbatim): choice(types) → severity →
        randint(1, n_regions) → sample(regions, n_regions) → duration.
        Reordering breaks CRN identity across runs sharing the same
        ``world_seed``.
        """
        # Pick which type of disruption.
        event_type = self.rng.choice(list(self.params.types))
        # Sample (or pass through) the configured severity Distribution.
        severity = float(self.params.severity.sample(self.rng))
        # How many regions does this event hit? Always ≥ 1, ≤ len(regions).
        n_regions = self.rng.randint(1, len(self.params.regions))
        # Which specific regions — uniform random subset of size n_regions.
        regions = self.rng.sample(list(self.params.regions), n_regions)
        # How many ticks the event lasts.
        duration = int(self.params.duration.sample(self.rng))
        return WorldEvent(event_type, severity, regions, duration)

    def tick(self, market) -> WorldEvent | None:
        """Advance one step.

        Order (preserved verbatim from ``event_manager.EventEngine.tick``):
        apply active events → drop expired → spawn check → fire queued
        callbacks at or before ``market.current_step()``.
        """
        # 1. Apply active events to the market state.
        for event in self.active:
            event.apply(market)
        # 2. Drop expired events.
        self.active = [e for e in self.active if e.duration > 0]

        # 3. Bernoulli check for spawning a new event this tick.
        new_event: WorldEvent | None = None
        if self.rng.random() < self.params.event_prob:
            new_event = self.spawn_event()
            self.active.append(new_event)

        # 4. Fire any callbacks whose ``delay`` step has been reached.
        current = market.current_step()
        fired: list[FutureEvent] = []
        for event in self.queued:
            if event.delay <= current:
                event.callback()
                fired.append(event)
        # Drop fired callbacks. ``not in fired`` is O(n²) in the worst
        # case but ``fired`` is tiny in practice.
        self.queued = [e for e in self.queued if e not in fired]

        return new_event

    def schedule(self, event_type: str, delay: int, callback: Callable[[], None]) -> None:
        """Queue ``callback`` to fire at simulation step ``delay``."""
        self.queued.append(FutureEvent(event_type, delay, callback))
        # Keep the queue sorted so the head is always the earliest-due
        # callback — readers can rely on this even if the tick loop
        # currently scans the full list.
        self.queued.sort(key=lambda x: x.delay)


__all__ = ["EventEngine", "FutureEvent", "WorldEvent"]
