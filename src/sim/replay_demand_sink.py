"""ReplayDemandSinkNode — demand sink driven by an observed time series.

Replaces the stochastic ``demand_dist`` draw with ``series[tick]``, while
keeping the full ADR 0015 multiplier chain (market multiplier × lifecycle
stage × freshness) applied on top.  This is the "tracer bullet" for
real-data demand replay (issue 02); setup-dir serialization follows in
issue 03.

CRN invariant (ADR 0003 / ADR 0020): the override consumes exactly as
many ``world_rng`` draws as the base class's one-draw-per-catalog-product
loop — one call to ``world_rng.random()`` per catalog item — so that
mixed replay/stochastic graphs keep all other RNG streams bit-identical.

Usage::

    sink = ReplayDemandSinkNode(
        id="sink-0",
        region="US",
        init_seed=0,
        product_id="P0000",
        series=[10, 12, 8, ...],
        n_steps=100,
        income_rate=1_000_000.0,
        cash=0.0,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import TYPE_CHECKING

from src.sim.node import DemandSinkNode

if TYPE_CHECKING:
    from src.sim.market import Market
    from src.sim.scenario import Ware


@dataclass
class ReplayDemandSinkNode(DemandSinkNode):
    """A demand sink that replays an observed series instead of sampling a distribution.

    Fields
    ------
    series:
        Per-tick demand values.  ``series[tick]`` is the base demand for
        that tick before the multiplier chain is applied.  Length must be
        ≥ ``n_steps`` — validated at construction time.
    n_steps:
        Number of simulation steps the scenario will run.  Used only for
        the length check; not stored beyond ``__post_init__``.
    """

    series: list[int] = field(default_factory=list)
    n_steps: int = field(default=0)

    def __post_init__(self) -> None:
        if len(self.series) < self.n_steps:
            raise ValueError(
                f"ReplayDemandSinkNode '{self.id}': series length {len(self.series)} "
                f"is less than n_steps {self.n_steps}. "
                "Provide a series with at least n_steps entries."
            )

    @property
    def _node_type(self) -> str:
        return "replay_demand_sink"

    def demand_target(
        self,
        tick: int,
        market: "Market",
        catalog: "list[Ware]",
        world_rng: Random,
    ) -> int:
        """Compute demand target using the replay series for this tick.

        The market multiplier chain is applied exactly as in the base class.
        One ``world_rng.random()`` draw is burned per catalog item (discarded)
        to preserve CRN alignment with graphs that mix replay and stochastic
        sinks (ADR 0003 / ADR 0020).

        Parameters
        ----------
        tick:
            Current simulation tick (0-based).
        market:
            Live ``Market`` instance used for the demand multiplier.
        catalog:
            Scenario catalog.  Iterated in declaration order to burn the
            correct number of world_rng draws (ADR 0003).
        world_rng:
            Shared world RNG.  Exactly ``len(catalog)`` draws are consumed
            per call, then discarded.  The actual demand comes from ``series[tick]``.

        Returns
        -------
        int
            Realised demand target (≥ 0) for ``self.product_id``.
        """
        result_for_product = 0

        for ware in catalog:
            pid = ware.product_id
            # Market multiplier — deterministic, no RNG draw (ADR 0015).
            market_mult = market.demand_multiplier(pid, self.region)

            # Burn one world_rng draw per catalog item to preserve CRN
            # alignment (ADR 0003 / ADR 0020).  The draw is discarded.
            world_rng.random()

            if pid == self.product_id:
                # ``tick`` is 1-based (matches ``market.current_step()`` which
                # increments before demand_target is called); map to 0-based
                # series index.
                base = float(self.series[tick - 1])
                raw = base * market_mult
                result_for_product = max(0, int(raw))

        return result_for_product


__all__ = ["ReplayDemandSinkNode"]
