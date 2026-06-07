"""Node hierarchy — typed node ABCs and concrete subclasses.

Three concrete node types share a ``Node`` ABC:

- ``FactoryNode``       — produces one product, absorbs cash at unit_cost.
- ``IntermediateNode``  — holds inventory, sets prices, routes orders across
                         multiple upstream suppliers. Warehouse vs. shop is
                         expressed via the free-form ``tags`` attribute.
- ``DemandSinkNode``    — generates cash at ``income_rate``, demands units
                         from a ``demand_dist``, buys from intermediates.

``Node`` carries the shared fields: ``id``, ``region``, ``policy``,
``init_seed``, and the computed ``level`` (set by ``Graph.compute_levels``
after graph construction; ``None`` until set). Per-class fields are documented
on each subclass in line with the PRD's "Node hierarchy" section.

This module is pure data — no imports from other ``src.sim`` modules except
``distributions``. Side-effect free; safe to import from tests.

Design note: nodes are *not* frozen dataclasses because the simulation engine
mutates runtime fields (``inventory``, ``cash``, ``pending``, …) in place
during each tick. Read-only construction fields (``id``, ``region``,
``init_seed``) are invariant after construction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from random import Random
from typing import TYPE_CHECKING, Any

from src.sim.distributions import Distribution

if TYPE_CHECKING:
    from src.sim.market import Market
    from src.sim.policy import NodePolicy
    from src.sim.scenario import Ware


@dataclass
class Node(ABC):
    """Abstract base for all node types in the supply-chain graph.

    Fields
    ------
    id:
        Unique node identifier within a scenario.
    region:
        Region key — must match a key in ``MarketParams.regions``.
    init_seed:
        Per-node RNG seed for step-0 state initialisation. Two nodes
        constructed from the same ``(NodeSubclass, init_seed)`` start
        bit-identical, regardless of attached policy.
    policy:
        Decision-making brain. ``None`` = no decisions (useful in tests /
        skeleton phases). Not serialised — re-attached by the scenario
        loading code.
    level:
        Echelon level in the DAG (0 = factory, increasing toward sink).
        Computed by ``Graph.compute_levels`` after graph construction;
        ``None`` until set.
    """

    id: str
    region: str
    init_seed: int
    policy: "NodePolicy | None" = field(default=None)
    level: int | None = field(default=None, init=False)

    # Abstract marker — prevents direct instantiation of Node.
    # Dataclasses + ABC: we need at least one abstract method.
    @property
    @abstractmethod
    def _node_type(self) -> str:
        """Return a string identifying the concrete node type."""


@dataclass
class FactoryNode(Node):
    """A factory that produces exactly one product each tick.

    Factories are the lowest echelon (level 0). They sell at ``list_price``
    which is always equal to ``unit_cost`` (ADR 0013: zero-margin factories).

    Fields
    ------
    produces_product_id:
        The single product ID this factory manufactures.
    unit_cost:
        Manufacturing cost per unit. ``list_price`` must equal this.
    capacity_per_tick:
        Maximum units producible per tick. Scalar or Distribution (sampled
        at construction time from ``init_seed``).
    inventory:
        Current on-hand stock (units). Starts at 0 by default.
    list_price:
        Selling price per unit. Must equal ``unit_cost`` (zero-margin).
    """

    produces_product_id: str = ""
    unit_cost: float = 0.0
    capacity_per_tick: int | float | Distribution = 0
    inventory: int = 0
    list_price: float = 0.0
    cash: float = 0.0

    @property
    def _node_type(self) -> str:
        return "factory"


@dataclass
class IntermediateNode(Node):
    """A distribution centre, warehouse, or shop — the intermediate tier.

    IntermediateNodes sit between factories (suppliers) and demand-sinks
    (buyers). They hold inventory, set selling prices, impose per-product
    minimum order sizes, and accumulate pending deliveries in transit.

    The ``tags`` list is informational only — it never branches the
    simulation mechanics. Use it to label tier roles in scenario authoring
    (e.g. ``["warehouse"]``, ``["shop", "premium"]``).

    Fields
    ------
    carried_products:
        Set of product IDs this node stocks.
    capacity:
        Total inventory capacity (sum across all products). Scalar or
        Distribution.
    tags:
        Free-form tier labels. Warehouse vs. shop distinction lives here.
    inventory:
        Current on-hand stock per product: ``{pid: qty}``.
    pending:
        In-transit deliveries per upstream supplier per product:
        ``{supplier_id: {pid: qty}}``.
    list_prices:
        Selling price per product: ``{pid: price}``.
    min_order_imposed:
        Minimum order size this node accepts from buyers, per product:
        ``{pid: min_qty}``. Orders below this threshold are rejected by
        the allocator.
    """

    carried_products: set[str] = field(default_factory=set)
    capacity: int | float | Distribution = 0
    tags: list[str] = field(default_factory=list)
    inventory: dict[str, int] = field(default_factory=dict)
    pending: dict[str, dict[str, int]] = field(default_factory=dict)
    list_prices: dict[str, float] = field(default_factory=dict)
    min_order_imposed: dict[str, int] = field(default_factory=dict)
    cash: float = 0.0

    @property
    def _node_type(self) -> str:
        return "intermediate"


@dataclass
class DemandSinkNode(Node):
    """A consumer group — the demand source in the graph.

    Demand-sinks are the highest echelon. They generate cash at
    ``income_rate`` each tick and buy units from upstream intermediates.
    Each sink is bound to a single product; ``demand_dist`` drives the
    per-tick demand target (multiplied by market factors).

    Fields
    ------
    product_id:
        The single product ID this sink demands.
    demand_dist:
        Base demand distribution — sampled each tick via ``world_rng``.
    income_rate:
        Cash created by the sink each tick (the only source of new cash
        in the system per ADR 0013).
    cash:
        Current cash balance. Grows by ``income_rate`` each tick; decreases
        as allocation payments land.
    activation_tick:
        Kept for RL/tuning compatibility; not used by ``demand_target``.
    """

    product_id: str = ""
    demand_dist: Distribution | None = None
    income_rate: float = 0.0
    cash: float = 0.0
    # Kept for RL/tuning compatibility; no longer used by demand_target.
    activation_tick: dict[str, int] = field(default_factory=dict)

    @property
    def _node_type(self) -> str:
        return "demand_sink"

    def demand_target(
        self,
        tick: int,
        market: "Market",
        catalog: "list[Ware]",
        world_rng: Random,
    ) -> int:
        """Compute the demand target for this tick.

        **Each sink is bound to exactly one product** (``self.product_id``), the
        same way a factory produces exactly one product. This method returns the
        demand only for that single bound product — a sink never demands more
        than one SKU.

        Why, then, the loop over the *whole* catalog? It is purely a
        Common-Random-Numbers (CRN) device, not a sign of multi-product demand.
        The simulator shares a single ``world_rng`` stream across all sinks, and
        a single stream is position-sensitive: how many draws one sink takes
        shifts the stream position for every sink after it. CRN paired
        evaluation (RL / tuning) relies on two runs of the *same scenario under
        different policies* seeing *identical* random shocks, so the only
        measured difference is the policy. For that to hold, the number of draws
        each sink consumes must be constant — a pure function of
        ``(catalog size, tick)`` — and must **not** depend on which product the
        sink is bound to or which products happen to be active. So we draw one
        sample per catalog product in catalog declaration order (ADR 0003),
        keep only the draw for ``self.product_id``, and discard the rest; the
        discarded draws exist solely to advance ``world_rng`` by a fixed amount
        and keep paired sinks aligned.

        Multiplier composition for each catalog pid::

            demand_target = demand_dist.sample(world_rng)
                            * market.demand_multiplier(pid, region)

        Only the value for ``self.product_id`` is returned as the integer
        demand target; the draws for all other pids advance ``world_rng`` to
        preserve CRN alignment across paired sinks.

        Note: this whole-catalog loop is inherited from the old ``Store`` model,
        where a store's *policy* could activate/deactivate SKUs and so vary its
        own draw count. A sink's product binding is fixed by the scenario and is
        never policy-dependent, so a cleaner, more robust design is a dedicated
        per-stream RNG seeded from ``(world_seed, product_id)`` — one draw, no
        loop, CRN by construction. See TODO.md §6 for the planned migration.

        Parameters
        ----------
        tick:
            Current simulation tick (0-based counter from ``market.current_step()``).
        market:
            Live ``Market`` instance; ``demand_multiplier`` must not draw
            from ``world_rng`` (it is a pure multiplier — see ADR 0015).
        catalog:
            The scenario catalog list. Iterated in declaration order to
            preserve the CRN draw sequence (ADR 0003).
        world_rng:
            Shared world RNG.  Exactly ``len(catalog)`` draws are consumed
            per call.

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

            # One demand sample per catalog pid — load-bearing CRN draw.
            # ``Uniform`` / ``Normal`` each consume exactly one world_rng.random()
            # call. ``Constant`` skips the draw; callers requiring strict CRN
            # alignment must use a draw-consuming distribution.
            if self.demand_dist is not None:
                base = float(self.demand_dist.sample(world_rng))
            else:
                base = 0.0

            raw = base * market_mult
            
            # Select only the demand for the product that the sink is bound to.
            if pid == self.product_id:
                result_for_product = max(0, int(raw))

        return result_for_product
