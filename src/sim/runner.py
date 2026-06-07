"""Graph-engine runner — the sole simulation engine.

Public surface
--------------
- ``build_world(scenario, *, policy_overrides=None) -> Simulation``
  Construct a ``Simulation`` bundle from a graph-mode ``Scenario``.
- ``Simulation``
  Mutable world bundle; call ``Simulation.tick()`` each step.
- ``Runner(scenario, *, policy_overrides=None)``
  Drive a ``Scenario`` to a full run log via ``Runner.run() -> dict``.
- ``TickResult``
  Frozen data carrier (kept for import compatibility; graph engine
  does not return TickResult from tick() — the value is ``None``).

Tick structure (ADR 0018, supersedes ADR 0014):
  tick_world → publish_offers → demand-pull walk (reverse-topological order,
  sinks first; factories last; each ready-set shuffled via allocation_rng) →
  produce → deliver (EventEngine callbacks) → consume_demand_sinks

The demand-pull schedule structure is computed once in ``build_world`` and
cached on ``Simulation.schedule``; only the per-tick ready-set shuffle
re-draws ``allocation_rng``.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any

from src.sim.event_engine import EventEngine, WorldEvent
from src.sim.market import Market
from src.sim.scenario import Scenario


# ---------------------------------------------------------------------------
# TickResult — kept for import compatibility
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TickResult:
    """Frozen data carrier retained for import compatibility.

    The graph engine's ``Simulation.tick()`` does not return a
    ``TickResult``; this class exists solely so that code importing
    it from this module does not break.
    """

    actions: dict[int, dict[str, Any]]
    demand_traces: dict[int, dict[str, int]]
    active_events: list[WorldEvent]


# ---------------------------------------------------------------------------
# Simulation (was GraphSimulation) — mutable world bundle
# ---------------------------------------------------------------------------

class Simulation:
    """Mutable world bundle for the multi-echelon graph engine.

    Construct via :func:`build_world`.

    Attributes
    ----------
    scenario       — the frozen ``Scenario`` used to build this bundle.
    world_rng      — shared world RNG (market/event draws).
    allocation_rng — per-phase buyer-shuffle RNG (ADR 0016).
    market         — regional demand/supply environment.
    event_engine   — disruption events + order-delivery callbacks.
    graph          — validated topology (``Graph`` instance).
    nodes          — ``{node_id: Node}`` mapping for fast lookup.
    levels         — ``{node_id: int}`` echelon levels (display-only; ADR 0018).
    schedule       — demand-pull ready-sets; list[list[node_id]] computed once
                     at build_world and cached (ADR 0018).
    """

    def __init__(
        self,
        scenario: Scenario,
        world_rng: Random,
        allocation_rng: Random,
        market: Market,
        event_engine: EventEngine,
        graph: Any,
        nodes: dict,
        levels: dict,
        schedule: list,
    ) -> None:
        self.scenario = scenario
        self.world_rng = world_rng
        self.allocation_rng = allocation_rng
        self.market = market
        self.event_engine = event_engine
        self.graph = graph
        self.nodes = nodes
        # levels is retained for display-only use (inspect.py); no longer
        # drives scheduling (ADR 0018 — level-bucket cascade removed).
        self.levels = levels
        # Demand-pull schedule: list of ready-sets in reverse-topological order.
        # Computed once at build_world; only the per-tick shuffle re-draws.
        self.schedule = schedule
        # Removed in issue 01; kept as None so downstream callers that
        # guard ``if registry is not None`` still work without modification.
        self.item_registry = None
        # Per-tick order-quantity accumulator: ``{buyer_id: {pid: qty}}``.
        # Populated during ``tick()`` and consumed by the runner's snapshot.
        self._last_tick_orders: dict[str, dict[str, int]] = {}
        # Current-tick sales accumulator for intermediate nodes: ``{node_id: {pid: qty}}``.
        # Populated during the demand-pull walk (after each sink/intermediate buys).
        # Injected into each intermediate's observation as ``observed_sales`` so
        # the policy sees the current tick's complete demand signal (not lagged).
        self._tick_sales: dict[str, dict[str, int]] = {}
        # Stashed central table from ``tick_world()`` for use by ``tick_decide_and_settle()``.
        self._current_table: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Run one full tick of the graph cascade (ADR 0018).

        Steps
        -----
        1. ``tick_world``        — advance market, events.
        2. ``publish_offers``    — all sellers publish live offers.
        3. Demand-pull walk      — reverse-topological order; sinks first,
                                   factories last; each ready-set shuffled
                                   by ``allocation_rng`` (ADR 0016).
        4. ``produce``           — factories produce up to capacity.
        5. ``deliver``           — already scheduled by EventEngine; no-op
                                   here (callbacks fired in tick_world).
        6. ``consume_demand_sinks`` — credit income_rate to each sink.
        """
        from src.sim.central_table import CentralTable, Offer
        from src.sim.node import FactoryNode, IntermediateNode

        # -------------------------------------------------------------------
        # 1. tick_world — advance shared world state.
        # -------------------------------------------------------------------
        self.market.tick()
        self.event_engine.tick(self.market)
        current_tick = self.market.current_step()

        # -------------------------------------------------------------------
        # 2. publish_offers — all sellers post current inventory to the table.
        # -------------------------------------------------------------------
        table = _publish_offers(self.nodes, CentralTable, Offer, FactoryNode, IntermediateNode)

        # -------------------------------------------------------------------
        # 3. Demand-pull walk.
        # -------------------------------------------------------------------
        self._last_tick_orders = {}
        self._tick_sales = {}
        _run_demand_pull_schedule(self, table, current_tick)

        # -------------------------------------------------------------------
        # 4. produce — factories run policy and produce up to capacity.
        # -------------------------------------------------------------------
        _produce_factories(self, current_tick)

        # -------------------------------------------------------------------
        # 5. deliver — EventEngine already fired delivery callbacks in
        #    tick_world (step 1 above) so no additional work here.
        # -------------------------------------------------------------------

        # -------------------------------------------------------------------
        # 6. consume_demand_sinks — credit income_rate cash to each sink.
        # -------------------------------------------------------------------
        _credit_sinks(self.nodes)

    # ------------------------------------------------------------------
    # Two-phase tick API — for RL interoperability
    # ------------------------------------------------------------------

    def tick_world(self) -> Any:
        """Phase 1: advance world state (market, events).

        Also publishes offers to a fresh ``CentralTable`` and stores
        it on ``self._current_table`` for ``tick_decide_and_settle()``
        and for observation encoders that need central-table data.

        Returns the current tick (int) so callers can track step count.
        """
        from src.sim.central_table import CentralTable, Offer
        from src.sim.node import FactoryNode, IntermediateNode

        self.market.tick()
        self.event_engine.tick(self.market)
        current_tick = self.market.current_step()

        table = _publish_offers(self.nodes, CentralTable, Offer, FactoryNode, IntermediateNode)

        # Stash the table so the encoder can snapshot it and
        # tick_decide_and_settle can reuse it.
        self._current_table = table
        return current_tick

    def tick_decide_and_settle(self, current_tick: int | None = None) -> None:
        """Phase 2: run the demand-pull walk, produce, and settle sinks.

        Must be called after ``tick_world()``.  Uses ``self._current_table``
        as the live central table (published by ``tick_world()``).

        Parameters
        ----------
        current_tick:
            If ``None``, reads from ``self.market.current_step()``.
        """
        if current_tick is None:
            current_tick = self.market.current_step()

        table = getattr(self, "_current_table", None)
        if table is None:
            from src.sim.central_table import CentralTable
            table = CentralTable()

        self._last_tick_orders = {}
        self._tick_sales = {}
        _run_demand_pull_schedule(self, table, current_tick)
        _produce_factories(self, current_tick)
        _credit_sinks(self.nodes)

        # Clean up the stashed table.
        self._current_table = None


# ---------------------------------------------------------------------------
# Internal helpers — shared by tick() and tick_decide_and_settle()
# ---------------------------------------------------------------------------

def _publish_offers(nodes: dict, CentralTable: Any, Offer: Any, FactoryNode: Any, IntermediateNode: Any) -> Any:
    """Publish all seller offers to a fresh CentralTable and return it."""
    table = CentralTable()
    for node_id, node in nodes.items():
        if isinstance(node, FactoryNode):
            pid = node.produces_product_id
            table.publish(
                node_id,
                pid,
                Offer(
                    available_qty=node.inventory,
                    list_price=node.list_price,
                    min_order=0,
                ),
            )
        elif isinstance(node, IntermediateNode):
            for pid in node.carried_products:
                qty = node.inventory.get(pid, 0)
                price = node.list_prices.get(pid, 0.0)
                min_order = node.min_order_imposed.get(pid, 0)
                table.publish(
                    node_id,
                    pid,
                    Offer(
                        available_qty=qty,
                        list_price=price,
                        min_order=min_order,
                    ),
                )
    return table


def _run_demand_pull_schedule(sim: "Simulation", table: Any, current_tick: int) -> None:
    """Execute the demand-pull walk for one tick.

    Walks ``sim.schedule`` (pre-computed reverse-topological ready-sets) in
    order — sinks first, factories last — shuffling each ready-set via
    ``sim.allocation_rng`` (ADR 0016).  For each buyer node: observe →
    decide → ``execute_buy`` per line.  Accumulates current-tick sales into
    ``sim._tick_sales`` so intermediates see an un-lagged demand signal when
    they are processed.

    Factories are not buyers and are skipped here; ``_produce_factories``
    handles them separately.
    """
    from src.sim.allocation import execute_buy, shuffle_buyers
    from src.sim.node import DemandSinkNode, IntermediateNode, FactoryNode
    from src.sim.observation import (
        build_factory_obs,
        build_intermediate_obs,
        build_sink_obs,
    )

    for ready_set in sim.schedule:
        # Shuffle the ready-set deterministically (ADR 0016).
        buyers = [sim.nodes[nid] for nid in ready_set if nid in sim.nodes]
        buyers = shuffle_buyers(buyers, sim.allocation_rng)

        for buyer in buyers:
            if isinstance(buyer, FactoryNode):
                # Factories are sources, not buyers; handled by _produce_factories.
                continue

            if isinstance(buyer, DemandSinkNode):
                # Samples world_rng for every catalog pid — CRN invariant (ADR 0003).
                demand_target = float(
                    buyer.demand_target(
                        tick=current_tick,
                        market=sim.market,
                        catalog=sim.scenario.catalog,
                        world_rng=sim.world_rng,
                    )
                )
                obs = build_sink_obs(buyer, tick=current_tick, demand_target=demand_target)
                direct_supplier_ids = sim.graph.suppliers_of(buyer.id)
                obs["direct_supplier_ids"] = direct_supplier_ids
                if buyer.policy is not None:
                    action = buyer.policy.decide(obs, table)
                else:
                    action = _default_sink_action(
                        buyer, demand_target, table,
                        allowed_supplier_ids=direct_supplier_ids,
                    )

                for supplier_id, qty in action.get("buy", []):
                    if qty <= 0:
                        continue
                    if supplier_id not in direct_supplier_ids:
                        continue
                    supplier = sim.nodes.get(supplier_id)
                    if supplier is None:
                        continue
                    lt = sim.graph.lead_time(supplier_id, buyer.id, buyer.product_id)
                    result = execute_buy(
                        buyer=buyer,
                        supplier=supplier,
                        pid=buyer.product_id,
                        qty_requested=qty,
                        table=table,
                        event_engine=sim.event_engine,
                        current_tick=current_tick,
                        lead_time=lt,
                    )
                    # Accumulate sales for the supplier (if intermediate) so that
                    # when it is processed later in the walk, its observed_sales
                    # reflects the complete current-tick demand.
                    if isinstance(supplier, IntermediateNode) and result.qty_filled > 0:
                        sup_sales = sim._tick_sales.setdefault(supplier_id, {})
                        pid_sold = buyer.product_id
                        sup_sales[pid_sold] = sup_sales.get(pid_sold, 0) + result.qty_filled

            elif isinstance(buyer, IntermediateNode):
                obs = build_intermediate_obs(buyer, tick=current_tick)
                # Inject current-tick sales (complete by demand-pull ordering):
                # all downstream buyers of this intermediate have already been
                # processed before this node is reached.
                obs["observed_sales"] = sim._tick_sales.get(buyer.id, {})
                direct_supplier_ids = sim.graph.suppliers_of(buyer.id)
                obs["direct_supplier_ids"] = direct_supplier_ids
                if buyer.policy is not None:
                    action = buyer.policy.decide(obs, table)
                else:
                    action = {"order": {}, "list_price": {}, "min_order_imposed": {}}

                orders = action.get("order", {})
                buyer_orders = sim._last_tick_orders.setdefault(buyer.id, {})
                for pid, order_lines in orders.items():
                    for supplier_id, qty in order_lines:
                        if qty <= 0:
                            continue
                        if supplier_id not in direct_supplier_ids:
                            continue
                        supplier = sim.nodes.get(supplier_id)
                        if supplier is None:
                            continue
                        lt = sim.graph.lead_time(supplier_id, buyer.id, pid)
                        execute_buy(
                            buyer=buyer,
                            supplier=supplier,
                            pid=pid,
                            qty_requested=qty,
                            table=table,
                            event_engine=sim.event_engine,
                            current_tick=current_tick,
                            lead_time=lt,
                        )
                        buyer_orders[pid] = buyer_orders.get(pid, 0) + qty


def _produce_factories(sim: "Simulation", current_tick: int) -> None:
    """Run factory produce step for all FactoryNodes."""
    from src.sim.node import FactoryNode
    from src.sim.observation import build_factory_obs

    for node_id, node in sim.nodes.items():
        if isinstance(node, FactoryNode):
            obs = build_factory_obs(node, tick=current_tick)
            if node.policy is not None:
                action = node.policy.decide(obs)
            else:
                from src.sim.distributions import Distribution
                cap = node.capacity_per_tick
                if isinstance(cap, Distribution):
                    produce_qty = int(cap.sample(sim.world_rng))
                else:
                    produce_qty = int(cap)
                action = {"produce_qty": produce_qty, "list_price": node.unit_cost}

            produce_qty = int(action.get("produce_qty", 0))
            node.inventory += produce_qty
            # ADR 0013 rule 3: production absorbs cash at unit_cost
            node.cash -= node.unit_cost * produce_qty
            new_price = action.get("list_price")
            if new_price is not None:
                node.list_price = float(new_price)


def _credit_sinks(nodes: dict) -> None:
    """Credit income_rate cash to each DemandSinkNode."""
    from src.sim.node import DemandSinkNode
    for node_id, node in nodes.items():
        if isinstance(node, DemandSinkNode):
            node.cash += node.income_rate


def _default_sink_action(
    sink: Any,
    demand_target: float,
    table: Any,
    *,
    allowed_supplier_ids: set | None = None,
) -> dict:
    """Default greedy action for a DemandSinkNode without an attached policy.

    Buys from the cheapest available direct supplier that has stock, up to
    ``demand_target`` units.  Returns ``{"buy": [(supplier_id, qty)]}``.
    """
    pid = sink.product_id
    offers = table.snapshot_for_buyer(pid)
    if not offers:
        return {"buy": []}

    # Filter to direct suppliers only.
    if allowed_supplier_ids is not None:
        offers = [(sid, o) for sid, o in offers if sid in allowed_supplier_ids]

    if not offers:
        return {"buy": []}

    # Sort by price ascending, use cheapest first.
    sorted_offers = sorted(offers, key=lambda t: t[1].list_price)

    remaining = int(demand_target)
    buys: list[tuple[str, int]] = []
    for supplier_id, offer in sorted_offers:
        if remaining <= 0:
            break
        if offer.available_qty <= 0:
            continue
        # Min-order check: skip this supplier if we can't meet its min_order.
        if offer.min_order > 0 and remaining < offer.min_order:
            continue
        # Cash check.
        affordable_qty = (
            int(sink.cash / offer.list_price)
            if offer.list_price > 0
            else offer.available_qty
        )
        qty = min(remaining, offer.available_qty, affordable_qty)
        if qty > 0:
            buys.append((supplier_id, qty))
            remaining -= qty

    return {"buy": buys}


# ---------------------------------------------------------------------------
# build_world — module-level factory
# ---------------------------------------------------------------------------

def build_world(
    scenario: Scenario,
    *,
    policy_overrides: dict | None = None,
) -> Simulation:
    """Construct and return a ``Simulation`` bundle from *scenario*.

    *scenario* must have ``is_graph == True`` (i.e. at least one
    ``NodeInstance`` in ``scenario.nodes``).

    Parameters
    ----------
    scenario:
        A graph-mode ``Scenario`` with ``nodes`` and ``edges`` populated.
    policy_overrides:
        Optional ``{node_id: NodePolicy}`` mapping.  When supplied, the
        node with the matching id receives the override policy in place of
        ``NodeInstance.policy``.

    Returns
    -------
    Simulation
        A fully-initialised bundle ready for ``Simulation.tick()``.
    """
    from src.sim.episode_sampler import _derive_seed
    from src.sim.graph import EdgeSpec, build_graph, build_demand_pull_schedule, compute_levels

    if not scenario.nodes:
        raise ValueError(
            "build_world requires a graph-mode scenario with at least one node "
            "(scenario.nodes must be non-empty)"
        )

    # Seeding (ADR 0016): world_rng from world_seed; allocation_rng from the
    # "allocation" sub-seed derived via _derive_seed.
    world_rng: Random = Random(scenario.world_seed)
    allocation_seed = _derive_seed(scenario.world_seed, "allocation")
    allocation_rng: Random = Random(allocation_seed)

    # Shared world objects.
    market = Market(
        scenario.market,
        world_rng,
        scenario.start_date,
        catalog=scenario.catalog,
    )
    event_engine = EventEngine(scenario.disruption, world_rng)

    # Build the Graph topology.
    node_ids = [ni.node.id for ni in scenario.nodes]
    edges: list[EdgeSpec] = list(scenario.edges)
    node_types: dict[str, str] = {ni.node.id: ni.node._node_type for ni in scenario.nodes}
    graph = build_graph(node_ids, edges, node_types=node_types)

    # Compute echelon levels (display-only; ADR 0018).
    levels = compute_levels(graph)

    # Compute and cache the demand-pull schedule structure (ADR 0018).
    # Only the per-tick ready-set shuffle re-draws allocation_rng.
    schedule = build_demand_pull_schedule(graph)

    # Attach policies and build the node lookup dict.
    # We deep-copy each node so that multiple build_world calls on the same
    # Scenario don't share mutable node state between simulation runs.
    import copy
    overrides = policy_overrides or {}
    nodes: dict[str, Any] = {}
    for ni in scenario.nodes:
        node = copy.deepcopy(ni.node)
        # Resolve the policy with a three-level precedence: an explicit
        # override wins, else ``NodeInstance.policy``, else the policy
        # already attached to the node object (``node.policy``). The last
        # fallback keeps scenarios that set ``node.policy = ...`` directly
        # (instead of passing ``NodeInstance(policy=...)``) working —
        # otherwise their shops would be left policy-less and never order.
        policy = overrides.get(node.id, ni.policy if ni.policy is not None else node.policy)
        node.policy = policy
        # Assign the computed echelon level onto the node (display-only).
        node.level = levels.get(node.id)
        nodes[node.id] = node

    return Simulation(
        scenario=scenario,
        world_rng=world_rng,
        allocation_rng=allocation_rng,
        market=market,
        event_engine=event_engine,
        graph=graph,
        nodes=nodes,
        levels=levels,
        schedule=schedule,
    )


# ---------------------------------------------------------------------------
# Runner (was GraphRunner) — drive a scenario for n_steps ticks
# ---------------------------------------------------------------------------

class Runner:
    """Drive a graph-mode ``Scenario`` for ``n_steps`` ticks.

    Wraps ``build_world`` + ``Simulation.tick()`` into a simple
    ``run() -> dict`` interface.

    The run log records per-tick node cash balances and total inventory
    to support cash-conservation and no-negative-inventory assertions.

    Usage::

        runner = Runner(scenario)
        log = runner.run()

    Parameters
    ----------
    scenario:
        A graph-mode ``Scenario`` (``scenario.is_graph`` must be True).
    policy_overrides:
        Optional ``{node_id: NodePolicy}`` mapping forwarded to
        ``build_world``.
    """

    def __init__(
        self,
        scenario: Scenario,
        *,
        policy_overrides: dict | None = None,
    ) -> None:
        self.scenario = scenario
        self._sim = build_world(scenario, policy_overrides=policy_overrides)

    @property
    def nodes(self) -> dict:
        """Expose the node lookup dict for post-run inspection."""
        return self._sim.nodes

    def run(self) -> dict[str, Any]:
        """Execute the full simulation and return a run log.

        The run log is compatible with ``DataExporter``:

        Returns
        -------
        dict
            Keys:

            ``"n_steps"``
                Number of ticks run.
            ``"ticks"``
                List of per-tick log dicts, each with ``"tick"``,
                ``"node_cash"``, ``"node_inventory"``.
            ``"global"``
                Shared world-state time series:
                ``"time"`` (``simulation_step``, ``simulation_date``),
                ``"market_supply"`` and ``"market_demand"`` per region
                (length ``n_steps + 1``), and ``"events"``.
        """
        # Capture step-0 snapshot before any tick.
        step_series: list[int] = [0]
        date_series: list[Any] = [self._sim.market.current_date()]
        market_supply_series: dict[str, list[float]] = {
            r: [float(self._sim.market.market_state[r]["market_supply"])]
            for r in self._sim.market.regions
        }
        market_demand_series: dict[str, list[float]] = {
            r: [float(self._sim.market.market_state[r]["market_demand"])]
            for r in self._sim.market.regions
        }

        ticks: list[dict[str, Any]] = []
        for _ in range(self.scenario.n_steps):
            self._sim.tick()
            tick_log = self._snapshot_tick()
            ticks.append(tick_log)
            # Append per-tick world state.
            step_series.append(self._sim.market.current_step())
            date_series.append(self._sim.market.current_date())
            for r in self._sim.market.regions:
                market_supply_series[r].append(
                    float(self._sim.market.market_state[r]["market_supply"])
                )
                market_demand_series[r].append(
                    float(self._sim.market.market_state[r]["market_demand"])
                )

        return {
            "n_steps": self.scenario.n_steps,
            "ticks": ticks,
            "global": {
                "time": {
                    "simulation_step": step_series,
                    "simulation_date": date_series,
                },
                "market_supply": market_supply_series,
                "market_demand": market_demand_series,
                "events": {"occurrences": [None] * self.scenario.n_steps},
            },
        }

    def _snapshot_tick(self) -> dict[str, Any]:
        """Capture a per-tick state snapshot.

        For ``IntermediateNode``s, also captures the pending (in-transit)
        inventory by product so callers can compute outstanding-order positions.
        """
        from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode

        node_cash: dict[str, float] = {}
        node_inventory: dict[str, Any] = {}
        node_pending: dict[str, Any] = {}

        for node_id, node in self._sim.nodes.items():
            if isinstance(node, DemandSinkNode):
                node_cash[node_id] = node.cash
                node_inventory[node_id] = {}
                node_pending[node_id] = {}
            elif isinstance(node, FactoryNode):
                node_cash[node_id] = getattr(node, "cash", 0.0)
                node_inventory[node_id] = {"_total": node.inventory}
                node_pending[node_id] = {}
            elif isinstance(node, IntermediateNode):
                node_cash[node_id] = getattr(node, "cash", 0.0)
                node_inventory[node_id] = dict(node.inventory)
                # Sum pending across all suppliers for each product.
                pending_totals: dict[str, int] = {}
                for sup_pending in node.pending.values():
                    for pid, qty in sup_pending.items():
                        pending_totals[pid] = pending_totals.get(pid, 0) + qty
                node_pending[node_id] = pending_totals

        return {
            "tick": self._sim.market.current_step(),
            "node_cash": node_cash,
            "node_inventory": node_inventory,
            "node_pending": node_pending,
            "node_orders": dict(self._sim._last_tick_orders),
        }


__all__ = [
    "Runner",
    "Simulation",
    "TickResult",
    "build_world",
]
