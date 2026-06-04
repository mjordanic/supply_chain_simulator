"""Graph-engine runner — the sole simulation engine post-Phase-4.

This module provides the graph-based simulation engine. The legacy
Store-engine classes (``Simulation``, ``Runner``, ``build_world``) have
been retired; these names now refer exclusively to the graph engine.

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

Renamed aliases kept for scenario-authoring compatibility:
- ``GraphSimulation`` → ``Simulation``
- ``GraphRunner``     → ``Runner``
- ``build_graph_world`` → ``build_world``

Tick structure (ADR 0014):
  tick_world → publish_offers → for p in 1..max_level: shuffle buyers →
  per buyer observe → decide → execute_buy per line → produce →
  deliver (EventEngine callbacks) → consume_demand_sinks
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any

from src.sim.event_engine import EventEngine, WorldEvent
from src.sim.item_registry import ItemRegistry
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
    world_rng      — shared world RNG (market/event/lifecycle draws).
    allocation_rng — per-phase buyer-shuffle RNG (ADR 0016).
    item_registry  — catalog + per-item lifecycle state.
    market         — regional demand/supply environment.
    event_engine   — disruption events + order-delivery callbacks.
    graph          — validated topology (``Graph`` instance).
    nodes          — ``{node_id: Node}`` mapping for fast lookup.
    levels         — ``{node_id: int}`` echelon levels.
    """

    def __init__(
        self,
        scenario: Scenario,
        world_rng: Random,
        allocation_rng: Random,
        item_registry: ItemRegistry,
        market: Market,
        event_engine: EventEngine,
        graph: Any,
        nodes: dict,
        levels: dict,
    ) -> None:
        self.scenario = scenario
        self.world_rng = world_rng
        self.allocation_rng = allocation_rng
        self.item_registry = item_registry
        self.market = market
        self.event_engine = event_engine
        self.graph = graph
        self.nodes = nodes
        self.levels = levels
        # Per-tick order-quantity accumulator: ``{buyer_id: {pid: qty}}``.
        # Populated during ``tick()`` and consumed by the runner's snapshot.
        self._last_tick_orders: dict[str, dict[str, int]] = {}
        # Per-tick sales accumulator for intermediate nodes: ``{node_id: {pid: qty}}``.
        # Tracks how many units each IntermediateNode sold to downstream buyers
        # in the previous tick.  Passed into the intermediate observation so
        # the policy's rate estimator sees actual demand, not just inventory
        # deltas (which are confused by simultaneous deliveries).
        self._last_tick_sales: dict[str, dict[str, int]] = {}
        # Stashed central table from ``tick_world()`` for use by ``tick_decide_and_settle()``.
        self._current_table: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Run one full tick of the graph cascade (ADR 0014).

        Steps
        -----
        1. ``tick_world``        — advance market, events, lifecycle.
        2. ``publish_offers``    — all sellers publish live offers.
        3. Phase cascade         — for each level p from 1 to max_level:
                                   shuffle buyers, then per buyer
                                   observe → decide → execute_buy per line.
        4. ``produce``           — factories produce up to capacity.
        5. ``deliver``           — already scheduled by EventEngine; no-op
                                   here (callbacks fired in tick_world).
        6. ``consume_demand_sinks`` — credit income_rate to each sink.
        """
        from src.sim.allocation import execute_buy, shuffle_buyers
        from src.sim.central_table import CentralTable, Offer
        from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode
        from src.sim.observation import (
            build_factory_obs,
            build_intermediate_obs,
            build_sink_obs,
        )

        # -------------------------------------------------------------------
        # 1. tick_world — advance shared world state.
        # -------------------------------------------------------------------
        self.market.tick()
        self.event_engine.tick(self.market)
        self.item_registry.tick()

        current_tick = self.market.current_step()

        # -------------------------------------------------------------------
        # 2. publish_offers — all sellers post current inventory to the table.
        # -------------------------------------------------------------------
        table = CentralTable()
        for node_id, node in self.nodes.items():
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

        # -------------------------------------------------------------------
        # 3. Phase cascade — levels 1 .. max_level.
        # -------------------------------------------------------------------
        # Reset per-tick accumulators (orders placed by intermediate nodes and
        # sales made by intermediate nodes to downstream buyers).
        self._last_tick_orders = {}
        # Capture the completed-tick sales before resetting so the current
        # tick's intermediate nodes receive the *previous* tick's sales in
        # their observation.  This is intentional: the policy observes what
        # was sold in the tick that just ended, not what will be sold now.
        prev_tick_sales = self._last_tick_sales
        self._last_tick_sales = {}

        if self.levels:
            max_level = max(self.levels.values())
        else:
            max_level = 0

        for p in range(1, max_level + 1):
            # Collect all buyers at this echelon level.
            buyers_at_level = [
                node
                for node_id, node in self.nodes.items()
                if self.levels.get(node_id, 0) == p
            ]
            if not buyers_at_level:
                continue

            # Shuffle buyers deterministically (ADR 0016).
            buyers_at_level = shuffle_buyers(buyers_at_level, self.allocation_rng)

            for buyer in buyers_at_level:
                if isinstance(buyer, DemandSinkNode):
                    # Full lifecycle/freshness composition via demand_target
                    # (ADR 0015, issue 10). Samples world_rng for every catalog
                    # pid in registry iteration order — CRN invariant preserved.
                    demand_target = float(
                        buyer.demand_target(
                            tick=current_tick,
                            market=self.market,
                            registry=self.item_registry,
                            world_rng=self.world_rng,
                        )
                    )
                    obs = build_sink_obs(
                        buyer, tick=current_tick, demand_target=demand_target
                    )
                    # Compute the set of direct suppliers for this buyer.
                    direct_supplier_ids = self.graph.suppliers_of(buyer.id)
                    # Inject the direct suppliers so an attached policy can
                    # restrict its buys to them — without this a sink whose
                    # product is also published by an *indirect* seller (e.g.
                    # the factory two echelons up) would buy from that seller
                    # and bypass its own shop, leaving the shop's demand
                    # signal at zero so it never reorders.
                    obs["direct_supplier_ids"] = direct_supplier_ids
                    if buyer.policy is not None:
                        action = buyer.policy.decide(obs, table)
                    else:
                        # Default greedy: buy from cheapest available direct supplier.
                        action = _default_sink_action(
                            buyer, demand_target, table,
                            allowed_supplier_ids=direct_supplier_ids,
                        )

                    # Execute each buy line.
                    for supplier_id, qty in action.get("buy", []):
                        if qty <= 0:
                            continue
                        # Safety: only buy from direct suppliers.
                        if supplier_id not in direct_supplier_ids:
                            continue
                        supplier = self.nodes.get(supplier_id)
                        if supplier is None:
                            continue
                        lt = self.graph.lead_time(supplier_id, buyer.id, buyer.product_id)
                        result = execute_buy(
                            buyer=buyer,
                            supplier=supplier,
                            pid=buyer.product_id,
                            qty_requested=qty,
                            table=table,
                            event_engine=self.event_engine,
                            current_tick=current_tick,
                            lead_time=lt,
                        )
                        # Track sales for IntermediateNode suppliers so the
                        # next-tick intermediate observation carries actual
                        # demand signal for the policy's rate estimator.
                        if isinstance(supplier, IntermediateNode) and result.qty_filled > 0:
                            sup_sales = self._last_tick_sales.setdefault(supplier_id, {})
                            pid_sold = buyer.product_id
                            sup_sales[pid_sold] = sup_sales.get(pid_sold, 0) + result.qty_filled

                elif isinstance(buyer, IntermediateNode):
                    obs = build_intermediate_obs(buyer, tick=current_tick)
                    # Inject the previous-tick sales so the policy's rate
                    # estimator receives an accurate demand signal even when
                    # deliveries and demand occur in the same tick (which
                    # would make the inventory-delta proxy misleading).
                    obs["prev_tick_sales"] = prev_tick_sales.get(buyer.id, {})
                    direct_supplier_ids = self.graph.suppliers_of(buyer.id)
                    # Inject the buyer's direct upstream suppliers so the policy
                    # routes reorders to them, not to its own published offers
                    # (a shop publishes its carried inventory for downstream
                    # sinks, so the central table would otherwise list the shop
                    # itself as a "supplier" and the lines below would drop it).
                    obs["direct_supplier_ids"] = direct_supplier_ids
                    if buyer.policy is not None:
                        action = buyer.policy.decide(obs, table)
                    else:
                        action = {"order": {}, "list_price": {}, "min_order_imposed": {}}

                    orders = action.get("order", {})
                    buyer_orders = self._last_tick_orders.setdefault(buyer.id, {})
                    for pid, order_lines in orders.items():
                        for supplier_id, qty in order_lines:
                            if qty <= 0:
                                continue
                            # Safety: only buy from direct suppliers.
                            if supplier_id not in direct_supplier_ids:
                                continue
                            supplier = self.nodes.get(supplier_id)
                            if supplier is None:
                                continue
                            lt = self.graph.lead_time(supplier_id, buyer.id, pid)
                            execute_buy(
                                buyer=buyer,
                                supplier=supplier,
                                pid=pid,
                                qty_requested=qty,
                                table=table,
                                event_engine=self.event_engine,
                                current_tick=current_tick,
                                lead_time=lt,
                            )
                            # Accumulate effective order quantity for this tick.
                            buyer_orders[pid] = buyer_orders.get(pid, 0) + qty

        # -------------------------------------------------------------------
        # 4. produce — factories run policy and produce up to capacity.
        # -------------------------------------------------------------------
        for node_id, node in self.nodes.items():
            if isinstance(node, FactoryNode):
                obs = build_factory_obs(node, tick=current_tick)
                if node.policy is not None:
                    action = node.policy.decide(obs)
                else:
                    # Default: produce up to capacity.
                    from src.sim.distributions import Distribution
                    cap = node.capacity_per_tick
                    if isinstance(cap, Distribution):
                        produce_qty = int(cap.sample(self.world_rng))
                    else:
                        produce_qty = int(cap)
                    action = {"produce_qty": produce_qty, "list_price": node.unit_cost}

                produce_qty = int(action.get("produce_qty", 0))
                node.inventory += produce_qty
                # ADR 0013 rule 3: production absorbs cash at unit_cost
                # (factories are zero-margin). The factory recovers exactly
                # this when it later sells at list_price == unit_cost, so the
                # produce-then-sell cycle nets to zero. Without this leg the
                # factory accumulated phantom cash from sale revenue alone.
                node.cash -= node.unit_cost * produce_qty
                # Update list price if provided.
                new_price = action.get("list_price")
                if new_price is not None:
                    node.list_price = float(new_price)

        # -------------------------------------------------------------------
        # 5. deliver — EventEngine already fired delivery callbacks in
        #    tick_world (step 1 above) so no additional work here.
        # -------------------------------------------------------------------

        # -------------------------------------------------------------------
        # 6. consume_demand_sinks — credit income_rate cash to each sink.
        # -------------------------------------------------------------------
        for node_id, node in self.nodes.items():
            if isinstance(node, DemandSinkNode):
                node.cash += node.income_rate

    # ------------------------------------------------------------------
    # Two-phase tick API — for RL interoperability
    # ------------------------------------------------------------------

    def tick_world(self) -> Any:
        """Phase 1: advance world state (market, events, lifecycle).

        Also publishes offers to a fresh ``CentralTable`` and stores
        it on ``self._current_table`` for ``tick_decide_and_settle()``
        and for observation encoders that need central-table data.

        Returns the current tick (int) so callers can track step count.
        """
        from src.sim.central_table import CentralTable, Offer
        from src.sim.node import FactoryNode, IntermediateNode

        self.market.tick()
        self.event_engine.tick(self.market)
        self.item_registry.tick()

        current_tick = self.market.current_step()

        # publish_offers — all sellers post current inventory to the table.
        table = CentralTable()
        for node_id, node in self.nodes.items():
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

        # Stash the table so the encoder can snapshot it and
        # tick_decide_and_settle can reuse it.
        self._current_table = table
        return current_tick

    def tick_decide_and_settle(self, current_tick: int | None = None) -> None:
        """Phase 2: run the phase cascade, produce, and settle sinks.

        Must be called after ``tick_world()``.  Uses ``self._current_table``
        as the live central table (published by ``tick_world()``).

        Parameters
        ----------
        current_tick:
            If ``None``, reads from ``self.market.current_step()``.
        """
        from src.sim.allocation import execute_buy, shuffle_buyers
        from src.sim.central_table import CentralTable, Offer
        from src.sim.node import DemandSinkNode, FactoryNode, IntermediateNode
        from src.sim.observation import (
            build_factory_obs,
            build_intermediate_obs,
            build_sink_obs,
        )

        if current_tick is None:
            current_tick = self.market.current_step()

        table = getattr(self, "_current_table", None)
        if table is None:
            # Fallback: build a fresh table (shouldn't normally happen if
            # tick_world was called first).
            from src.sim.central_table import CentralTable
            table = CentralTable()

        # Phase cascade (same as tick() but reusing the pre-built table).
        self._last_tick_orders = {}
        prev_tick_sales = self._last_tick_sales
        self._last_tick_sales = {}

        if self.levels:
            max_level = max(self.levels.values())
        else:
            max_level = 0

        for p in range(1, max_level + 1):
            buyers_at_level = [
                node
                for node_id, node in self.nodes.items()
                if self.levels.get(node_id, 0) == p
            ]
            if not buyers_at_level:
                continue

            buyers_at_level = shuffle_buyers(buyers_at_level, self.allocation_rng)

            for buyer in buyers_at_level:
                if isinstance(buyer, DemandSinkNode):
                    demand_target = float(
                        buyer.demand_target(
                            tick=current_tick,
                            market=self.market,
                            registry=self.item_registry,
                            world_rng=self.world_rng,
                        )
                    )
                    obs = build_sink_obs(
                        buyer, tick=current_tick, demand_target=demand_target
                    )
                    direct_supplier_ids = self.graph.suppliers_of(buyer.id)
                    # See the cascade above: inject direct suppliers so an
                    # attached sink policy buys only from its own shops, not
                    # from an indirect upstream seller of the same product.
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
                        supplier = self.nodes.get(supplier_id)
                        if supplier is None:
                            continue
                        lt = self.graph.lead_time(supplier_id, buyer.id, buyer.product_id)
                        result = execute_buy(
                            buyer=buyer,
                            supplier=supplier,
                            pid=buyer.product_id,
                            qty_requested=qty,
                            table=table,
                            event_engine=self.event_engine,
                            current_tick=current_tick,
                            lead_time=lt,
                        )
                        if isinstance(supplier, IntermediateNode) and result.qty_filled > 0:
                            sup_sales = self._last_tick_sales.setdefault(supplier_id, {})
                            pid_sold = buyer.product_id
                            sup_sales[pid_sold] = sup_sales.get(pid_sold, 0) + result.qty_filled

                elif isinstance(buyer, IntermediateNode):
                    obs = build_intermediate_obs(buyer, tick=current_tick)
                    obs["prev_tick_sales"] = prev_tick_sales.get(buyer.id, {})
                    direct_supplier_ids = self.graph.suppliers_of(buyer.id)
                    # Inject the buyer's direct upstream suppliers so the policy
                    # routes reorders to them, not to its own published offers
                    # (a shop publishes its carried inventory for downstream
                    # sinks, so the central table would otherwise list the shop
                    # itself as a "supplier" and the lines below would drop it).
                    obs["direct_supplier_ids"] = direct_supplier_ids
                    if buyer.policy is not None:
                        action = buyer.policy.decide(obs, table)
                    else:
                        action = {"order": {}, "list_price": {}, "min_order_imposed": {}}

                    orders = action.get("order", {})
                    buyer_orders = self._last_tick_orders.setdefault(buyer.id, {})
                    for pid, order_lines in orders.items():
                        for supplier_id, qty in order_lines:
                            if qty <= 0:
                                continue
                            if supplier_id not in direct_supplier_ids:
                                continue
                            supplier = self.nodes.get(supplier_id)
                            if supplier is None:
                                continue
                            lt = self.graph.lead_time(supplier_id, buyer.id, pid)
                            execute_buy(
                                buyer=buyer,
                                supplier=supplier,
                                pid=pid,
                                qty_requested=qty,
                                table=table,
                                event_engine=self.event_engine,
                                current_tick=current_tick,
                                lead_time=lt,
                            )
                            buyer_orders[pid] = buyer_orders.get(pid, 0) + qty

        # Produce — factories run policy and produce up to capacity.
        for node_id, node in self.nodes.items():
            if isinstance(node, FactoryNode):
                obs = build_factory_obs(node, tick=current_tick)
                if node.policy is not None:
                    action = node.policy.decide(obs)
                else:
                    from src.sim.distributions import Distribution
                    cap = node.capacity_per_tick
                    if isinstance(cap, Distribution):
                        produce_qty = int(cap.sample(self.world_rng))
                    else:
                        produce_qty = int(cap)
                    action = {"produce_qty": produce_qty, "list_price": node.unit_cost}

                produce_qty = int(action.get("produce_qty", 0))
                node.inventory += produce_qty
                # ADR 0013 rule 3: production absorbs cash at unit_cost
                # (zero-margin factory). Recovered exactly on sale at
                # list_price == unit_cost, so produce-then-sell nets to zero.
                node.cash -= node.unit_cost * produce_qty
                new_price = action.get("list_price")
                if new_price is not None:
                    node.list_price = float(new_price)

        # consume_demand_sinks — credit income_rate cash to each sink.
        for node_id, node in self.nodes.items():
            if isinstance(node, DemandSinkNode):
                node.cash += node.income_rate

        # Clean up the stashed table.
        self._current_table = None


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
# build_world — module-level factory (was build_graph_world)
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
    from src.sim.graph import EdgeSpec, build_graph, compute_levels

    if not scenario.is_graph:
        raise ValueError(
            "build_world requires a graph-mode scenario "
            "(scenario.is_graph must be True)"
        )

    # Seeding (ADR 0016): world_rng from world_seed; allocation_rng from the
    # "allocation" sub-seed derived via _derive_seed.
    world_rng: Random = Random(scenario.world_seed)
    allocation_seed = _derive_seed(scenario.world_seed, "allocation")
    allocation_rng: Random = Random(allocation_seed)

    # Shared world objects — same construction order as the old build_world.
    item_registry = ItemRegistry(
        scenario.item_lifecycle, scenario.catalog, world_rng
    )
    market = Market(
        scenario.market,
        world_rng,
        scenario.start_date,
        registry=item_registry,
    )
    event_engine = EventEngine(scenario.disruption, world_rng)

    # Build the Graph topology.
    node_ids = [ni.node.id for ni in scenario.nodes]
    edges: list[EdgeSpec] = list(scenario.edges)
    graph = build_graph(node_ids, edges)

    # Compute echelon levels.
    levels = compute_levels(graph)

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
        # Assign the computed echelon level onto the node.
        node.level = levels.get(node.id)
        nodes[node.id] = node

    return Simulation(
        scenario=scenario,
        world_rng=world_rng,
        allocation_rng=allocation_rng,
        item_registry=item_registry,
        market=market,
        event_engine=event_engine,
        graph=graph,
        nodes=nodes,
        levels=levels,
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
                (length ``n_steps + 1``),
                ``"products"`` with resolved freshness + lifecycle data.
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

        # Build the ``global.products`` section from the item registry.
        products_global: dict[str, dict[str, Any]] = {}
        for pid, item in self._sim.item_registry.items.items():
            products_global[pid] = {
                "freshness_alpha": item.freshness_alpha,
                "freshness_decay": item.freshness_decay,
                "init_stock_share": item.init_stock_share,
                "lifecycle_stage": [item.lifecycle_stage] * len(step_series),
            }

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
                "products": products_global,
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


# ---------------------------------------------------------------------------
# Backward-compatibility aliases (kept for existing scenario imports)
# ---------------------------------------------------------------------------

#: Alias for scenarios that import ``GraphSimulation`` directly.
GraphSimulation = Simulation

#: Alias for scenarios that import ``GraphRunner`` directly.
GraphRunner = Runner

#: Alias for scenarios that import ``build_graph_world`` directly.
build_graph_world = build_world


__all__ = [
    "Runner",
    "Simulation",
    "TickResult",
    "build_world",
    # Backward-compatibility aliases
    "GraphSimulation",
    "GraphRunner",
    "build_graph_world",
]
