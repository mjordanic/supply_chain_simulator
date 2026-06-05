"""Single-episode rollout primitives for tuning studies.

Self-contained — no imports from ``src.rl``. Delegates per-tick state machine
to ``src.sim.runner.Simulation`` via ``build_world`` + ``Simulation.tick()``.

Graph-mode migration (issue 13)
--------------------------------
``run_policy_episode`` now builds a graph-shaped scenario:

    FactoryNode(F_{pid}) → IntermediateNode("S") → DemandSinkNode(sink_{pid})

The policy (a ``MultiSupplierTextbookPolicy`` or any ``IntermediatePolicy``)
is attached to the ``IntermediateNode`` by node id.  ``_record_active_subset``
collects the same field set from the IntermediateNode as the old Store-based
path did.
"""

from __future__ import annotations

from src.sim.episode_sampler import EpisodeSpec
from src.sim.metrics import RunSlice, aggregate_episode
from src.sim.policy import Policy


# ---------------------------------------------------------------------------
# _make_tracking_sinks — DemandSinkNode subclass that records demand_target
# ---------------------------------------------------------------------------


def _make_tracking_sink_class():
    """Return a DemandSinkNode subclass that records demand_target each tick."""
    from src.sim.node import DemandSinkNode

    class _TrackingDemandSinkNode(DemandSinkNode):
        """Records the last demand_target computed by ``demand_target()``."""

        _last_demand: int = 0

        def demand_target(self, tick, market, catalog, world_rng):  # type: ignore[override]
            result = super().demand_target(tick, market, catalog, world_rng)
            self._last_demand = int(result)
            return result

    return _TrackingDemandSinkNode


# Cache the class at module level so repeated calls share the same type.
_TrackingDemandSinkNode = None  # initialised lazily below


def _get_tracking_sink_class():
    global _TrackingDemandSinkNode
    if _TrackingDemandSinkNode is None:
        _TrackingDemandSinkNode = _make_tracking_sink_class()
    return _TrackingDemandSinkNode


# ---------------------------------------------------------------------------
# _build_graph_scenario — convert EpisodeSpec into a graph-mode Scenario
# ---------------------------------------------------------------------------


def _build_graph_scenario(spec: EpisodeSpec):
    """Convert a store-mode ``EpisodeSpec`` into a graph-mode Scenario.

    Topology::

        FactoryNode(F_{pid}) ─┐
                               ├─► IntermediateNode("S") ─► DemandSinkNode(sink_{pid})
        FactoryNode(F_{pid2}) ─┘

    One factory per active product, one intermediate node ("S") carrying all K
    active products, one demand-sink per active product.

    The EpisodeSpec's ``scenario.stores[0].template`` drives capacity,
    init_balance, holding_rate, order_fee, delivery_lag, and init_stock_pct.
    The catalog's ``unit_cost`` and ``base_price`` drive factory costs and
    intermediate list_prices.

    Returns
    -------
    tuple
        ``(graph_scenario, holding_rate, order_fee)`` where
        ``graph_scenario`` is a graph-mode ``Scenario`` with the policy slot
        on "S" left empty (caller attaches via ``build_world`` overrides).
    """
    from src.sim.distributions import Constant
    from src.sim.graph import EdgeSpec
    from src.sim.node import FactoryNode, IntermediateNode
    from src.sim.scenario import NodeInstance, Scenario

    TrackSink = _get_tracking_sink_class()

    sto = spec.scenario
    active_pids = list(spec.active_subset)
    catalog_by_pid = {w.product_id: w for w in sto.catalog}
    template = sto.stores[0].template

    capacity = int(template.capacity)
    init_balance = float(template.init_balance)
    holding_rate = float(template.holding_rate)
    order_fee = float(template.order_fee)
    delivery_lag = int(template.delivery_lag)
    init_stock_pct = float(getattr(template, "init_stock_pct", 0.0))

    # --- FactoryNodes: one per active product ---
    factory_nodes = []
    for i, pid in enumerate(active_pids):
        ware = catalog_by_pid[pid]
        # Produce enough per tick to service demand (capacity / K + buffer).
        factory_cap = max(10, capacity // max(1, len(active_pids)) + 5)
        fn = FactoryNode(
            id=f"F_{pid}",
            region=template.region,
            init_seed=100 + i,
            produces_product_id=pid,
            unit_cost=float(ware.unit_cost),
            capacity_per_tick=Constant(factory_cap),
            inventory=factory_cap * delivery_lag,  # pre-stock factories
            list_price=float(ware.unit_cost),
        )
        factory_nodes.append(fn)

    # --- IntermediateNode "S": carries all K active products ---
    total_init_stock = int(capacity * init_stock_pct)
    per_sku_stock = total_init_stock // max(1, len(active_pids))
    init_inventory = {pid: per_sku_stock for pid in active_pids}
    init_list_prices = {
        pid: float(catalog_by_pid[pid].base_price) for pid in active_pids
    }
    s_node = IntermediateNode(
        id="S",
        region=template.region,
        init_seed=999,
        carried_products=set(active_pids),
        capacity=capacity,
        inventory=dict(init_inventory),
        list_prices=dict(init_list_prices),
        min_order_imposed={pid: 0 for pid in active_pids},
        cash=init_balance,
    )

    # --- DemandSinkNodes: one per active product (tracking subclass) ---
    sink_nodes = []
    for i, pid in enumerate(active_pids):
        ware = catalog_by_pid[pid]
        # income_rate: enough to keep the sink solvent (base_price × 20 per tick)
        income_rate = float(ware.base_price) * 20.0
        sink = TrackSink(
            id=f"sink_{pid}",
            region=template.region,
            init_seed=1000 + i,
            demand_dist=None,   # will use world's base_demand via market
            income_rate=income_rate,
            cash=init_balance / max(1, len(active_pids)),
            activation_tick={pid: 0},
            product_id=pid,
        )
        # Set a simple Uniform demand distribution
        from src.sim.distributions import Uniform
        sink.demand_dist = Uniform(2, 8)
        sink_nodes.append(sink)

    # --- NodeInstances ---
    all_node_instances = []
    for fn in factory_nodes:
        all_node_instances.append(NodeInstance(node=fn, init_seed=fn.init_seed, policy=None))
    all_node_instances.append(NodeInstance(node=s_node, init_seed=s_node.init_seed, policy=None))
    for sn in sink_nodes:
        all_node_instances.append(NodeInstance(node=sn, init_seed=sn.init_seed, policy=None))

    # --- Edges ---
    edges = []
    for pid in active_pids:
        edges.append(EdgeSpec(
            supplier_id=f"F_{pid}",
            buyer_id="S",
            default_lead_time=delivery_lag,
        ))
        edges.append(EdgeSpec(
            supplier_id="S",
            buyer_id=f"sink_{pid}",
            default_lead_time=1,
        ))

    # --- Graph Scenario (stores kept for backward-compat; is_graph==True wins) ---
    graph_scenario = Scenario(
        catalog=sto.catalog,
        market=sto.market,
        disruption=sto.disruption,
        item_lifecycle=sto.item_lifecycle,
        stores=sto.stores,
        n_steps=sto.n_steps,
        start_date=sto.start_date,
        world_seed=sto.world_seed,
        nodes=all_node_instances,
        edges=edges,
    )

    return graph_scenario, holding_rate, order_fee


# ---------------------------------------------------------------------------
# _record_active_subset — collect one tick of IntermediateNode state
# ---------------------------------------------------------------------------


def _record_active_subset(
    pids: list,
    s_node,
    sales_this_tick: dict,
    live_sinks: dict,
    orders_this_tick: dict,
    run_slice: RunSlice,
    catalog_lookup: dict,
    order_fee: float,
    holding_rate: float,
) -> None:
    """Project one tick of IntermediateNode state onto the active-subset RunSlice.

    Collects the same field set as the legacy Store path:
    sales, demand, inventory, price, msrp, revenue, holding_cost, order_cost,
    order_fee.

    Parameters
    ----------
    pids:
        Ordered list of active product ids (inner axis of RunSlice).
    s_node:
        The live ``IntermediateNode`` ("S") after the tick has settled.
    sales_this_tick:
        ``{pid: qty}`` — units S sold to downstream sinks this tick.
        Taken from ``sim._last_tick_sales.get("S", {})``.
    live_sinks:
        ``{pid: _TrackingDemandSinkNode}`` — each node's ``._last_demand``
        holds the demand_target sampled in the most recent tick.
    orders_this_tick:
        ``{pid: qty}`` — units S ordered from upstream factories this tick.
        Taken from ``sim._last_tick_orders.get("S", {})``.
    run_slice:
        Mutable ``RunSlice`` to append onto.
    catalog_lookup:
        ``{pid: Ware}`` — provides ``unit_cost`` and ``base_price`` (MSRP).
    order_fee:
        Fixed fee charged once per tick when any order was placed by S.
    holding_rate:
        Per-unit-per-tick holding cost rate (fraction of unit_cost).
    """
    sales_row = [int(sales_this_tick.get(pid, 0)) for pid in pids]
    demand_row = [int(getattr(live_sinks.get(pid), "_last_demand", 0)) for pid in pids]
    inventory_row = [int(s_node.inventory.get(pid, 0)) for pid in pids]
    price_row = [float(s_node.list_prices.get(pid, 0.0)) for pid in pids]
    msrp_row = [float(catalog_lookup[pid].base_price) for pid in pids]
    revenue_row = [sales_row[i] * price_row[i] for i in range(len(pids))]
    holding_cost_row = [
        inventory_row[i] * holding_rate * float(catalog_lookup[pids[i]].unit_cost)
        for i in range(len(pids))
    ]
    order_cost_row = [
        int(orders_this_tick.get(pid, 0)) * float(catalog_lookup[pid].unit_cost)
        for pid in pids
    ]
    # Order fee: charged once per non-zero-order tick, attributed to the first pid.
    any_order = any(orders_this_tick.get(pid, 0) > 0 for pid in pids)
    order_fee_row = [
        float(order_fee) if (any_order and i == 0) else 0.0
        for i in range(len(pids))
    ]

    run_slice.sales.append(sales_row)
    run_slice.demand.append(demand_row)
    run_slice.inventory.append(inventory_row)
    run_slice.price.append(price_row)
    run_slice.msrp.append(msrp_row)
    run_slice.revenue.append(revenue_row)
    run_slice.holding_cost.append(holding_cost_row)
    run_slice.order_cost.append(order_cost_row)
    run_slice.order_fee.append(order_fee_row)


# ---------------------------------------------------------------------------
# run_policy_episode — main public entry point
# ---------------------------------------------------------------------------


def run_policy_episode(policy: Policy, spec: EpisodeSpec) -> dict[str, float]:
    """Roll out ``policy`` on a graph-shaped scenario derived from ``spec``.

    ``spec.scenario`` carries a store-mode Scenario (catalog, template, seeds).
    This function converts it into a graph-mode scenario:

        FactoryNode(F_{pid}) → IntermediateNode("S") → DemandSinkNode(sink_{pid})

    The ``policy`` (an ``IntermediatePolicy`` / ``MultiSupplierTextbookPolicy``)
    is attached to the IntermediateNode "S" via ``policy_overrides={"S": policy}``.

    Parameters
    ----------
    policy:
        Any policy implementing ``decide(obs_intermediate, central_table)``.
        The textbook policies (``OrderUpToPolicy`` etc.) satisfy this.
    spec:
        ``EpisodeSpec`` from the tuning episode sampler.

    Returns
    -------
    dict[str, float]
        Aggregate KPIs from ``aggregate_episode``.
    """
    from src.sim.runner import build_world

    graph_scenario, holding_rate, order_fee = _build_graph_scenario(spec)
    active_pids = list(spec.active_subset)
    catalog_lookup = {w.product_id: w for w in spec.scenario.catalog}

    sim = build_world(graph_scenario, policy_overrides={"S": policy})

    s_node = sim.nodes["S"]
    live_sinks = {pid: sim.nodes[f"sink_{pid}"] for pid in active_pids}

    run_slice = RunSlice(active_pids=active_pids)

    for _ in range(spec.scenario.n_steps):
        sim.tick()

        sales_this_tick = dict(sim._last_tick_sales.get("S", {}))
        orders_this_tick = dict(sim._last_tick_orders.get("S", {}))

        _record_active_subset(
            pids=active_pids,
            s_node=s_node,
            sales_this_tick=sales_this_tick,
            live_sinks=live_sinks,
            orders_this_tick=orders_this_tick,
            run_slice=run_slice,
            catalog_lookup=catalog_lookup,
            order_fee=order_fee,
            holding_rate=holding_rate,
        )

    return aggregate_episode(run_slice)


__all__ = ["run_policy_episode"]
