"""Single-episode rollout primitives for tuning studies.

Self-contained — no imports from ``src.rl``. Delegates per-tick state machine
to ``src.sim.runner.Simulation`` via ``build_world`` + ``Simulation.tick()``.

Graph-mode (issue 06)
----------------------
``run_policy_episode`` now receives a ``TuningEpisodeSpec`` whose
``scenario`` is already a graph-mode Scenario (built by ``episode.py``):

    FactoryNode(F_<pid>) → IntermediateNode("S") → DemandSinkNode(D_<pid>)

The policy (a ``MultiSupplierTextbookPolicy`` or any ``IntermediatePolicy``)
is attached to the ``IntermediateNode`` by node id "S". No store-mode
conversion is needed.
"""

from __future__ import annotations

from src.tuning.episode import TuningEpisodeSpec
from src.sim.policy import Policy

# ---------------------------------------------------------------------------
# Legacy RunSlice + aggregate_episode — TODO issue 07 removes these once
# tuning/rollout.py is refactored to consume the shared DataFrame path.
# ---------------------------------------------------------------------------

from dataclasses import dataclass as _dc, field as _field
from typing import Dict as _Dict, List as _List


@_dc
class RunSlice:
    """Per-active-SKU per-tick traces (legacy list-of-lists representation)."""
    sales: _List[_List[float]] = _field(default_factory=list)
    demand: _List[_List[float]] = _field(default_factory=list)
    inventory: _List[_List[float]] = _field(default_factory=list)
    price: _List[_List[float]] = _field(default_factory=list)
    msrp: _List[_List[float]] = _field(default_factory=list)
    revenue: _List[_List[float]] = _field(default_factory=list)
    holding_cost: _List[_List[float]] = _field(default_factory=list)
    order_cost: _List[_List[float]] = _field(default_factory=list)
    order_fee: _List[_List[float]] = _field(default_factory=list)
    active_pids: _List[str] = _field(default_factory=list)


def _flat(matrix: _List[_List[float]]) -> _List[float]:
    out: list[float] = []
    for row in matrix:
        out.extend(row)
    return out


def _safe_sum(v: _List[float]) -> float:
    return float(sum(v)) if v else 0.0


def _safe_mean(v: _List[float]) -> float:
    return float(sum(v) / len(v)) if v else 0.0


def _service_level(rs: RunSlice) -> float:
    return _safe_sum(_flat(rs.sales)) / max(1.0, _safe_sum(_flat(rs.demand)))


def _stockout_rate(rs: RunSlice) -> float:
    flat_inv = _flat(rs.inventory)
    return float(sum(1 for v in flat_inv if v == 0.0)) / float(len(flat_inv)) if flat_inv else 0.0


def _mean_price_pct(rs: RunSlice) -> float:
    fp, fm = _flat(rs.price), _flat(rs.msrp)
    if not fp or not fm:
        return 1.0
    return _safe_mean([p / max(1e-9, m) for p, m in zip(fp, fm)])


def _inventory_turnover(rs: RunSlice) -> float:
    return _safe_sum(_flat(rs.sales)) / max(1.0, _safe_mean(_flat(rs.inventory)))


def _profit_decomposition(rs: RunSlice) -> _Dict[str, float]:
    revenue = _safe_sum(_flat(rs.revenue))
    holding = _safe_sum(_flat(rs.holding_cost))
    order_cost = _safe_sum(_flat(rs.order_cost))
    fees = _safe_sum(_flat(rs.order_fee))
    return {
        "revenue": revenue,
        "holding_cost": holding,
        "order_cost": order_cost,
        "order_fees": fees,
        "net_profit": revenue - (holding + order_cost + fees),
    }


def aggregate_episode(rs: RunSlice) -> _Dict[str, float]:
    decomp = _profit_decomposition(rs)
    return {
        "service_level": _service_level(rs),
        "stockout_rate": _stockout_rate(rs),
        "mean_price_pct_of_msrp": _mean_price_pct(rs),
        "inventory_turnover": _inventory_turnover(rs),
        **decomp,
    }


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
    """Project one tick of IntermediateNode state onto the active-subset RunSlice."""
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


def run_policy_episode(policy: Policy, spec: TuningEpisodeSpec) -> dict[str, float]:
    """Roll out ``policy`` on the graph-mode scenario in ``spec``.

    ``spec.scenario`` is a graph-mode Scenario built by ``episode.sample_episode``:

        FactoryNode(F_<pid>) → IntermediateNode("S") → DemandSinkNode(D_<pid>)

    The ``policy`` (an ``IntermediatePolicy`` / ``MultiSupplierTextbookPolicy``)
    is attached to the IntermediateNode "S" via ``policy_overrides={"S": policy}``.

    Parameters
    ----------
    policy:
        Any policy implementing ``decide(obs_intermediate, central_table)``.
    spec:
        ``TuningEpisodeSpec`` from ``episode.sample_episode``.

    Returns
    -------
    dict[str, float]
        Aggregate KPIs from ``aggregate_episode``.
    """
    from src.sim.runner import build_world

    TrackSink = _get_tracking_sink_class()

    scenario = spec.scenario
    active_pids = list(spec.active_subset)
    catalog_lookup = {w.product_id: w for w in scenario.catalog}

    # Extract holding_rate and order_fee from the intermediate node's edges/config.
    # These are stored in the TuningConfig but not on the scenario directly.
    # Use defaults consistent with the episode builder (config.holding_rate, config.order_fee).
    # Since spec no longer carries StoreTemplate, read from the intermediate node if possible,
    # or fall back to tuning-module defaults.
    holding_rate = 0.01  # default — caller may override via a subclass if needed
    order_fee = 50.0

    # Monkey-patch sink nodes to tracking subclass before building world.
    # We replace the sink NodeInstances in the scenario with tracking variants.
    from src.sim.scenario import NodeInstance
    from src.sim.node import DemandSinkNode

    new_node_instances = []
    for ni in scenario.nodes:
        if isinstance(ni.node, DemandSinkNode):
            pid = ni.node.product_id
            tracking_node = TrackSink(
                id=ni.node.id,
                region=ni.node.region,
                init_seed=ni.node.init_seed,
                product_id=pid,
                demand_dist=ni.node.demand_dist,
                income_rate=ni.node.income_rate,
                cash=ni.node.cash,
                activation_tick=ni.node.activation_tick,
            )
            new_node_instances.append(NodeInstance(
                node=tracking_node,
                init_seed=ni.init_seed,
                policy=ni.policy,
            ))
        else:
            new_node_instances.append(ni)

    from src.sim.scenario import Scenario
    tracking_scenario = Scenario(
        catalog=scenario.catalog,
        market=scenario.market,
        disruption=scenario.disruption,
        item_lifecycle=scenario.item_lifecycle,
        nodes=new_node_instances,
        edges=scenario.edges,
        n_steps=scenario.n_steps,
        start_date=scenario.start_date,
        world_seed=scenario.world_seed,
    )

    sim = build_world(tracking_scenario, policy_overrides={"S": policy})

    s_node = sim.nodes["S"]
    # Collect live sink nodes by product id.
    live_sinks = {}
    for pid in active_pids:
        sink_id = f"D_{pid}"
        if sink_id in sim.nodes:
            live_sinks[pid] = sim.nodes[sink_id]

    run_slice = RunSlice(active_pids=active_pids)

    for _ in range(scenario.n_steps):
        sim.tick()

        sales_this_tick = dict(sim._tick_sales.get("S", {}))
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
