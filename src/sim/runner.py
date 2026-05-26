"""Runner with the full integration loop.

The runner owns one simulation execution: given a frozen ``Scenario``,
it instantiates the world (``Market``, ``EventEngine``, ``ItemRegistry``,
and a per-instance ``Store`` for each ``StoreInstance``), then runs the
observe → decide → advance → log loop for ``n_steps`` ticks.

The seeding contract (ADR 0001):

- ``world_rng`` is seeded from ``Scenario.world_seed``. Consumed by
  ``ItemRegistry`` (lifecycle stage progression), ``Market`` (demand /
  supply update + ``sample_demand`` draws + trend resampling), and
  ``EventEngine`` (event spawning).
- Each ``Policy`` instance owns its own ``policy_rng``. World and policy
  streams never share an RNG. Two ``Scenario`` runs that differ only in
  the attached ``Policy`` produce identical world streams.
- Two stores instantiated from the same ``(StoreTemplate, init_seed)``
  start step 0 bit-identical regardless of the attached ``Policy``.
  Per-store initialisation consumes only a per-instance
  ``init_rng = Random(init_seed)``; never ``world_rng`` or ``policy_rng``.

Per-step loop (verbatim from the previous ``SimulationRunner._advance``,
modulo the runner-internal skeleton replaced by the typed modules):

1. ``market.tick()`` — advance demand/supply state, increment the
   internal step counter and date.
2. ``event_engine.tick(market)`` — apply active events, spawn check,
   then fire any delivery callbacks scheduled for ``current_step``.
3. ``item_registry.tick()`` — one lifecycle draw per item.
4. For each store: build observation against post-tick state and call
   ``store.decide`` to thread policy actions back into store state.
5. For each store: dispatch newly placed orders by scheduling
   ``EventEngine`` callbacks at ``current_step + lead_time``.
6. For each store, for each product in inventory: draw realised demand
   via ``market.sample_demand(pid, store, price)`` and call
   ``store.settle(pid, demand, price, order_qty)`` to fold sales /
   revenue / costs / balance.
7. Append a ``run_log`` snapshot.

A step-0 baseline snapshot is appended *before* the loop runs so every
metric series in the run log has length ``n_steps + 1`` (initial state
+ n post-tick states).

New sim surface (issue 03):

- ``build_world(scenario, *, policy_overrides=None) -> Simulation`` —
  module-level free function that constructs the mutable bundle.
- ``Simulation`` — mutable bundle holding ``(scenario, world_rng,
  item_registry, market, event_engine, stores)``.
- ``Simulation.tick_world() -> list[WorldEvent]`` — phase 1 (world only).
- ``Simulation.tick_decide_and_settle() -> TickResult`` — phase 2
  (per-store observe/decide/dispatch/demand).
- ``Simulation.tick() -> TickResult`` — convenience composing both phases.
- ``TickResult`` — frozen dataclass with actions, demand_traces,
  active_events.

``Runner`` keeps its unchanged public API (``Runner(scenario).run() ->
dict``); its body now delegates to ``build_world`` + ``Simulation.tick()``
+ the existing log-collection helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Any

from src.sim.event_engine import EventEngine, WorldEvent
from src.sim.item_registry import ItemRegistry
from src.sim.market import Market
from src.sim.policy import Policy
from src.sim.scenario import Scenario
from src.sim.store import Store


# ---------------------------------------------------------------------------
# TickResult — frozen data carrier for one simulation tick
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TickResult:
    """Data produced by one call to ``Simulation.tick()`` or phase 2 only.

    ``actions``       — ``{store_idx: {"order": {...}, "price": {...}, ...}}``.
    ``demand_traces`` — ``{store_idx: {pid: realised_demand_int}}``.
    ``active_events`` — snapshot of events active *after* ``tick_world``
                        fired (carry-through from phase 1; set to ``[]``
                        when constructed from phase 2 alone).
    """

    actions: dict[int, dict[str, Any]]
    demand_traces: dict[int, dict[str, int]]
    active_events: list[WorldEvent]


# ---------------------------------------------------------------------------
# Simulation — mutable world bundle
# ---------------------------------------------------------------------------

class Simulation:
    """Mutable bundle returned by ``build_world``.

    Attributes
    ----------
    scenario      — the frozen ``Scenario`` used to build this bundle.
    world_rng     — shared world RNG (all market/event/lifecycle draws).
    item_registry — catalog + per-item lifecycle state.
    market        — regional demand/supply environment.
    event_engine  — disruption events + order-delivery callbacks.
    stores        — one ``Store`` per ``StoreInstance``.
    """

    def __init__(
        self,
        scenario: Scenario,
        world_rng: Random,
        item_registry: ItemRegistry,
        market: Market,
        event_engine: EventEngine,
        stores: list[Store],
    ) -> None:
        self.scenario = scenario
        self.world_rng = world_rng
        self.item_registry = item_registry
        self.market = market
        self.event_engine = event_engine
        self.stores = stores

    # ------------------------------------------------------------------
    # Two-phase API
    # ------------------------------------------------------------------

    def tick_world(self) -> list[WorldEvent]:
        """Phase 1: advance the world (market, events, lifecycle).

        Returns the list of ``WorldEvent`` objects active after this tick
        so callers can log them or inject actions before phase 2.
        """
        self.market.tick()
        active_events = self.event_engine.tick(self.market)
        self.item_registry.tick()
        return active_events

    def tick_decide_and_settle(
        self,
        active_events: list[WorldEvent] | None = None,
    ) -> TickResult:
        """Phase 2: per-store observe → decide → dispatch → settle demand.

        ``active_events`` is forwarded into the returned ``TickResult``
        unchanged.  Callers that run both phases independently should pass
        the return value of ``tick_world()`` here; callers using the
        convenience ``tick()`` wrapper can ignore this parameter.

        Demand-sampling order: ``store.inventory`` iteration order (catalog
        order via the ``__init__`` registration loop), matching ADR 0003.
        """
        if active_events is None:
            active_events = []

        actions: dict[int, dict[str, Any]] = {}
        for i, store in enumerate(self.stores):
            obs = store.observe(
                self.market.current_step(),
                self.item_registry,
            )
            actions[i] = store.decide(obs)

        for i, store in enumerate(self.stores):
            _dispatch_orders(store, actions[i], self.market, self.event_engine)

        demand_traces: dict[int, dict[str, int]] = {}
        for i, store in enumerate(self.stores):
            demand_traces[i] = _process_demand(store, actions[i], self.market)

        return TickResult(
            actions=actions,
            demand_traces=demand_traces,
            active_events=active_events,
        )

    def tick(self) -> TickResult:
        """Convenience: run both phases and return a combined ``TickResult``.

        Equivalent to ``tick_decide_and_settle(tick_world())``.
        """
        active_events = self.tick_world()
        return self.tick_decide_and_settle(active_events)


# ---------------------------------------------------------------------------
# build_world — module-level factory
# ---------------------------------------------------------------------------

def build_world(
    scenario: Scenario,
    *,
    policy_overrides: list[Policy] | None = None,
) -> Simulation:
    """Construct and return a ``Simulation`` bundle from ``scenario``.

    ``policy_overrides``, when supplied, must have the same length as
    ``scenario.stores``.  Store ``i`` is built with
    ``policy_overrides[i]`` in place of ``StoreInstance.policy``.
    Override wins when both are set — the canonical pattern for
    spec-based callers (tuning, RL eval, RL env) whose specs carry
    ``policy=None``.  Scenario-authoring callers (``main.py``,
    hand-authored scenarios) leave ``policy_overrides=None`` and let
    ``StoreInstance.policy`` flow through.
    """
    if policy_overrides is not None and len(policy_overrides) != len(scenario.stores):
        raise ValueError(
            f"policy_overrides has {len(policy_overrides)} entries but "
            f"scenario has {len(scenario.stores)} stores"
        )

    world_rng: Random = Random(scenario.world_seed)
    # Construction order: ItemRegistry → Market → EventEngine (matches Runner).
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

    stores: list[Store] = []
    for i, s in enumerate(scenario.stores):
        policy = (
            policy_overrides[i]
            if policy_overrides is not None
            else s.policy
        )
        stores.append(
            Store(
                s.template,
                s.init_seed,
                policy,
                scenario.catalog,
                freshness_alpha=item_registry.default_freshness_alpha,
                freshness_decay=item_registry.default_freshness_decay,
                item_registry=item_registry,
            )
        )

    return Simulation(
        scenario=scenario,
        world_rng=world_rng,
        item_registry=item_registry,
        market=market,
        event_engine=event_engine,
        stores=stores,
    )


# ---------------------------------------------------------------------------
# Runner — unchanged public API; body delegates to build_world + Simulation
# ---------------------------------------------------------------------------

class Runner:
    """Drive a ``Scenario`` to a full ``RunLog``.

    Public entry points: ``Runner(scenario)`` builds the world, then
    ``run()`` returns the log. The ``stores`` attribute is exposed so
    determinism tests can read step-0 state pre-run.

    Implementation note: the body of ``__init__`` now delegates to
    ``build_world``; ``run()`` uses ``Simulation.tick()`` per step.
    The public API and all log contents are unchanged.
    """

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        # Delegate world construction to the shared factory.
        _sim = build_world(scenario)
        self.world_rng = _sim.world_rng
        self.item_registry = _sim.item_registry
        self.market = _sim.market
        self.event_engine = _sim.event_engine
        self.stores = _sim.stores
        # Keep a reference to the Simulation bundle for use in run().
        self._sim = _sim

    def run(self) -> dict[str, Any]:
        """Execute the full simulation and return the accumulated run log."""
        run_log = self._init_run_log()

        # Step-0 baseline snapshot: pre-tick state, no actions yet.
        self._log_state(run_log, actions={}, demand_traces=None, active_events=[])

        for _ in range(self.scenario.n_steps):
            result = self._sim.tick()
            run_log["global"]["events"]["occurrences"].append(
                _event_payloads(result.active_events)
            )
            self._log_state(
                run_log,
                actions=result.actions,
                demand_traces=result.demand_traces,
                active_events=result.active_events,
            )

        return run_log

    # ------------------------------------------------------------------ private

    def _init_run_log(self) -> dict[str, Any]:
        """Allocate the run-log skeleton with all expected keys preset."""
        run_log: dict[str, Any] = {
            "global": {
                "time": {"simulation_step": [], "simulation_date": []},
                "market_supply": {r: [] for r in self.market.regions},
                "market_demand": {r: [] for r in self.market.regions},
                "events": {"occurrences": []},
                "products": {
                    pid: {
                        "lifecycle_stage": [],
                        "freshness_alpha": item.freshness_alpha,
                        "freshness_decay": item.freshness_decay,
                        "init_stock_share": item.init_stock_share,
                    }
                    for pid, item in self.item_registry.items.items()
                },
            },
            "stores": {},
        }
        for i, store in enumerate(self.stores):
            run_log["stores"][i] = {
                "balance": [],
                "active_product_count": [],
                "demand_trace": [],
                "step0_inventory": dict(store.inventory),
                "step0_active_items": list(store.active_items),
                "step0_capacity": store.capacity,
                "products": {
                    pid: {
                        "inventory": [],
                        "demand": [],
                        "sales": [],
                        "order_quantity": [],
                        "outstanding_orders": [],
                        "promotion_status": [],
                        "active_status": [],
                        "price": [],
                        "revenue": [],
                        "total_cost": [],
                        "holding_cost": [],
                        "profit": [],
                    }
                    for pid in store.inventory
                },
            }
        return run_log

    def _log_state(
        self,
        run_log: dict[str, Any],
        actions: dict[int, dict[str, Any]],
        demand_traces: dict[int, dict[str, int]] | None,
        active_events: list[WorldEvent],
    ) -> None:
        """Append one timestep of state to ``run_log``."""
        run_log["global"]["time"]["simulation_step"].append(
            self.market.current_step()
        )
        run_log["global"]["time"]["simulation_date"].append(
            self.market.current_date()
        )

        for region in self.market.regions:
            state = self.market.market_state[region]
            run_log["global"]["market_supply"][region].append(state["market_supply"])
            run_log["global"]["market_demand"][region].append(state["market_demand"])

        for pid, item in self.item_registry.items.items():
            run_log["global"]["products"][pid]["lifecycle_stage"].append(
                item.lifecycle_stage
            )

        for i, store in enumerate(self.stores):
            store_log = run_log["stores"][i]
            action = actions.get(i, {})
            order_decisions = action.get("order", {})
            price_decisions = action.get("price", {})

            store_log["balance"].append(store.balance)
            store_log["active_product_count"].append(len(store.active_items))
            if demand_traces is None:
                store_log["demand_trace"].append({})
            else:
                store_log["demand_trace"].append(dict(demand_traces.get(i, {})))

            for pid, product_log in store_log["products"].items():
                product_log["inventory"].append(store.inventory.get(pid, 0))
                product_log["demand"].append(store.demand.get(pid, 0))
                product_log["sales"].append(store.sales.get(pid, 0))
                product_log["order_quantity"].append(order_decisions.get(pid, 0))
                product_log["outstanding_orders"].append(store.pending.get(pid, 0))
                product_log["promotion_status"].append(
                    "On Promotion" if pid in store.promotions else "Regular Price"
                )
                product_log["active_status"].append(pid in store.active_items)
                product_log["price"].append(
                    price_decisions.get(pid, store.prices.get(pid, 0.0))
                )
                product_log["revenue"].append(store.revenue.get(pid, 0.0))
                product_log["total_cost"].append(store.total_cost.get(pid, 0.0))
                product_log["holding_cost"].append(store.holding_cost.get(pid, 0.0))
                product_log["profit"].append(
                    store.revenue.get(pid, 0.0) - store.total_cost.get(pid, 0.0)
                )


# ---------------------------------------------------------------------------
# Module-level helpers (shared by Runner and Simulation)
# ---------------------------------------------------------------------------

def _event_payloads(events: list[WorldEvent]) -> list[dict[str, Any]]:
    """Flatten the active ``WorldEvent`` list for JSON-friendly logging."""
    return [
        {
            "type": event.event_type,
            "severity": event.severity,
            "regions": list(event.affected_regions),
            "duration": event.duration,
        }
        for event in events
    ]


def _dispatch_orders(
    store: Store,
    action: dict[str, Any],
    market: Market,
    event_engine: EventEngine,
) -> None:
    """Schedule a delivery callback for every positive-qty order."""
    orders = action.get("order", {})
    if not orders:
        return
    current_step = market.current_step()
    supply = market.market_state[store.region]["market_supply"]
    supply_factor = max(market.params.supply_factor_min, supply)
    for pid, qty in orders.items():
        if qty <= 0:
            continue
        base_lead = store.delivery_lags[pid]
        adjusted_lead = int(base_lead / supply_factor)
        arrival_time = current_step + adjusted_lead
        event_engine.schedule(
            event_type="order_arrival",
            delay=arrival_time,
            callback=_make_delivery_callback(store, pid, qty),
        )


def _process_demand(
    store: Store,
    action: dict[str, Any],
    market: Market,
) -> dict[str, int]:
    """Sample realised demand and settle accounting for one store."""
    prices = action.get("price", {})
    orders = action.get("order", {})
    traces: dict[str, int] = {}
    current_step = market.current_step()
    for pid in list(store.inventory.keys()):
        price = prices.get(pid, store.prices[pid])
        order_qty = orders.get(pid, 0)
        demand = market.sample_demand(
            pid, store, price, current_step=current_step
        )
        store.settle(pid, demand=demand, price=price, order_qty=order_qty)
        store.prices[pid] = price
        traces[pid] = demand
    return traces


def _make_delivery_callback(store: Store, pid: str, qty: int):
    """Bind ``(store, pid, qty)`` into a zero-arg callback for ``EventEngine``."""

    def _callback() -> None:
        store.deliver(pid, qty)

    return _callback


__all__ = ["Runner", "Simulation", "TickResult", "build_world"]
