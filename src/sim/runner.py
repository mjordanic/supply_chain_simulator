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
"""

from __future__ import annotations

from random import Random
from typing import Any

from src.sim.event_engine import EventEngine
from src.sim.item_registry import ItemRegistry
from src.sim.market import Market
from src.sim.scenario import Scenario
from src.sim.store import Store


class Runner:
    """Drive a ``Scenario`` to a full ``RunLog``.

    Public entry points: ``Runner(scenario)`` builds the world, then
    ``run()`` returns the log. The ``stores`` attribute is exposed so
    determinism tests can read step-0 state pre-run.
    """

    def __init__(self, scenario: Scenario) -> None:
        # Source-of-truth artifact; held for later metadata queries.
        self.scenario = scenario
        # Single shared world RNG seeded from the scenario seed. Every
        # world-side stochastic draw goes through this stream.
        self.world_rng: Random = Random(scenario.world_seed)
        # Construction order matters for RNG bookkeeping. ItemRegistry first
        # so its lifecycle-distribution draws (if any) are stable; Market
        # carries the registry so ``sample_demand`` works without
        # rebinding; EventEngine last.
        self.item_registry = ItemRegistry(
            scenario.item_lifecycle, scenario.catalog, self.world_rng
        )
        self.market = Market(
            scenario.market,
            self.world_rng,
            scenario.start_date,
            registry=self.item_registry,
        )
        self.event_engine = EventEngine(scenario.disruption, self.world_rng)
        # One ``Store`` per ``StoreInstance``. Each receives its own
        # ``init_seed`` so the per-store init RNG stays independent of
        # the world stream.
        self.stores: list[Store] = [
            Store(
                s.template,
                s.init_seed,
                s.policy,
                scenario.catalog,
                freshness_alpha=self.item_registry.default_freshness_alpha,
                freshness_decay=self.item_registry.default_freshness_decay,
                item_registry=self.item_registry,
            )
            for s in scenario.stores
        ]

    def run(self) -> dict[str, Any]:
        """Execute the full simulation and return the accumulated run log."""
        # Pre-allocate the nested run-log skeleton so the per-step append
        # paths can stay dict-key-lookup-only (no defaultdicts).
        run_log = self._init_run_log()

        # Step-0 baseline snapshot: pre-tick state, no actions yet. Every
        # metric series therefore ends up with length n_steps + 1.
        self._log_state(run_log, actions={}, demand_traces=None)

        for _ in range(self.scenario.n_steps):
            # World ticks first; policy.decide observes the post-tick state.
            self.market.tick()
            event = self.event_engine.tick(self.market) #applies deliveries and disruption events to the market
            self.item_registry.tick()
            run_log["global"]["events"]["occurrences"].append(
                self._event_payload(event)
            )

            # Phase 1: gather one action per store.
            actions: dict[int, dict[str, Any]] = {}
            for i, store in enumerate(self.stores):
                obs = store.observe(
                    self.market.market_state,
                    self.market.current_step(),
                    self.item_registry,
                )
                actions[i] = store.decide(obs)

            # Phase 2: dispatch newly placed orders as scheduled callbacks.
            for i, store in enumerate(self.stores):
                self._dispatch_orders(store, actions[i])

            # Phase 3: realise demand for every (store, product) and
            # settle the accounting. ``demand_traces`` is logged for
            # observability — useful when debugging CRN drift.
            demand_traces: dict[int, dict[str, int]] = {}
            for i, store in enumerate(self.stores):
                demand_traces[i] = self._process_demand(store, actions[i])

            self._log_state(run_log, actions=actions, demand_traces=demand_traces)

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
                        # Static per-product values resolved once at
                        # construction (issues 04, 08). DataExporter
                        # reads these to populate the products parquet
                        # so downstream analysis can reproduce demand
                        # math (and verify the budgeted stock weights)
                        # without re-resolving Ware overrides.
                        "freshness_alpha": item.freshness_alpha,
                        "freshness_decay": item.freshness_decay,
                        "init_stock_share": item.init_stock_share,
                    }
                    for pid, item in self.item_registry.items.items()
                },
            },
            "stores": {},
        }
        # Per-store sub-tree. ``step0_*`` fields capture the pre-run
        # baseline so determinism tests can compare without re-running.
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

    @staticmethod
    def _event_payload(event) -> dict[str, Any] | None:
        """Flatten a ``WorldEvent`` for JSON-friendly logging. ``None`` passthrough."""
        if event is None:
            return None
        return {
            "type": event.event_type,
            "severity": event.severity,
            # ``list(...)`` so we don't smuggle a tuple/mutable ref into the log.
            "regions": list(event.affected_regions),
            "duration": event.duration,
        }

    def _dispatch_orders(self, store: Store, action: dict[str, Any]) -> None:
        """Schedule a delivery callback for every positive-qty order.

        Lead-time math is verbatim from ``SimulationRunner._submit_order``:
        ``adjusted_lead_time = int(base_lead_time / supply_factor)`` with a
        floor of ``params.supply_factor_min`` on the supply factor to avoid
        divide-by-zero.
        """
        # Pull the order plan out of the action dict. Empty ⇒ early exit.
        orders = action.get("order", {})
        if not orders:
            return
        current_step = self.market.current_step()
        # Region-level supply driving the lead-time adjustment.
        supply = self.market.market_state[store.region]["market_supply"]
        # Floor at ``supply_factor_min`` so a depressed supply can't produce a divide-by-zero.
        supply_factor = max(
            self.market.params.supply_factor_min,
            supply / self.market.params.supply_divisor,
        )
        for pid, qty in orders.items():
            if qty <= 0:
                continue
            # Base lead time per product (per-store copy).
            base_lead = store.delivery_lags[pid]
            # Lower supply ⇒ longer effective lead time.
            adjusted_lead = int(base_lead / supply_factor)
            arrival_time = current_step + adjusted_lead
            self.event_engine.schedule(
                event_type="order_arrival",
                delay=arrival_time,
                callback=_make_delivery_callback(store, pid, qty),
            )

    def _process_demand(
        self, store: Store, action: dict[str, Any]
    ) -> dict[str, int]:
        """Sample realised demand and settle accounting for one store.

        Iteration order is ``store.inventory`` (catalog order via the
        ``__init__`` registration loop), matching the verbatim port of
        ``SimulationRunner._process_demand``. Every product consumes one
        ``Market.sample_demand`` call — and therefore one ``world_rng``
        draw — regardless of whether it is in the active assortment.
        """
        # Effective per-product prices for this tick (fall back to the
        # store's current price if the policy didn't override).
        prices = action.get("price", {})
        orders = action.get("order", {})
        # Demand trace returned for logging.
        traces: dict[str, int] = {}
        current_step = self.market.current_step()
        # ``list(...)`` snapshots the key set — the loop body mutates
        # store state but never the inventory key set, so this is purely
        # defensive against accidental future changes.
        for pid in list(store.inventory.keys()):
            price = prices.get(pid, store.prices[pid])
            order_qty = orders.get(pid, 0)
            # One ``world_rng`` draw per iteration — load-bearing for CRN.
            demand = self.market.sample_demand(
                pid, store, price, current_step=current_step
            )
            store.settle(pid, demand=demand, price=price, order_qty=order_qty)
            # Reflect the realised effective price back into the store
            # so subsequent observations see the post-action state.
            store.prices[pid] = price
            traces[pid] = demand
        return traces

    def _log_state(
        self,
        run_log: dict[str, Any],
        actions: dict[int, dict[str, Any]],
        demand_traces: dict[int, dict[str, int]] | None,
    ) -> None:
        """Append one timestep of state to ``run_log``."""
        # Time axis — appended once per tick.
        run_log["global"]["time"]["simulation_step"].append(
            self.market.current_step()
        )
        run_log["global"]["time"]["simulation_date"].append(
            self.market.current_date()
        )

        # Per-region demand/supply snapshot.
        for region in self.market.regions:
            state = self.market.market_state[region]
            run_log["global"]["market_supply"][region].append(state["market_supply"])
            run_log["global"]["market_demand"][region].append(state["market_demand"])

        # Per-product lifecycle snapshot.
        for pid, item in self.item_registry.items.items():
            run_log["global"]["products"][pid]["lifecycle_stage"].append(
                item.lifecycle_stage
            )

        # Per-store metrics — balance, active count, demand trace,
        # plus a long per-product accounting tail.
        for i, store in enumerate(self.stores):
            store_log = run_log["stores"][i]
            action = actions.get(i, {})
            order_decisions = action.get("order", {})
            price_decisions = action.get("price", {})

            store_log["balance"].append(store.balance)
            store_log["active_product_count"].append(len(store.active_items))
            if demand_traces is None:
                # Step-0 baseline ⇒ no demand drawn yet.
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
                # Profit recomputed at log time so a future change to
                # ``Store.settle`` doesn't quietly desync this column.
                product_log["profit"].append(
                    store.revenue.get(pid, 0.0) - store.total_cost.get(pid, 0.0)
                )


def _make_delivery_callback(store: Store, pid: str, qty: int):
    """Bind ``(store, pid, qty)`` into a zero-arg callback for ``EventEngine``.

    A separate factory keeps each callback's closure independent — using a
    naked lambda inside a loop would capture loop variables by reference
    and every callback would end up calling ``store.deliver`` with the
    *last* iteration's pid and qty.
    """

    def _callback() -> None:
        store.deliver(pid, qty)

    return _callback


__all__ = ["Runner"]
