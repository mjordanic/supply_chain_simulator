"""Public ``Store`` (issue 05).

Replaces the issue-03 runner-internal ``_Store`` skeleton and the previous
``StoreAgent`` from ``src/agents/base_agent.py``. The accounting math
(revenue, holding/order/total cost, balance evolution, capacity-clamped
deliveries, pending bookkeeping) is preserved verbatim from
``StoreAgent``.

Two construction-shape changes vs. the old ``StoreAgent``:

1. The old two-step ``__init__`` + ``load_items`` is folded into a single
   ``Store(template, init_seed, policy, catalog)`` constructor. Initial
   active-SKU selection and stock allocation consume only a per-instance
   ``init_rng = Random(init_seed)`` — never ``world_rng`` or
   ``policy_rng``. This is what lets two stores sharing
   ``(template, init_seed)`` start step 0 bit-identical regardless of
   attached policy (T1 test 4 + T4 ``test_init_state_independent_of_policy``).
2. Stochastic template fields (any field on ``StoreTemplate`` typed as
   ``scalar | Distribution``) are sampled lazily here against the
   ``init_rng``, replacing the old ``init_params`` dict-broadcast.

The full integration pass (issue 07) wires per-step ``settle`` /
``deliver`` calls into the Runner; this module owns the per-store math
in isolation.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

from src.sim import freshness_curve
from src.sim.item_registry import ItemRegistry
from src.sim.policy import Policy
from src.sim.scenario import StoreTemplate, Ware
from src.sim.store_initializer import init_store_state


class Store:
    """One store's per-step state and accounting surface.

    Construction-time RNG: only ``Random(init_seed)``. World-tick math
    against ``world_rng`` and policy choices against ``policy_rng`` happen
    later, through the Runner. Two ``Store``s built from the same
    ``(template, init_seed, catalog)`` are bit-identical at step 0
    regardless of which ``Policy`` is attached.
    """

    def __init__(
        self,
        template: StoreTemplate,
        init_seed: int,
        policy: Policy | None,
        catalog: list[Ware],
        freshness_alpha: float = 0.0,
        freshness_decay: float = 1.0,
        item_registry: ItemRegistry | None = None,
    ) -> None:
        self.template = template
        self.init_seed = init_seed
        self.policy = policy
        # Catalog-wide freshness curve parameters. Plain-Store callers
        # (T4 unit tests) keep the no-op scalar default. The Runner
        # passes ``item_registry`` so per-Ware overrides
        # (issue 04) take precedence in ``freshness_multiplier``; the
        # scalar fields stay as a fallback for the registry-less path.
        self.freshness_alpha = float(freshness_alpha)
        self.freshness_decay = float(freshness_decay)
        self.item_registry = item_registry

        # Resolve all step-0 state through the pure ``init_store_state``
        # seam. The bit-identity contract — same ``(template, init_seed,
        # catalog, item_registry)`` ⇒ identical step-0 state regardless
        # of attached policy — is pinned by ``test_store_initializer``;
        # ``Store`` only fans the resolved values out into its
        # per-product bookkeeping dicts. The registry seam is what
        # routes per-``Ware`` ``init_stock_share`` weights (issue 08)
        # into the stock allocation; the registry-less path falls back
        # to the legacy even split.
        state = init_store_state(template, init_seed, catalog, item_registry)
        self.region: str = state.region
        self.capacity = state.capacity
        self.balance = state.balance
        self.init_stock_pct = state.init_stock_pct
        self.delivery_lag = state.delivery_lag
        self.holding_rate = state.holding_rate
        self.order_fee = state.order_fee

        # Active items keep insertion order so step-by-step iteration
        # over ``active_items`` is deterministic without depending on
        # dict / set hash randomisation.
        self.active_items: list[str] = list(state.active_ids)
        # Step at which each product was last activated. Initial active
        # SKUs are deliberately *not* populated by ``init_store_state``
        # today so the freshness curve evaluates to 1 at step 0
        # (baseline-equivalent behaviour; explicit
        # ``init_freshness="fresh"`` mode lands in issue 07).
        self.activation_tick: dict[str, int] = dict(state.activation_tick)
        self.inventory: dict[str, int] = {}
        self.sales: dict[str, int] = {}
        self.demand: dict[str, int] = {}
        self.delivery_lags: dict[str, float] = {}
        self.holding_rates: dict[str, float] = {}
        self.pending: defaultdict[str, int] = defaultdict(int)
        self.promotions: dict[str, Any] = {}
        self.promo_cooldown: dict[str, int] = {}
        self.needs_init_order: set[str] = set()
        self.prices: dict[str, float] = {}
        self.costs: dict[str, float] = {}
        self.holding_cost: dict[str, float] = {}
        self.revenue: dict[str, float] = {}
        self.order_cost: dict[str, float] = {}
        self.total_cost: dict[str, float] = {}

        for w in catalog:
            self._register_item(
                w.product_id, w.base_price, w.unit_cost, state.inventory[w.product_id]
            )

    def _register_item(
        self, product_id: str, base_price: float, unit_cost: float, stock: int
    ) -> None:
        """Initialise per-product bookkeeping fields for ``product_id``."""
        self.inventory[product_id] = stock
        self.sales[product_id] = 0
        self.delivery_lags[product_id] = self.delivery_lag
        self.holding_rates[product_id] = self.holding_rate
        self.prices[product_id] = base_price
        self.costs[product_id] = unit_cost

    def settle(
        self,
        product_id: str,
        demand: int,
        price: float,
        order_qty: int,
    ) -> tuple[int, float, float]:
        """Settle one product's demand and accounting for one step.

        Verbatim port of ``StoreAgent.settle``. Negative demand is clamped
        to zero. Sales are min(demand, inventory). Holding cost is charged
        on residual inventory; order cost is order_qty × unit_cost. A
        fixed ``order_fee`` applies iff ``order_qty > 0``. Balance is
        updated by ``revenue − total_cost``.
        """
        demand = max(0, demand)
        self.demand[product_id] = demand
        sold = min(demand, self.inventory[product_id])
        self.sales[product_id] = sold
        self.inventory[product_id] = max(0, self.inventory[product_id] - sold)

        self.revenue[product_id] = sold * price
        self.holding_cost[product_id] = (
            self.inventory[product_id]
            * self.holding_rates[product_id]
            * self.costs[product_id]
        )
        self.order_cost[product_id] = order_qty * self.costs[product_id]

        fee = self.order_fee if order_qty > 0 else 0
        self.total_cost[product_id] = (
            self.holding_cost[product_id] + self.order_cost[product_id] + fee
        )
        self.balance = (
            self.balance + self.revenue[product_id] - self.total_cost[product_id]
        )

        return sold, self.revenue[product_id], self.balance

    def deliver(self, product_id: str, qty: int) -> None:
        """Receive a scheduled delivery, clamping at remaining capacity.

        Verbatim port of ``StoreAgent.deliver``: pending decrements by the
        dispatched ``qty`` (not the capacity-clamped received amount), so
        an over-delivered order is no longer in-flight even if part of it
        was dropped at the receiving dock.
        """
        space = self.capacity - sum(self.inventory.values())
        received = min(qty, space)
        self.inventory[product_id] += received
        self.pending[product_id] = max(0, self.pending.get(product_id, 0) - qty)

    def activate_item(self, product_id: str, current_step: int = 0) -> None:
        """Activate ``product_id`` and mark it for first replenishment.

        ``activation_tick[product_id]`` is overwritten with
        ``current_step`` so the freshness curve resets to ``τ = 0`` on
        every (re-)activation. ``deactivate_item`` deliberately leaves
        the entry alone — the next activation overwrites it.
        """
        if product_id not in self.active_items:
            self.active_items.append(product_id)
        self.activation_tick[product_id] = current_step
        self.needs_init_order.add(product_id)

    def deactivate_item(self, product_id: str) -> None:
        """Drop ``product_id`` from the active assortment if present."""
        if product_id in self.active_items:
            self.active_items.remove(product_id)

    def is_active(self, product_id: str) -> bool:
        return product_id in self.active_items

    def freshness_multiplier(self, product_id: str, current_step: int) -> float:
        """Return the per-(store, product) freshness factor at ``current_step``.

        Products with no entry in ``activation_tick`` (initial active
        SKUs, or never-activated SKUs) return ``1.0`` — no hype boost,
        no penalty. Activated products evaluate ``FreshnessCurve`` with
        ``τ = current_step − activation_tick[product_id]`` against the
        per-``Ware`` ``(α, β)`` resolved by ``ItemRegistry`` when one is
        attached, else the catalog-wide scalar fallback.
        """
        if product_id not in self.activation_tick:
            return 1.0
        tau = current_step - self.activation_tick[product_id]
        if self.item_registry is not None:
            alpha = self.item_registry.freshness_alpha(product_id)
            decay = self.item_registry.freshness_decay(product_id)
        else:
            alpha = self.freshness_alpha
            decay = self.freshness_decay
        return freshness_curve.multiplier(alpha, decay, tau)

    def observe(
        self,
        market_state: Mapping[str, Mapping[str, float]],
        step: int,
        registry: ItemRegistry | None = None,
    ) -> dict[str, Any]:
        """Build the full observation payload consumed by ``BaselinePolicy.decide``.

        Field set is the contract documented inline on ``BaselinePolicy.decide``.
        ``registry`` is required when the attached policy uses cross-product
        relationships; pass ``None`` (default) for skeleton flows where the
        related-products graph is not consulted.
        """
        related: dict[str, list[tuple[str, float]]]
        if registry is not None:
            related = {pid: registry.related(pid) for pid in self.inventory}
        else:
            related = {pid: [] for pid in self.inventory}

        return {
            "current_sim_step": step,
            "region": self.region,
            "inventory": dict(self.inventory),
            "active_products": list(self.active_items),
            "max_capacity": self.capacity,
            "outstanding_orders": dict(self.pending),
            "initial_order_needed": set(self.needs_init_order),
            "product_prices": dict(self.prices),
            "unit_costs": dict(self.costs),
            "related_products": related,
            "balance": self.balance,
            "sales": dict(self.sales),
            "promotions": dict(self.promotions),
            "promotion_cooldown": dict(self.promo_cooldown),
            "market_state": dict(market_state[self.region]),
        }

    def decide(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Delegate to the attached policy and reconcile its decisions.

        Verbatim port of ``StoreAgent.decide``. ``policy=None`` returns an
        empty action dict (skeleton-mode for determinism tests).
        """
        if self.policy is None:
            return {}
        decisions = self.policy.decide(observation)

        self.promotions = decisions.get("promotions", {})
        self.promo_cooldown = decisions.get("promotion_cooldown", {})

        for pid, qty in decisions.get("order", {}).items():
            if qty > 0:
                if pid in self.needs_init_order:
                    self.needs_init_order.remove(pid)
                self.pending[pid] += qty

        current_step = int(observation.get("current_sim_step", 0))
        for pid in decisions.get("activate", []):
            self.activate_item(pid, current_step)
        for pid in decisions.get("deactivate", []):
            self.deactivate_item(pid)

        return decisions


__all__ = ["Store"]
