"""Public ``Store`` (issue 05).

A ``Store`` is a single retail-agent in the simulation. It owns:

- a region tag plus operating parameters (capacity, holding rate, lead
  time, order fee, opening balance);
- per-product inventory, sales, demand, prices, and accounting counters
  (revenue, holding cost, order cost, total cost);
- a set of currently active SKUs plus an ``activation_tick`` map that
  drives the per-(store, product) freshness curve;
- a reference to an attached ``Policy`` (may be ``None`` for
  determinism-test fixtures that don't make decisions).

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

The accounting math (revenue, holding/order/total cost, balance
evolution, capacity-clamped deliveries, pending bookkeeping) is
preserved verbatim from ``StoreAgent``.
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
        # Keep the template handy for downstream lookups (and to make
        # the construction inputs introspectable on the live ``Store``).
        self.template = template
        # The per-store init RNG seed — captured for traceability;
        # consumed inside ``init_store_state`` via ``Random(init_seed)``.
        self.init_seed = init_seed
        # Attached decision-making policy (or ``None`` for skeleton runs).
        self.policy = policy
        # Catalog-wide freshness curve parameters. Plain-Store callers
        # (T4 unit tests) keep the no-op scalar default. The Runner
        # passes ``item_registry`` so per-Ware overrides
        # (issue 04) take precedence in ``freshness_multiplier``; the
        # scalar fields stay as a fallback for the registry-less path.
        self.freshness_alpha = float(freshness_alpha)
        self.freshness_decay = float(freshness_decay)
        # Optional registry — when supplied, the freshness multiplier
        # and the weighted initial-stock allocator both consult it.
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
        # Operating geography for the store. ``Market`` uses this to
        # route demand/supply state from the right region.
        self.region: str = state.region
        # Total units of inventory the store can hold across all SKUs.
        self.capacity = state.capacity
        # Cash balance — mutates each ``settle`` call.
        self.balance = state.balance
        # Fraction of capacity initially filled (cached so reports can
        # cite the realised value rather than re-sampling).
        self.init_stock_pct = state.init_stock_pct
        # Default lead time for new orders (per-product copy below).
        self.delivery_lag = state.delivery_lag
        # Per-step holding cost rate (per-product copy below).
        self.holding_rate = state.holding_rate
        # Fixed fee per non-zero order.
        self.order_fee = state.order_fee

        # Active items keep insertion order so step-by-step iteration
        # over ``active_items`` is deterministic without depending on
        # dict / set hash randomisation.
        self.active_items: list[str] = list(state.active_ids)
        # Step at which each product was last activated. Initial active
        # SKUs are deliberately *not* populated by ``init_store_state``
        # in baseline mode so the freshness curve evaluates to 1 at
        # step 0; ``"fresh"`` mode seeds them at ``τ=0``.
        self.activation_tick: dict[str, int] = dict(state.activation_tick)
        # Per-product on-hand units. Populated in the ``_register_item``
        # loop below from ``state.inventory``.
        self.inventory: dict[str, int] = {}
        # Units actually sold this step (≤ demand).
        self.sales: dict[str, int] = {}
        # Realised demand this step (may exceed inventory ⇒ lost sales).
        self.demand: dict[str, int] = {}
        # Per-product copy of the template lead time. Kept per-product
        # so a future per-SKU lead time override can be wired without
        # changing the runner.
        self.delivery_lags: dict[str, float] = {}
        # Per-product holding cost rate.
        self.holding_rates: dict[str, float] = {}
        # Units in transit per product. ``defaultdict(int)`` so deliver
        # paths can decrement without an explicit zero-init.
        self.pending: defaultdict[str, int] = defaultdict(int)
        # Currently active promotions: ``pid → {discount, duration, start_step}``.
        self.promotions: dict[str, Any] = {}
        # End-of-cooldown step per product (post-promo recovery).
        self.promo_cooldown: dict[str, int] = {}
        # Newly activated products that haven't yet placed their first
        # replenishment — used by the policy's "initial order" branch.
        self.needs_init_order: set[str] = set()
        # Current selling price per product (set initially to base_price).
        self.prices: dict[str, float] = {}
        # MSRP / authored base price per product. Frozen at registration so
        # the policy can compute price decisions against a stable reference
        # instead of compounding multiplicatively on ``self.prices``.
        self.base_prices: dict[str, float] = {}
        # Unit cost per product (used by holding/order cost math and as
        # the per-product price floor in the policy).
        self.costs: dict[str, float] = {}
        # Last-step holding cost per product (for the run log).
        self.holding_cost: dict[str, float] = {}
        # Last-step revenue per product.
        self.revenue: dict[str, float] = {}
        # Last-step order cost (qty × unit_cost) per product.
        self.order_cost: dict[str, float] = {}
        # Last-step total cost (holding + order + per-order fee).
        self.total_cost: dict[str, float] = {}

        for w in catalog:
            # Initialise every catalog SKU's bookkeeping, even those
            # inactive at step 0 (their inventory will simply be 0).
            self._register_item(
                w.product_id, w.base_price, w.unit_cost, state.inventory[w.product_id]
            )

    def _register_item(
        self, product_id: str, base_price: float, unit_cost: float, stock: int
    ) -> None:
        """Initialise per-product bookkeeping fields for ``product_id``."""
        self.inventory[product_id] = stock
        self.sales[product_id] = 0
        # Pull defaults from the per-store fields resolved by
        # ``init_store_state`` — copies make the per-product dicts
        # mutable independent of the per-store scalars.
        self.delivery_lags[product_id] = self.delivery_lag
        self.holding_rates[product_id] = self.holding_rate
        self.prices[product_id] = base_price
        self.base_prices[product_id] = base_price
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
        # Clamp negative demand to zero. Negative draws can come from
        # Gaussian shocks pushed below zero by the multiplier stack.
        demand = max(0, demand)
        self.demand[product_id] = demand
        # Lost sales surface as ``demand > sold`` in the run log.
        sold = min(demand, self.inventory[product_id])
        self.sales[product_id] = sold
        # Inventory decrement — ``max(0, ...)`` is belt-and-suspenders
        # given ``sold <= inventory`` above.
        self.inventory[product_id] = max(0, self.inventory[product_id] - sold)

        # Revenue is realised at the effective sell price (not base price).
        self.revenue[product_id] = sold * price
        # Holding cost on residual inventory: units × rate × unit_cost.
        self.holding_cost[product_id] = (
            self.inventory[product_id]
            * self.holding_rates[product_id]
            * self.costs[product_id]
        )
        # Order cost = qty ordered × unit_cost.
        self.order_cost[product_id] = order_qty * self.costs[product_id]

        # Fixed fee charged if we actually ordered something this step.
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
        # Remaining headroom across all SKUs (single shared capacity).
        space = self.capacity - sum(self.inventory.values())
        # Receive what fits; the rest is silently dropped (matches old behaviour).
        received = min(qty, space)
        self.inventory[product_id] += received
        # Pending count drops by the FULL dispatched ``qty`` even when
        # part of it was dropped. See the docstring for the rationale.
        self.pending[product_id] = max(0, self.pending.get(product_id, 0) - qty)

    def activate_item(self, product_id: str, current_step: int = 0) -> None:
        """Activate ``product_id`` and mark it for first replenishment.

        ``activation_tick[product_id]`` is overwritten with
        ``current_step`` so the freshness curve resets to ``τ = 0`` on
        every (re-)activation. ``deactivate_item`` deliberately leaves
        the entry alone — the next activation overwrites it.
        """
        # Avoid duplicate entries in the ordered active list.
        if product_id not in self.active_items:
            self.active_items.append(product_id)
        # Re-seed the freshness clock at ``current_step``.
        self.activation_tick[product_id] = current_step
        # Flag so the policy places an initial replenishment order next tick.
        self.needs_init_order.add(product_id)

    def deactivate_item(self, product_id: str) -> None:
        """Drop ``product_id`` from the active assortment if present."""
        if product_id in self.active_items:
            self.active_items.remove(product_id)

    def is_active(self, product_id: str) -> bool:
        """Return whether ``product_id`` is in the active assortment."""
        return product_id in self.active_items

    def freshness_multiplier(self, product_id: str, current_step: int) -> float:
        """Return the per-(store, product) freshness factor at ``current_step``.

        Products with no entry in ``activation_tick`` (initial active
        SKUs under baseline mode, or never-activated SKUs) return
        ``1.0`` — no hype boost, no penalty. Activated products evaluate
        ``FreshnessCurve`` with ``τ = current_step − activation_tick[product_id]``
        against the per-``Ware`` ``(α, β)`` resolved by ``ItemRegistry``
        when one is attached, else the catalog-wide scalar fallback.
        """
        # "No entry" ⇒ no freshness effect. This is what makes
        # baseline-mode initial actives skip the hype window.
        if product_id not in self.activation_tick:
            return 1.0
        # Ticks since (re-)activation.
        tau = current_step - self.activation_tick[product_id]
        if self.item_registry is not None:
            # Prefer per-Ware overrides resolved by the registry.
            alpha = self.item_registry.freshness_alpha(product_id)
            decay = self.item_registry.freshness_decay(product_id)
        else:
            # Registry-less path (T4 unit tests): use the catalog-wide
            # scalar fallback supplied at construction.
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
        # Per-product cross-correlation graph. ``registry`` is the
        # authoritative source when supplied; otherwise hand back empty
        # graphs so the policy's cross-price branch becomes a no-op.
        related: dict[str, list[tuple[str, float]]]
        if registry is not None:
            related = {pid: registry.related(pid) for pid in self.inventory}
        else:
            related = {pid: [] for pid in self.inventory}

        # Build a snapshot dict — every value is a *copy* of mutable
        # state so the policy can't accidentally mutate the store.
        return {
            "current_sim_step": step,
            "region": self.region,
            "inventory": dict(self.inventory),
            "active_products": list(self.active_items),
            "max_capacity": self.capacity,
            "outstanding_orders": dict(self.pending),
            "initial_order_needed": set(self.needs_init_order),
            "product_prices": dict(self.prices),
            "base_prices": dict(self.base_prices),
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
        # Policy returns a dict with keys: promotions, promotion_cooldown,
        # order, price, activate, deactivate.
        decisions = self.policy.decide(observation)

        # Reflect promotion plan back into store state.
        self.promotions = decisions.get("promotions", {})
        self.promo_cooldown = decisions.get("promotion_cooldown", {})

        # Add positive-qty orders to ``pending`` so the runner can
        # schedule delivery callbacks and other consumers see in-transit
        # counts. ``needs_init_order`` is cleared as soon as the
        # first-replenishment order is placed.
        for pid, qty in decisions.get("order", {}).items():
            if qty > 0:
                if pid in self.needs_init_order:
                    self.needs_init_order.remove(pid)
                self.pending[pid] += qty

        # Pass the step to activate_item so the freshness clock resets
        # to the actual tick rather than always to 0.
        current_step = int(observation.get("current_sim_step", 0))
        for pid in decisions.get("activate", []):
            self.activate_item(pid, current_step)
        for pid in decisions.get("deactivate", []):
            self.deactivate_item(pid)

        return decisions


__all__ = ["Store"]
