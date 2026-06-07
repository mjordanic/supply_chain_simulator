"""Policy ABC + ``HeuristicPolicy`` + textbook reorder-policy family.

A ``Policy`` is the decision-making brain attached to a ``Store``. It
receives an ``Observation`` dict each tick and returns an ``Action``
dict with four required keys (``order``, ``price``, ``activate``,
``deactivate``) plus an optional ``promotions`` map. Post-promo
cooldowns and the first-order flag are policy-internal state on
``HeuristicPolicy`` (``promo_cooldown`` / ``needs_init_order``) — they
no longer round-trip through Store.

Each ``Policy`` instance owns its own ``policy_rng`` seeded from
``policy_seed``. World streams (``world_rng``) and policy streams never
share an RNG, so swapping a policy on a Scenario does not perturb the
world.

``HeuristicPolicy`` is the heuristic kitchen-sink demonstrator ported verbatim
from the deleted ``src/agents/policy.py``. The behaviour (cooldown gating,
capacity respect, price floors at unit cost, periodic catalog review)
is preserved; what changed is the hyperparameter pathway. The old
``init_params`` / ``live_params`` broadcasting dicts (encoded across
two methods) are gone in favour of plain keyword arguments. Stochastic
choices (``randint`` for promo duration / order-cooldown jitter,
``shuffle`` of the inactive product list during catalog review,
``promo_discount`` sampling) all flow through ``self.policy_rng`` —
never the global ``random`` module and never ``world_rng``.

Textbook reorder-policy family:

- ``TextbookReorderPolicy`` — abstract base. Implements the shared
  observe→pilot→trigger→allocate pipeline via a ``decide`` method.
  Subclasses implement ``_trigger`` and ``_quantity`` hooks (or, when the
  trigger needs side-channel information like the periodic schedule,
  ``_trigger_with_safety`` / ``_quantity_with_levels`` directly).
- ``OrderUpToPolicy`` — (s,S) continuous review. Becomes the canonical
  CRN comparison anchor for RL.
- ``ReorderPointPolicy`` — (s,Q) continuous review with a fixed (or
  rate-derived) order quantity.
- ``PeriodicOrderUpToPolicy`` — (R,S) periodic review.
- ``PeriodicReorderPolicy`` — (R,s,S) periodic review with a
  reorder-point gate.

All four concrete policies share a single demand-units framing so the
reorder formulas are scale-invariant in capacity:

    effective_safety_ticks = round(safety_lead_pct_of_lag × delivery_lag)
    s = (delivery_lag + effective_safety_ticks) × rate
    S = s + cover_horizon_ticks × rate

where ``rate`` is the censored-sales rate estimate over the recent
``demand_window = max(1, delivery_lag)`` ticks.

Three private module-level helpers extracted for unit-testability:

- ``_estimate_rate`` — censored-sales rate estimator (windowed mean of
  realised sales; censoring detection lives in ``_has_stockout`` and is
  applied by the caller via the effective safety horizon).
- ``_has_stockout`` — detects whether a pid had any stockout tick in
  the recent window (sales > 0 AND sales == inv-before-settle).
- ``_allocate_two_pass_fair_share`` — two-pass fair-share + water-fill
  + integer mop-up allocator. Splits a shared capacity / cash pool
  across multiple SKUs in an iteration-order-independent way, then
  applies a per-non-pilot ``min_qty`` floor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from random import Random
from typing import Any, Mapping

from src.sim.distributions import Distribution


class Policy(ABC):
    """Base class for policies. Owns a per-instance ``policy_rng``."""

    def __init__(self, policy_seed: int | None = None) -> None:
        # Private RNG seeded from the caller's seed. NEVER shared with
        # ``world_rng`` — that would couple policy choices to world
        # stochasticity and break paired comparisons.
        self.policy_rng: Random = Random(policy_seed)

    @abstractmethod
    def decide(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Return an action dict for one simulation step."""


# ---------------------------------------------------------------------------
# NodePolicy ABC family (multi-echelon graph engine)
#
# ``NodePolicy`` is the base for all graph-node decision-makers. It mirrors
# the structure of ``Policy`` (owns ``policy_rng`` seeded from
# ``policy_seed``) but is *not* a subclass of ``Policy`` — the two
# families represent orthogonal simulation layers and should not be mixed.
#
# Three type-paired subclass ABCs enforce the per-node-type decide signature:
#   - ``FactoryPolicy``       — produces goods, sets list price
#   - ``IntermediatePolicy``  — orders from upstream, sets prices + min orders
#   - ``DemandSinkPolicy``    — buys from upstream for the bound product
# ---------------------------------------------------------------------------


class NodePolicy(ABC):
    """Base class for graph-node policies.

    Each instance owns its own ``policy_rng`` seeded from ``policy_seed``.
    Swapping a policy on a node must NOT perturb ``world_rng`` or
    ``allocation_rng`` (determinism invariant from ADR 0016).
    """

    def __init__(self, policy_seed: int | None = None) -> None:
        self.policy_seed: int | None = policy_seed
        self.policy_rng: Random = Random(policy_seed)

    @abstractmethod
    def decide(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Return a decision dict. Concrete signature varies by node type."""


class FactoryPolicy(NodePolicy, ABC):
    """Policy ABC for ``FactoryNode``.

    ``decide`` receives a factory observation and returns production
    instructions. ``list_price`` must equal ``unit_cost`` per ADR 0013 —
    concrete implementations should not deviate from that.
    """

    @abstractmethod
    def decide(self, obs_factory: Mapping[str, Any]) -> dict[str, Any]:
        """Return ``{"produce_qty": int, "list_price": float}``."""


class IntermediatePolicy(NodePolicy, ABC):
    """Policy ABC for ``IntermediateNode``.

    ``decide`` receives an intermediate observation *and* a live
    ``central_table`` snapshot, enabling routing decisions that account
    for upstream supply availability and prices.
    """

    @abstractmethod
    def decide(
        self, obs_intermediate: Mapping[str, Any], central_table: Any
    ) -> dict[str, Any]:
        """Return order, list_price, and min_order_imposed decisions.

        Returns
        -------
        dict with keys:
            ``order``             — ``{pid: [(supplier_id, qty), ...]}``
            ``list_price``        — ``{pid: float}``
            ``min_order_imposed`` — ``{pid: int}``
        """


class DemandSinkPolicy(NodePolicy, ABC):
    """Policy ABC for ``DemandSinkNode``.

    ``decide`` receives a sink observation and a live ``central_table``
    snapshot, then returns a buy plan for the sink's bound product.
    """

    @abstractmethod
    def decide(
        self, obs_sink: Mapping[str, Any], central_table: Any
    ) -> dict[str, Any]:
        """Return ``{"buy": [(supplier_id, qty), ...]}``."""


def _maybe_sample(value: Any, rng: Random) -> Any:
    """Sample a ``Distribution`` value once; pass through scalars."""
    if isinstance(value, Distribution):
        return value.sample(rng)
    return value


class _HeuristicPolicyOld(Policy):
    """Heuristic kitchen-sink demonstrator — retired.

    Dead code retained to keep the diff size manageable. The class is no
    longer exported or used anywhere in the production code.

    Original docstring follows:

    Heuristic kitchen-sink demonstrator policy with kwargs hyperparameters.

    Observation contract (every step, supplied by ``Store.observe``):

    - ``current_sim_step`` (int): current simulation tick.
    - ``inventory`` (dict[pid, int]): on-hand units per product.
    - ``max_capacity`` (int | float): total store capacity.
    - ``outstanding_orders`` (dict[pid, int]): in-transit pending qty.
    - ``active_products`` (Iterable[pid]): products in the active assortment.
    - ``product_prices`` (dict[pid, float]): current realised prices per pid.
    - ``base_prices`` (dict[pid, float]): immutable MSRP reference prices
      per pid. The policy adjusts off this fixed reference rather than
      its own previous-tick decision, so the markdown / markup factors
      do not compound multiplicatively.
    - ``unit_costs`` (dict[pid, float]): unit costs per pid (price floor).
    - ``related_products`` (dict[pid, list[(pid, float)]]): cross-product
      correlation graph used by the cross-price adjustment.
    - ``balance`` (float): store balance, divides into per-product budget.
    - ``sales`` (dict[pid, int]): units sold this step (for trend log).
    - ``promotions`` (dict[pid, dict]): currently active promotions.

    The policy owns the post-promo cooldown map and the first-order
    flag for newly-activated products as ``self.promo_cooldown`` and
    ``self.needs_init_order`` — neither field passes through the
    observation any more.

    All stochastic choices (``randint``, ``shuffle``, ``promo_discount``
    sampling) consume ``self.policy_rng`` only.
    """

    def __init__(
        self,
        *,
        policy_seed: int | None = None,
        min_qty: int = 10,
        init_qty_factor: float = 0.3,
        promo_len: int | Distribution = 5,
        promo_cd_len: int = 10,
        review_interval: int = 5,
        promo_threshold: float = 0.7,
        target_active_count: int = 10,
        active_margin: int = 2,
        max_activations_per_review: int | None = None,
        slow_sales_limit: int = 5,
        stock_lo_ratio: float = 0.2,
        stock_hi_ratio: float = 0.6,
        price_up_factor: float = 1.1,
        price_down_factor: float = 0.9,
        history_window: int = 10,
        slow_mover_lookback_factor: int = 6,
        trend_threshold: float = 0.05,
        cross_price_adj: float = 0.05,
        max_history: int = 100,
        inactive_price_factor: float = 0.5,
        reorder_factor: float = 0.3,
        qty_factor: float = 0.5,
        order_cd_len: int = 10,
        order_cd_jitter: float = 0.3,
        promo_discount: float | Distribution = 0.7,
    ) -> None:
        super().__init__(policy_seed=policy_seed)

        # Promo duration. Scalar ⇒ fixed length; Distribution ⇒ sampled
        # per-promo via ``policy_rng``. Sample is ``int(...)``-cast at use site.
        self.promo_len = promo_len

        # Minimum order quantity — anything below this is rounded down to 0.
        self.min_qty = min_qty
        # Multiplier on free-space when sizing the *first* order after activation.
        self.init_qty_factor = init_qty_factor
        # Cooldown length applied after a promotion expires.
        self.promo_cd_len = promo_cd_len
        # How often (in steps) to run the catalog review pass.
        self.review_interval = review_interval
        # Stock ratio above which a product is considered "heavy" and
        # becomes a promotion candidate.
        self.promo_threshold = promo_threshold
        # Target active-SKU count — drives the catalog-review activation branch.
        self.target_active_count = target_active_count
        # Tolerance band around ``target_active_count``. Deactivation may
        # drop the active count down to ``target - margin`` so a wave of
        # dud-drops isn't blocked by the target floor; activation only
        # kicks in once the count slips below ``target - margin`` so the
        # policy doesn't churn on ±1 deviations. Activation always aims
        # back at the centre, ``target_active_count``.
        self.active_margin = active_margin
        # Per-pass cap on activations. ``None`` ⇒ fill all the way to
        # ``target`` in a single pass (bounded by available
        # growth-potential inactives). A finite cap is useful when
        # capacity / budget can't absorb a burst of init orders without
        # each one falling below ``min_qty``.
        self.max_activations_per_review = max_activations_per_review
        # Threshold for "no movement in window" deactivation.
        self.slow_sales_limit = slow_sales_limit
        # Stock-ratio band used by the dynamic pricing decision.
        self.stock_lo_ratio = stock_lo_ratio
        self.stock_hi_ratio = stock_hi_ratio
        # Pricing factors applied when stock/trend signals fire.
        self.price_up_factor = price_up_factor
        self.price_down_factor = price_down_factor
        # Window (in steps) used to compute the rolling sales trend.
        self.history_window = history_window
        # Multiplier on ``history_window`` for the extended lookback used to
        # confirm a slow mover when the product has been fully stocked out
        # across the recent window.
        self.slow_mover_lookback_factor = slow_mover_lookback_factor
        # Trend-magnitude threshold for triggering price moves.
        self.trend_threshold = trend_threshold
        # Per-related-product price adjustment magnitude.
        self.cross_price_adj = cross_price_adj
        # Max length of the per-product sales/stock history deques.
        self.max_history = max_history
        # Multiplier on base price for *inactive* products (kept at clearance).
        self.inactive_price_factor = inactive_price_factor
        # Stock-position threshold for triggering re-orders.
        self.reorder_factor = reorder_factor
        # Target stock-fill factor for the periodic re-order.
        self.qty_factor = qty_factor
        # Length and jitter of per-product order cooldown.
        self.order_cd_len = order_cd_len
        self.order_cd_jitter = order_cd_jitter
        # Discount applied during promotions. Can be a Distribution for
        # randomised markdowns; sampled via ``policy_rng``.
        self.promo_discount = promo_discount

        # Bounded per-product histories. ``deque(maxlen=…)`` is O(1)
        # append + automatic drop of the oldest entry.
        self.sales_log: dict[str, deque[int]] = {}
        self.stock_log: dict[str, deque[int]] = {}
        # Per-product "no-orders-before-this-step" cooldown map.
        self.order_cd: dict[str, int] = {}
        # Post-promotion cooldown end-step per product. Owned by the
        # policy — Store no longer round-trips this through observation.
        self.promo_cooldown: dict[str, int] = {}
        # Newly-activated products that have not yet placed their first
        # replenishment order. Populated when the policy emits an
        # ``activate`` decision; cleared when the corresponding order is
        # placed in the same tick. Previously lived on Store as
        # ``needs_init_order`` and travelled back through the observation.
        self.needs_init_order: set[str] = set()

    def decide(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Build and return one action dict for the given observation."""
        # Update rolling history before consulting it.
        self._record_sales(observation)
        self._record_stock(observation)

        # Pull observation fields into locals — keeps the four planner
        # calls readable and avoids dict-lookup repetition.
        promos = dict(observation["promotions"])
        step = observation["current_sim_step"]
        inventory = observation["inventory"]
        capacity = observation["max_capacity"]
        pending = observation["outstanding_orders"]
        active_items = set(observation["active_products"])
        prices = observation["product_prices"]
        # ``base_prices`` falls back to the (possibly stale) current price
        # when the observation predates this field — keeps older Store
        # call sites working without forcing a coordinated upgrade.
        base_prices = observation.get("base_prices", prices)
        related = observation["related_products"]
        balance = observation["balance"]
        costs = observation["unit_costs"]

        # Catalog review runs FIRST so activations / deactivations take
        # effect within the same tick: a newly-activated SKU places its
        # initial order this tick instead of waiting a cycle, and a SKU
        # being dropped doesn't burn an order or active-tier price
        # before being marked down to clearance.
        activate, deactivate = self._review_catalog(step, active_items)
        active_items = (active_items | set(activate)) - set(deactivate)
        # Flag activations for the init-order branch in ``_plan_orders``;
        # consumed and discarded in the same tick by that planner.
        for pid in activate:
            self.needs_init_order.add(pid)

        # Remaining sub-passes operate on the updated catalog.
        # ``_plan_promos`` mutates ``self.promo_cooldown`` and
        # ``_plan_orders`` mutates ``self.needs_init_order``.
        promos = self._plan_promos(
            promos, step, inventory, capacity, active_items
        )
        orders = self._plan_orders(
            inventory, pending, capacity, active_items, balance, costs, step
        )
        pricing = self._plan_prices(
            inventory, base_prices, capacity, active_items, promos, related, costs
        )

        return {
            "promotions": promos,
            "order": orders,
            "price": pricing,
            "activate": activate,
            "deactivate": deactivate,
        }

    @staticmethod
    def _free_capacity(
        inventory: Mapping[str, int],
        pending: Mapping[str, int],
        capacity: float,
    ) -> int:
        """Capacity headroom = total - (on-hand + in-transit). Floored at 0."""
        return max(0, int(capacity - (sum(inventory.values()) + sum(pending.values()))))

    def _record_sales(self, observation: Mapping[str, Any]) -> None:
        """Append last-step sales to each product's rolling history."""
        for pid, qty in observation["sales"].items():
            # ``setdefault`` materialises a maxlen-bounded deque on first sight.
            log = self.sales_log.setdefault(pid, deque(maxlen=self.max_history))
            log.append(qty)

    def _record_stock(self, observation: Mapping[str, Any]) -> None:
        """Append current stock to each product's rolling history."""
        for pid, stock in observation["inventory"].items():
            log = self.stock_log.setdefault(pid, deque(maxlen=self.max_history))
            log.append(stock)

    def _plan_promos(
        self,
        promos: dict[str, dict[str, Any]],
        step: int,
        inventory: Mapping[str, int],
        capacity: float,
        active_items: set[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Expire reached-duration promos, prune cooldowns, start new ones on heavy stock.

        Mutates ``self.promo_cooldown`` in place — the cooldown map is
        pure policy state and no longer round-trips through the Store
        observation.
        """
        # Promotion windows are step-based; once duration is reached, expire.
        expired = [pid for pid, p in promos.items() if step - p["start_step"] >= p["duration"]]
        for pid in expired:
            del promos[pid]
            # Set the cooldown end so the same product can't go on
            # promo again until ``promo_cd_len`` steps have passed.
            self.promo_cooldown[pid] = step + self.promo_cd_len

        # Drop already-elapsed cooldowns.
        self.promo_cooldown = {pid: s for pid, s in self.promo_cooldown.items() if s > step}

        # Per-SKU "slice" — the fair share of capacity if it were evenly
        # divided among the active assortment. Treating "heavy stock" as a
        # fraction of the SLICE (not the whole capacity) is what makes the
        # promo threshold meaningful in many-SKU stores.
        n_active = max(1, len(active_items) if active_items is not None else 1)
        slice_size = max(1.0, capacity / n_active)

        for pid, stock in inventory.items():
            # Only consider non-promoted, non-cooldown products.
            if pid not in promos and pid not in self.promo_cooldown:
                # "Heavy stock" gate: stock above promo_threshold × per-SKU slice.
                if stock > slice_size * self.promo_threshold:
                    # Sample (or pass through) the promo discount.
                    discount = float(_maybe_sample(self.promo_discount, self.policy_rng))
                    # Sample (or pass through) the promo duration; cast to int.
                    duration = int(_maybe_sample(self.promo_len, self.policy_rng))
                    promos[pid] = {
                        "discount": discount,
                        "duration": duration,
                        "start_step": step,
                    }
        return promos

    def _plan_orders(
        self,
        inventory: Mapping[str, int],
        pending: Mapping[str, int],
        capacity: float,
        active_items: set[str],
        balance: float,
        costs: Mapping[str, float],
        step: int,
    ) -> dict[str, int]:
        """Decide replenishment qty per product, respecting capacity + cooldown + budget.

        Reads ``self.needs_init_order`` (policy-owned) for the first-order
        branch and clears the flag for every pid that places a positive
        order this tick.
        """
        orders: dict[str, int] = {}
        # Running headroom — decremented as we allocate qty per pid.
        space = self._free_capacity(inventory, pending, capacity)

        for pid in inventory.keys():
            # Cooldown gate: skip if we recently placed an order for this pid.
            if pid in self.order_cd and step < self.order_cd[pid]:
                orders[pid] = 0
                continue

            # Only order for actively-stocked products, with headroom,
            # and a non-empty active assortment (avoid div-by-zero below).
            if pid in active_items and space > self.min_qty and len(active_items) > 0:
                # Stock position = on-hand + in-transit.
                position = inventory[pid] + pending.get(pid, 0)
                # Per-product budget = (balance / #active) / unit_cost.
                max_qty = max(0, int(balance / len(active_items) / costs[pid]))

                if pid in self.needs_init_order:
                    # First-time order after activation: opportunistic fill.
                    slice_size = capacity / max(1, len(active_items)) if capacity else 0.0
                    qty = self._initial_order(space, max_qty, slice_size)
                elif position <= self.reorder_factor * (capacity / len(active_items)):
                    # Steady-state re-order: top up to ``qty_factor × per-SKU share``.
                    target = self.qty_factor * capacity / len(active_items)
                    qty = self._reorder_qty(position, space, max_qty, target)
                else:
                    qty = 0

                if qty > 0:
                    # Stagger orders with a jitter so the policy doesn't
                    # produce perfectly periodic spikes.
                    jitter_bound = max(0, int(self.order_cd_len * self.order_cd_jitter))
                    jitter = self.policy_rng.randint(-jitter_bound, jitter_bound)
                    self.order_cd[pid] = step + self.order_cd_len + jitter
                    # First-order flag is consumed exactly once.
                    self.needs_init_order.discard(pid)

                qty = int(qty)
                orders[pid] = qty
                space = max(0, space - qty)
            else:
                orders[pid] = 0

        return orders

    def _initial_order(
        self, space: int, max_qty: int, slice_size: float | None = None
    ) -> int:
        """Opportunistic first-order qty: fill init_qty_factor × space, capped by budget.

        Also capped at the per-SKU "fair slice" when supplied, so a fresh
        SKU in a many-SKU store doesn't gobble up enough headroom to
        starve the rest of the assortment.
        """
        qty = min(max_qty, int(space * self.init_qty_factor))
        if slice_size is not None:
            qty = min(qty, int(slice_size * self.init_qty_factor))
        # Sub-min orders are folded to zero (the fixed fee dominates).
        return qty if qty >= self.min_qty else 0

    def _reorder_qty(self, position: int, space: int, max_qty: int, target: float) -> int:
        """Top-up qty: ``target − position``, capped by free space and budget."""
        qty = min(max_qty, int(target - position))
        qty = min(qty, space)
        return qty if qty >= self.min_qty else 0

    def _plan_prices(
        self,
        inventory: Mapping[str, int],
        base_prices: Mapping[str, float],
        capacity: float,
        active_items: set[str],
        promos: Mapping[str, Mapping[str, Any]],
        related: Mapping[str, list[tuple[str, float]]],
        costs: Mapping[str, float],
    ) -> dict[str, float]:
        """Per-product price decision: dynamic factor for actives, clearance for inactives.

        Reference is the immutable ``base_prices[pid]`` (MSRP), NOT the
        last-tick realised price. Reading the realised price made each
        markdown / markup compound multiplicatively, so a sequence of
        moderate stock-heavy ticks dragged the price toward the unit-cost
        floor in ~10 ticks; reading the base price keeps the policy's
        decisions on an additive ladder around MSRP.
        """
        pricing: dict[str, float] = {}
        for pid in inventory.keys():
            base = base_prices[pid]
            cost = costs[pid]
            if pid in active_items:
                # Combine stock/trend/cross/promo factors into one multiplier.
                factor = self._compute_price_factor(
                    pid, inventory, capacity, active_items, promos, related
                )
                decision = base * factor
            else:
                # Inactive ⇒ clearance pricing relative to MSRP.
                decision = self.inactive_price_factor * base
            # Guardrail: never price below unit cost.
            pricing[pid] = max(cost, decision)
        return pricing

    def _compute_price_factor(
        self,
        pid: str,
        inventory: Mapping[str, int],
        capacity: float,
        active_items: set[str],
        promos: Mapping[str, Mapping[str, Any]],
        related: Mapping[str, list[tuple[str, float]]],
    ) -> float:
        """Aggregate stock+trend+cross+promo signals into one multiplicative factor."""
        # Rolling sales trend for the focal product.
        trend = self._compute_sales_trend(pid)
        # Stock ratio relative to the per-SKU "fair slice" of capacity.
        # Using ``inventory[pid] / capacity`` makes the ratio vanishingly
        # small for any single SKU in a many-SKU store, so the
        # ``ratio > stock_hi_ratio`` markdown branch never fires and
        # ``ratio < stock_lo_ratio`` is always true. Dividing by
        # ``capacity / n_active`` instead gives a meaningful "how full is
        # this SKU's bucket?" signal regardless of catalog size.
        n_active = max(1, len(active_items))
        slice_size = max(1.0, capacity / n_active) if capacity else 0.0
        ratio = inventory[pid] / slice_size if slice_size else 0.0

        if trend > self.trend_threshold or ratio < self.stock_lo_ratio:
            # Strong demand or low stock ⇒ raise price. Mirrors the
            # markdown branch's OR semantics so markup/markdown fire on
            # symmetric trigger breadth — previously the AND form made
            # the markup branch fire only on the narrow trend∧stock
            # intersection, biasing realised prices below MSRP.
            factor = self.price_up_factor
        elif trend < -self.trend_threshold or ratio > self.stock_hi_ratio:
            # Cooling demand or heavy stock ⇒ markdown.
            factor = self.price_down_factor
        else:
            factor = 1.0

        # Per-related-product cross-adjustment. Active + in-inventory only.
        # Cross-related ratio uses the same per-SKU slice as the focal
        # SKU above — using ``capacity`` directly made the ratio tiny in
        # many-SKU stores and pinned this branch to always-fire-down,
        # systematically pushing prices toward the cost floor.
        for rel_id, corr in related.get(pid, []):
            if rel_id in active_items and rel_id in inventory:
                rel_ratio = inventory[rel_id] / slice_size if slice_size else 0.0
                if rel_ratio > self.stock_hi_ratio:
                    # Related over-stocked ⇒ small price up on focal.
                    factor *= 1 + self.cross_price_adj * corr
                elif rel_ratio < self.stock_lo_ratio:
                    # Related under-stocked ⇒ small price down on focal.
                    factor *= 1 - self.cross_price_adj * corr

        # Active promo? Apply its discount on top.
        if pid in promos:
            factor *= promos[pid]["discount"]

        return factor

    def _compute_sales_trend(self, pid: str) -> float:
        """Normalised rolling first-difference of sales over the recent window."""
        log = self.sales_log.get(pid)
        if log is None or len(log) < self.history_window:
            return 0.0
        # Slice the tail window from the deque.
        recent = list(log)[-self.history_window:]
        n = len(recent)
        # Sum of consecutive differences.
        total_change = sum(recent[i] - recent[i - 1] for i in range(1, n))
        avg_change = total_change / (n - 1)
        avg_sales = sum(recent) / n
        # Normalise so units cancel; zero-avg-sales ⇒ flat trend.
        return avg_change / avg_sales if avg_sales > 0 else 0.0

    def _review_catalog(self, step: int, active_items: set[str]) -> tuple[list[str], list[str]]:
        """Periodic activation/deactivation pass — runs every ``review_interval`` steps.

        Implements a soft hysteresis band around ``target_active_count``:
        deactivation freely drops slow movers down to ``target - margin``;
        activation only fires once the count slips below that lower band,
        then refills back up to the centre (``target_active_count``),
        capped per pass by ``max_activations_per_review`` when set.
        """
        if step % self.review_interval != 0:
            return [], []

        to_activate: list[str] = []
        to_deactivate: list[str] = []

        # Lower edge of the soft target band.
        low = self.target_active_count - self.active_margin

        # Drop every detected slow mover this review pass — but never
        # below the lower band. The old implementation dropped at most
        # one per review (``break``); on a many-SKU catalog where dozens
        # of products drift to dead simultaneously, that left a long
        # tail of inactive-but-still-stocked SKUs bleeding holding cost
        # for hundreds of ticks while the policy slowly worked through.
        remaining = len(active_items)
        for pid in list(active_items):
            if remaining <= low:
                break
            if self._is_slow_mover(pid):
                to_deactivate.append(pid)
                remaining -= 1

        # Activation hysteresis: only kick in once the count has slipped
        # below the lower band. Refill toward the centre, capped per
        # pass by ``max_activations_per_review`` (None ⇒ uncapped).
        if remaining < low:
            deficit = self.target_active_count - remaining
            cap = self.max_activations_per_review
            slots = deficit if cap is None else min(deficit, cap)
            inactive = list(set(self.sales_log.keys()) - active_items)
            # Shuffle so candidates aren't biased by iteration order.
            self.policy_rng.shuffle(inactive)
            for pid in inactive:
                if len(to_activate) >= slots:
                    break
                if self._has_growth_potential(pid):
                    to_activate.append(pid)

        return to_activate, to_deactivate

    def _is_slow_mover(self, pid: str) -> bool:
        """Has this active SKU stayed unsold long enough to drop?"""
        sales_log = self.sales_log.get(pid)
        stock_log = self.stock_log.get(pid)
        if sales_log is None or stock_log is None or len(sales_log) < self.history_window:
            return False
        recent_sales = list(sales_log)[-self.history_window:]
        recent_stock = list(stock_log)[-self.history_window:]

        if sum(recent_stock) == 0:
            # Fully stocked out for a while and still no sales => slow mover.
            return sum(list(sales_log)[-self.history_window * self.slow_mover_lookback_factor:]) == 0
        # "Supply-constrained, not slow" needs a SUSTAINED low-stock pattern,
        # not just one tick of stockout. The old per-element early-exit
        # protected products that briefly hit 0 between an order and its
        # arrival even though they were genuine slow movers overall.
        low_threshold = self.slow_sales_limit / len(recent_stock)
        low_fraction = sum(1 for s in recent_stock if s < low_threshold) / len(recent_stock)
        if low_fraction > 0.5:
            return False
        return sum(recent_sales) < self.slow_sales_limit

    def _has_growth_potential(self, pid: str) -> bool:
        """Heuristic: zero sales in the window ⇒ untested ⇒ worth activating."""
        log = self.sales_log.get(pid)
        if log is None:
            return False
        recent = list(log)[-self.history_window:]
        return sum(recent) == 0


# ---------------------------------------------------------------------------
# RLIntermediatePolicy — graph-engine RL shim (issue 12)
# ---------------------------------------------------------------------------


class RLIntermediatePolicy(IntermediatePolicy):
    """RL shim for an ``IntermediateNode`` in the graph engine.

    The RL environment works in two phases:

    1. The env encodes the observation, calls the actor network, and
       decodes the action into an ``IntermediatePolicy``-compatible dict.
    2. The decoded dict is injected via ``set_pending_action()``.
    3. The runner then calls ``decide()`` which returns the injected action.

    ``decide()`` raises ``RuntimeError`` if called before
    ``set_pending_action`` — this mirrors the old ``RLPolicy`` contract
    and makes accidental call-order bugs loud.

    The pending action format is the ``IntermediatePolicy.decide()``
    return dict::

        {
          "order":             {pid: [(supplier_id, qty), ...]},
          "list_price":        {pid: float},
          "min_order_imposed": {pid: int},
        }

    ``decide()`` also accepts the legacy store-policy dict shape (with
    ``"price"`` / ``"activate"`` / etc. keys) — this allows the existing
    ``decode_action`` helper to work without modification during the
    transition.  Unrecognised keys are passed through transparently.
    """

    def __init__(self, policy_seed: int | None = None) -> None:
        super().__init__(policy_seed=policy_seed)
        self._pending_action: dict | None = None

    def set_pending_action(self, action_dict: dict) -> None:
        """Inject the decoded action for the next ``decide()`` call."""
        self._pending_action = action_dict

    def decide(
        self, obs_intermediate: Any, central_table: Any = None
    ) -> dict[str, Any]:
        """Return the pre-injected action dict.

        Raises
        ------
        RuntimeError
            If called before ``set_pending_action``.
        """
        if self._pending_action is None:
            raise RuntimeError(
                "RLIntermediatePolicy.decide() called before set_pending_action(). "
                "The RL environment must call set_pending_action(action_dict) "
                "before the runner calls decide()."
            )
        action = self._pending_action
        self._pending_action = None  # consume the pending action
        return action


__all__ = [
    "Policy",
    "NodePolicy",
    "TextbookReorderPolicy",
    # Phase-1 graph-engine concrete policies (issue 06)
    "StaticFactoryPolicy",
    "DefaultDemandSinkPolicy",
    "IntermediatePolicy",
    # Phase-2 textbook policy family re-rooted on MultiSupplierTextbookPolicy (issue 09)
    "MultiSupplierTextbookPolicy",
    "OrderUpToPolicy",
    "ReorderPointPolicy",
    "PeriodicOrderUpToPolicy",
    "PeriodicReorderPolicy",
    # Phase-5 RL graph-engine shim (issue 12)
    "RLIntermediatePolicy",
]


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers (extracted for unit-testability)
# ─────────────────────────────────────────────────────────────────────────────


def _estimate_rate(
    sales_log: dict[str, list[int]],
    demand_window: int,
    inv_before_settle: dict[str, list[int]],
    stockout_safety_bonus_ticks: int = 0,  # retained for test compatibility; unused internally
) -> dict[str, float]:
    """Estimate the per-pid demand rate from censored sales history.

    The estimate is the plain arithmetic mean of realised sales over the
    most recent ``demand_window`` ticks. Sales are *censored* by inventory
    — on a stockout tick the realised qty is at most the on-hand stock,
    so the true demand can be strictly larger. This function intentionally
    does NOT attempt to uncensor; the response to censoring is to extend
    the effective safety horizon at the caller site (via
    ``stockout_safety_bonus_pct_of_lag``), which is the textbook practice.

    Args:
        sales_log: per-pid list of historical sales quantities (most
            recent last). May be shorter than ``demand_window``.
        demand_window: number of recent ticks to average over. If the
            log is shorter, uses the full log.
        inv_before_settle: per-pid list of inventory *before* demand was
            settled (= post-settle inventory + sales). Currently unused
            by this function but kept in the signature so the helper can
            grow an uncensoring step later without a call-site churn.
            Stockout detection lives in ``_has_stockout`` and is applied
            by the caller.
        stockout_safety_bonus_ticks: unused in this function — the
            estimator always returns the plain censored mean. Caller is
            responsible for bumping the safety horizon when
            ``_has_stockout`` reports a censored window. Retained in the
            signature so existing unit tests that pass it explicitly
            continue to work without modification.

    Returns:
        dict mapping each pid to its estimated rate (float). Pids absent
        from ``sales_log`` are not included in the output. Pids with an
        empty log get rate 0.0.
    """
    result: dict[str, float] = {}
    for pid, log in sales_log.items():
        # Empty log ⇒ no signal ⇒ rate 0.0. The cold-start pilot pass in
        # ``TextbookReorderPolicy.decide`` handles never-seen pids before
        # ``_estimate_rate`` is ever called for them, so rate 0.0 here is
        # really only reachable when a pid was passed in deliberately empty.
        if not log:
            result[pid] = 0.0
            continue
        # Tail slice: take up to the last ``demand_window`` entries.
        # When the log is shorter than the window we use everything we
        # have — better a noisy estimate than no estimate.
        window = log[-demand_window:] if demand_window < len(log) else log
        result[pid] = sum(window) / len(window)
    return result


def _has_stockout(
    pid: str,
    sales_log: dict[str, list[int]],
    inv_before_settle: dict[str, list[int]],
    demand_window: int,
) -> bool:
    """Return True if the pid had at least one stockout tick in the window.

    A stockout tick is one where ``sales[pid] == inv_before_settle[pid]``
    AND ``sales[pid] > 0`` — i.e. all available inventory was exhausted
    by demand on that tick. The ``> 0`` guard is essential: a tick with
    zero sales AND zero pre-settle inventory trivially satisfies the
    equality but is *not* censorship — there was nothing to sell.

    Args:
        pid: product ID to check.
        sales_log: per-pid history of realised sales.
        inv_before_settle: per-pid history of inventory measured *before*
            demand settlement (= post-settle inventory + sales).
        demand_window: only look at the most recent ``demand_window``
            ticks; anything older is ignored.

    Returns:
        True iff any tick in the window was a stockout.
    """
    sales = sales_log.get(pid, [])
    inv_bs = inv_before_settle.get(pid, [])
    if not sales or not inv_bs:
        return False
    # Guard against ragged logs — pid might have been added to one log
    # before the other on a partial first observation.
    n = min(len(sales), len(inv_bs), demand_window)
    for s, inv in zip(sales[-n:], inv_bs[-n:]):
        # Stockout iff demand met supply exactly AND there was demand.
        if s > 0 and s == inv:
            return True
    return False


def _allocate_two_pass_fair_share(
    desired: dict[str, int],
    unit_costs: dict[str, float],
    free_space: int,
    cash: float,
    min_qty: int,
    pilot_pids: set[str],
) -> dict[str, int]:
    """Two-pass fair-share + water-fill + mop-up capacity/cash allocator.

    The allocator turns each SKU's *desired* qty into a *feasible* qty
    subject to two shared pools — capacity headroom (``free_space``) and
    purchasing budget (``cash``). It is designed to be:

    1. **Iteration-order-independent in pass 1.** Two pids that desire
       the same qty receive the same allocation regardless of dict
       traversal order. (Pass 2 / mop-up break ties order-dependently
       only when the pool shrinks below the fair-share grain.)
    2. **Conserving.** Sum of allocations is ≤ ``free_space`` and dollar
       cost is ≤ ``cash``, by construction.
    3. **Pilot-aware.** Cold-start pilot pids bypass the ``min_qty``
       floor so a probe of 3 units doesn't get folded to zero.

    Pass 1 (fair-share):
        Each active SKU is allocated ``min(desired, space_block,
        cash_block_in_qty)`` where the pools are divided equally per SKU.
        Equal slicing across SKUs → result depends only on the *set* of
        active pids, not the order in which they were inserted.

    Pass 2 (water-fill):
        Leftover space / cash is redistributed to SKUs with remaining
        shortfall (desired > allocated). Bounded to ≤ K rounds so we
        always terminate; a round with zero net gain also breaks out.

    Mop-up:
        After integerising the float allocations we lose up to ``K - 1``
        units to floor(). A single greedy pass over shortfalls reclaims
        those leftover whole units one at a time.

    min_qty floor:
        Applied only to non-pilot allocations after both passes. Pilot
        pids bypass it so cold-start probes (deliberately small) survive.

    Args:
        desired: requested qty per pid. Zero entries are kept in the
            output as 0 (so the action dict has a stable set of keys).
        unit_costs: per-pid unit cost (for cash accounting). Missing
            entries default to 1.0; non-positive costs are treated as
            "no cash constraint" for that SKU.
        free_space: total units of capacity headroom across all SKUs.
        cash: available cash for purchasing (in currency units).
        min_qty: minimum order quantity for non-pilot SKUs. An allocation
            strictly between 0 and ``min_qty`` is set to 0
            (textbook-pure: no trivial orders).
        pilot_pids: PIDs exempt from the ``min_qty`` floor (cold-start
            probes).

    Returns:
        dict mapping every pid in ``desired`` to its allocated qty
        (``int ≥ 0``). Pids absent from ``desired`` are absent from the
        result.
    """
    # ``K`` = SKU count; used as the upper bound on water-fill rounds.
    # Each round either fills at least one shortfall or breaks early via
    # the ``progress`` flag, so K rounds is a safe ceiling.
    K = len(desired)
    if K == 0:
        return {}

    # Work in floats internally; convert to int at the end (mop-up
    # reclaims the lost fractional units).
    pids = list(desired.keys())
    # ``allocated`` is the running per-pid allocation, in float units.
    allocated: dict[str, float] = {pid: 0.0 for pid in pids}
    # Running pool counters — decremented as allocations consume them.
    remaining_space = float(free_space)
    remaining_cash = float(cash)

    def _cash_qty(pid: str, qty: float) -> float:
        """Convert desired qty to max qty affordable given remaining cash.

        Not used in the hot path (pass 1 / pass 2 inline the same logic
        with the per-SKU cash *slice* instead of the whole pool), but
        retained as a small reusable building block.
        """
        cost = unit_costs.get(pid, 1.0)
        if cost <= 0:
            return qty
        return min(qty, remaining_cash / cost)

    # ── Pass 1: fair-share ────────────────────────────────────────────────
    # ``active`` = SKUs that actually want > 0 units. SKUs with desired 0
    # carry through to the output as 0 with no further work.
    active = [pid for pid in pids if desired[pid] > 0]
    n_active = len(active)

    if n_active > 0:
        # Equal capacity slice per active SKU — the "fair-share grain".
        space_per_sku = remaining_space / n_active
        # Compute each SKU's pass-1 allocation in a staging dict first,
        # then commit. Decoupling write-from-read keeps the result
        # independent of iteration order on ``active``.
        pass1: dict[str, float] = {}
        for pid in active:
            want = float(desired[pid])
            from_space = min(want, space_per_sku)
            cost = unit_costs.get(pid, 1.0)
            # Cash slice in *qty* units. A 0-cost SKU is treated as
            # cash-unconstrained (``from_space`` becomes the binding cap).
            cash_cap = (remaining_cash / n_active) / cost if cost > 0 else from_space
            pass1[pid] = min(from_space, cash_cap)

        # Commit pass-1 allocations and debit the shared pools.
        for pid in active:
            allocated[pid] = pass1[pid]
        remaining_space -= sum(pass1.values())
        remaining_cash -= sum(
            pass1[pid] * unit_costs.get(pid, 1.0) for pid in active
        )

    # ── Pass 2: water-fill ───────────────────────────────────────────────
    # Up to K rounds — far more than ever needed in practice (most cases
    # converge in 1–2 rounds). The ``progress`` guard breaks out early
    # when a round can't deliver any gain (e.g. remaining capacity is
    # too small for any SKU to absorb at the current granularity).
    for _ in range(K):
        # Shortfall = SKUs that still want more than they got in pass 1.
        shortfall_pids = [
            pid for pid in pids if allocated[pid] < desired[pid]
        ]
        if not shortfall_pids or remaining_space <= 0 or remaining_cash <= 0:
            break
        n_sf = len(shortfall_pids)
        # Fair-share grain for this round, computed off the *remaining*
        # pool. Successive rounds use a finer grain over a smaller
        # shortfall set, so leftover capacity flows toward the SKUs
        # that need it most.
        space_per_sku = remaining_space / n_sf
        # Did any SKU absorb anything this round? If not, break out.
        progress = False
        round_alloc: dict[str, float] = {}
        for pid in shortfall_pids:
            need = float(desired[pid]) - allocated[pid]
            from_space = min(need, space_per_sku)
            cost = unit_costs.get(pid, 1.0)
            cash_cap = (remaining_cash / n_sf) / cost if cost > 0 else from_space
            gain = min(from_space, cash_cap)
            if gain > 0:
                round_alloc[pid] = gain
                progress = True
        # Commit this round's gains (again decoupled from iteration order).
        for pid, gain in round_alloc.items():
            allocated[pid] += gain
        remaining_space -= sum(round_alloc.values())
        remaining_cash -= sum(
            gain * unit_costs.get(pid, 1.0) for pid, gain in round_alloc.items()
        )
        # No SKU could absorb anything ⇒ stop (further rounds would be
        # identical no-ops).
        if not progress:
            break

    # ── Integer conversion ────────────────────────────────────────────────
    # floor() loses up to ``K - 1`` units in aggregate. Track the
    # leftover so the mop-up pass can hand it back as whole units.
    int_alloc: dict[str, int] = {pid: int(allocated[pid]) for pid in pids}
    remaining_space_int = free_space - sum(int_alloc.values())
    remaining_cash_float = cash - sum(
        int_alloc[pid] * unit_costs.get(pid, 1.0) for pid in pids
    )

    # ── Mop-up: greedy over shortfalls ───────────────────────────────────
    # Reclaim the integer-rounding tail (one unit per shortfall pid at
    # most). This pass IS iteration-order-dependent at the tie-breaking
    # level, but it only ever distributes ≤ K-1 units so the impact on
    # the final action is bounded and small.
    mop_shortfalls = [
        pid for pid in pids if int_alloc[pid] < desired[pid]
    ]
    for pid in mop_shortfalls:
        if remaining_space_int <= 0:
            break
        cost = unit_costs.get(pid, 1.0)
        # Skip SKUs we can no longer afford at unit granularity.
        if cost > 0 and remaining_cash_float < cost:
            continue
        gain = 1
        int_alloc[pid] += gain
        remaining_space_int -= gain
        remaining_cash_float -= cost

    # ── min_qty floor ─────────────────────────────────────────────────────
    # Applied LAST so pass 1 / pass 2 / mop-up don't waste capacity on
    # a SKU that will end up zeroed anyway. Pilot pids bypass this floor
    # — a 3-unit cold-start probe is the whole point of a pilot.
    for pid in pids:
        if pid not in pilot_pids and 0 < int_alloc[pid] < min_qty:
            int_alloc[pid] = 0

    return int_alloc


# ─────────────────────────────────────────────────────────────────────────────
# TextbookReorderPolicy abstract base
# ─────────────────────────────────────────────────────────────────────────────


class TextbookReorderPolicy(Policy):
    """Abstract base for the textbook reorder-policy family.

    Why these denominators
    ----------------------
    **EOQ cycle length (``cover_horizon_ticks``).** The EOQ result for the
    optimal order cycle is ``T* = sqrt(2K / (D·h))``, where ``K`` is the
    fixed ordering cost, ``D`` is demand rate, and ``h`` is holding cost
    per unit per time. Lead time does not appear in this formula: the
    optimal cycle length is independent of how long the supplier takes to
    deliver. Consequently ``cover_horizon_ticks`` is expressed as an
    absolute number of ticks and is the same for every SKU regardless of
    its individual delivery lag.

    **Safety stock (``safety_lead_pct_of_lag``).** The textbook newsvendor
    result for safety stock under normally-distributed demand is
    ``SS = z · σ · sqrt(L)``, where ``z`` is the service-level multiplier,
    ``σ`` is demand standard deviation per unit time, and ``L`` is lead time.
    Safety stock therefore scales with lead time, not independently of it.
    A SKU with ``lag=10`` needs roughly ``sqrt(10/1) ≈ 3×`` more safety stock
    than one with ``lag=1`` under the same demand volatility.

    **Linear approximation.** For the deterministic-demand ``rate × ticks``
    framing used throughout this policy family, the analogous approximation
    is ``SS ≈ k · L · rate``, where ``k`` is the safety fraction. This
    matches the rest of the policy's ``ticks × rate`` structure and preserves
    the qualitative invariant that safety scales with lead time. The safety
    fraction ``safety_lead_pct_of_lag`` is therefore multiplied by the
    per-pid ``delivery_lag`` inside ``decide()`` to compute the effective
    safety horizon in ticks:

        effective_safety_ticks = round(safety_lead_pct_of_lag × delivery_lag[pid])
        effective_bonus_ticks  = round(stockout_safety_bonus_pct_of_lag × delivery_lag[pid])
        s = (delivery_lag + effective_safety_ticks + effective_bonus_ticks) × rate
        S = s + cover_horizon_ticks × rate

    The default ``safety_lead_pct_of_lag = 1/3`` keeps a thin safety
    buffer (one-third of the per-pid lead time worth of demand) — a
    plausible textbook starting point for low-volatility demand.

    Subclasses provide two decision hooks:

    - ``_trigger(pid, step, position, s) -> bool``: should this SKU
      reorder this tick?
    - ``_quantity(pid, position, s, S) -> int``: if triggered, how many
      units to order (before allocator clamping)?

    The trigger may need information beyond ``(position, s)`` — for the
    periodic-review variants it needs the tick number and the review
    cadence. Such subclasses override ``_trigger_with_safety`` /
    ``_quantity_with_levels`` directly, bypassing the simple hooks.

    ``decide`` implements the shared pipeline:

    1. Read ``obs["active_products"]`` as the universe of SKUs.
    2. **Pilot pass.** For any active pid never observed before (no row
       in ``sales_log`` *before* this tick's log update) AND with zero
       on-hand + in-transit stock, schedule a pilot qty sized from
       ``opening_budget_pct × balance / K_active / unit_cost[pid]``,
       clamped to free space. Pilot pids bypass the ``min_qty`` floor
       at the allocator. Existing stock suppresses the pilot — demand
       surfaces through natural sales, no probe needed.
    3. For every known active pid: compute ``position`` (on-hand +
       in-transit) and ``rate`` (censored-sales mean over the last
       ``demand_window`` ticks).
    4. If ``stockout_safety_bonus_pct_of_lag > 0`` and the recent window
       contains any stockout tick, extend the effective safety horizon for
       this tick by ``round(stockout_safety_bonus_pct_of_lag × delivery_lag)``.
    5. Call ``_trigger_with_safety`` → ``_quantity_with_levels`` to get
       the per-pid desired qty.
    6. Run ``_allocate_two_pass_fair_share`` for a feasible allocation
       across the shared capacity + cash pools.
    7. Emit the action dict (flat ``base_prices``; no dynamic pricing,
       no promotions, no catalog churn).

    Both levels are recomputed every tick from the latest rate estimate, so
    they self-adjust as demand drifts and are scale-invariant in
    capacity by construction.
    """

    def __init__(
        self,
        *,
        policy_seed: int | None = None,
        cover_horizon_ticks: int = 14,
        safety_lead_pct_of_lag: float = 1 / 3,
        opening_budget_pct: float = 0.50,
        stockout_safety_bonus_pct_of_lag: float = 0.0,
        min_qty: int = 0,
    ) -> None:
        super().__init__(policy_seed=policy_seed)

        # Cycle length in ticks. Drives ``S - s`` for the order-up-to
        # quantity and ``Q`` for the rate-derived (s,Q) default. Bigger
        # horizon ⇒ fewer-larger orders (good when ordering cost is high
        # relative to holding cost). Independent of lead time (EOQ result).
        # Default 14 ticks ≈ a two-week reorder cycle, a common starting
        # point in retail-inventory textbooks.
        self.cover_horizon_ticks = cover_horizon_ticks
        # Safety fraction: multiplied by the per-pid delivery_lag to give
        # the effective extra safety ticks beyond the lead time.
        # ``effective_safety_ticks = round(safety_lead_pct_of_lag × lag)``.
        # Default 1/3 keeps a thin safety buffer (one-third of the lead
        # time worth of demand) — defensible textbook starting point for
        # low-volatility demand. See ADR 0008.
        self.safety_lead_pct_of_lag = safety_lead_pct_of_lag
        # Fraction of cash to spend on the very first order for each
        # newly-observed pid. Sized as
        # ``opening_budget_pct × balance / K_active / unit_cost[pid]``
        # so the pilot scales with the active catalog size — a 100-SKU
        # store doesn't blow the budget on the first tick.
        self.opening_budget_pct = opening_budget_pct
        # When > 0, a stockout in the recent ``demand_window`` extends
        # the effective safety horizon for this tick only by
        # ``round(stockout_safety_bonus_pct_of_lag × delivery_lag)``.
        # Stays off (0.0) by default — purely opt-in adaptiveness.
        self.stockout_safety_bonus_pct_of_lag = stockout_safety_bonus_pct_of_lag
        # Allocator floor on non-pilot orders: an allocation strictly
        # between 0 and ``min_qty`` gets folded to 0. Set to 0 (default)
        # to stay textbook-pure (no minimum order quantity).
        self.min_qty = min_qty

        # Per-pid rolling sales history (most recent last). Appended to
        # every tick by ``_update_logs``; consumed by ``_estimate_rate``.
        # Unbounded by design — textbook policies do not require a
        # window cap, and the simulator's episode lengths keep growth
        # bounded in practice.
        self.sales_log: dict[str, list[int]] = {}
        # Per-pid rolling inventory-before-settle history. ``inv_before_settle``
        # for a tick = post-settle inventory + sales that tick. Used by
        # ``_has_stockout`` to detect censored demand windows.
        self.inv_before_settle_log: dict[str, list[int]] = {}

    @abstractmethod
    def _trigger(self, pid: str, step: int, position: int, s: float) -> bool:
        """Return True if this SKU should reorder this tick.

        Args:
            pid: product ID.
            step: current simulation step.
            position: inventory position (on-hand + in-transit).
            s: the reorder point ``(delivery_lag + effective_safety) × rate``,
               already computed by the base class.
        """

    @abstractmethod
    def _quantity(self, pid: str, position: int, s: float, S: float) -> int:
        """Return how many units to order (before allocation clamping).

        Args:
            pid: product ID.
            position: inventory position (on-hand + in-transit).
            s: the reorder point.
            S: the order-up-to level ``s + cover_horizon × rate``.
        """

    def _update_logs(self, observation: Mapping[str, Any]) -> None:
        """Append current-tick sales and inv_before_settle to per-pid logs.

        Run as the first non-snapshot step in ``decide`` so the rate
        estimate and stockout detection see the freshest tick.
        """
        inventory = observation["inventory"]
        sales = observation["sales"]
        for pid in inventory:
            # inv_before_settle = post-settle inventory + sales this tick.
            # Why this formula: by the time the policy observes the world,
            # demand has already drained inventory. To recover what was on
            # the shelf *before* demand hit, we add the sales back in.
            inv_bs = inventory.get(pid, 0) + sales.get(pid, 0)
            self.inv_before_settle_log.setdefault(pid, []).append(inv_bs)
            self.sales_log.setdefault(pid, []).append(sales.get(pid, 0))

    def _get_delivery_lag(self, pid: str, observation: Mapping[str, Any]) -> float:
        """Return the delivery lag (in ticks) for ``pid`` from the observation."""
        # Per-pid delivery lags as of issue 01 — ``Store.observe()``
        # populates this map for every active pid.
        delivery_lags = observation.get("delivery_lags", {})
        if pid in delivery_lags:
            return float(delivery_lags[pid])
        # Fallback path for older observations that lack the per-pid
        # map: a single store-level scalar, or 2 if even that is missing.
        return float(observation.get("delivery_lag", 2))

    def _free_space(self, observation: Mapping[str, Any]) -> int:
        """Remaining capacity headroom = capacity - (on-hand + in-transit).

        Clamped at 0 because at the upper limit the store can technically
        hold ``capacity`` units exactly; negative headroom is meaningless.
        """
        inv = observation["inventory"]
        pending = observation.get("outstanding_orders", {})
        capacity = observation["max_capacity"]
        return max(0, int(capacity - sum(inv.values()) - sum(pending.values())))

    def decide(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Build one action dict for the given observation.

        See the class-level docstring for the pipeline. Side effects:
        ``_update_logs`` mutates ``self.sales_log`` and
        ``self.inv_before_settle_log`` in place every tick.
        """
        # Snapshot which pids are "already known" BEFORE updating logs.
        # A pid is cold-start if we have never completed a decide call
        # for it, i.e. its sales_log row didn't exist at entry to this
        # tick. Without this snapshot the pilot branch would never fire
        # — by the time we'd check, _update_logs would have already
        # written this tick's first entry for that pid.
        known_pids: set[str] = set(self.sales_log.keys())

        # Update rolling logs with current-tick data so rate estimates
        # and stockout detection use the freshest observation.
        self._update_logs(observation)

        # Active assortment for this tick. Pulled out as a list so the
        # K_active count is stable through the for-loop below.
        active_pids: list[str] = list(observation["active_products"])
        # K_active = #active SKUs, used as the divisor for the per-SKU
        # pilot budget slice. ``max(1, …)`` guards a zero-active edge
        # case (no division-by-zero on empty catalogs).
        K_active = max(1, len(active_pids))
        # Current simulation tick — needed by the periodic-review
        # subclasses' trigger.
        step: int = observation["current_sim_step"]
        # On-hand units per pid.
        inventory: Mapping[str, int] = observation["inventory"]
        # In-transit units per pid (orders placed but not yet delivered).
        pending: Mapping[str, int] = observation.get("outstanding_orders", {})
        # Cash balance — used by the pilot sizing and the allocator's
        # cash pool.
        balance: float = observation["balance"]
        # Per-pid unit cost. Pilot qty and allocator cash accounting both
        # consume this.
        unit_costs: Mapping[str, float] = observation["unit_costs"]
        # Reference MSRP per pid for the emitted flat-price decision.
        # Falls back to the realised ``product_prices`` for older
        # observations that pre-date the ``base_prices`` field.
        base_prices: Mapping[str, float] = observation.get(
            "base_prices", observation.get("product_prices", {})
        )
        # Capacity headroom — the upper bound on the sum of allocations.
        free_space = self._free_space(observation)

        # ``desired`` accumulates the per-pid pre-allocation request.
        desired: dict[str, int] = {}
        # ``pilot_pids`` tracks which pids are cold-start probes; the
        # allocator skips the ``min_qty`` floor for these.
        pilot_pids: set[str] = set()

        for pid in active_pids:
            cost = unit_costs.get(pid, 1.0)
            # Inventory position drives both the reorder trigger and the
            # order-up-to quantity. Includes in-transit so a SKU mid-
            # shipment doesn't double-order.
            position = inventory.get(pid, 0) + pending.get(pid, 0)
            delivery_lag = self._get_delivery_lag(pid, observation)

            # ── Pilot pass ────────────────────────────────────────────
            # Cold-start branch for never-previously-observed pids. We
            # check the SNAPSHOT (``known_pids``) taken before
            # _update_logs ran, so the very first decide call for a pid
            # enters this branch even though the log now has one entry
            # (from this tick's update). Unknown pids always skip the
            # steady-state path — a single-tick rate estimate is too
            # noisy to drive an (s,S) decision.
            #
            # The pilot probe itself only fires when the SKU also has no
            # stock to observe demand against (``position == 0``). With
            # existing stock, demand surfaces through natural sales and
            # no probe is needed.
            if pid not in known_pids:
                if (
                    position == 0
                    and self.opening_budget_pct > 0
                    and cost > 0
                ):
                    # Per-pid pilot budget: a fraction of balance,
                    # divided across the active catalog, converted to qty.
                    pilot_qty = int(
                        self.opening_budget_pct * balance / K_active / cost
                    )
                    # Cap at free space (shared pool — the allocator
                    # later distributes it more carefully when multiple
                    # pilots compete).
                    pilot_qty = min(pilot_qty, free_space)
                    if pilot_qty > 0:
                        desired[pid] = pilot_qty
                        pilot_pids.add(pid)
                continue

            # ── Steady-state pids ─────────────────────────────────────
            # Demand window = max(1, delivery_lag). Sized to the lead
            # time so the estimate averages over roughly one ordering
            # cycle; never 0 (would zero out the rate).
            demand_window = max(1, int(delivery_lag))
            rates = _estimate_rate(
                sales_log={pid: self.sales_log[pid]},
                demand_window=demand_window,
                inv_before_settle={pid: self.inv_before_settle_log.get(pid, [])},
            )
            # ``rate`` is the censored-sales mean. May undershoot true
            # demand on a recently stocked-out pid — handled by the
            # safety-bonus branch below.
            rate = rates.get(pid, 0.0)

            # Per-pid safety horizon in ticks: the fraction of this pid's
            # delivery lag. Computed with round() so the result is an
            # integer number of ticks, consistent with the rest of the
            # ``ticks × rate`` framing.
            effective_safety = round(self.safety_lead_pct_of_lag * delivery_lag)

            # Stockout-adaptive safety bump (opt-in). Extends the safety
            # horizon for this tick only — does not persist into future
            # ticks. The textbook response to censored demand: don't
            # uncensor the rate, widen the buffer.
            if self.stockout_safety_bonus_pct_of_lag > 0 and _has_stockout(
                pid,
                self.sales_log,
                self.inv_before_settle_log,
                demand_window,
            ):
                effective_safety += round(
                    self.stockout_safety_bonus_pct_of_lag * delivery_lag
                )

            # Trigger and quantity hooks — subclasses override either
            # ``_trigger`` / ``_quantity`` (simple) or
            # ``_trigger_with_safety`` / ``_quantity_with_levels``
            # (when the decision needs raw rate / lag).
            if self._trigger_with_safety(
                pid, step, position, rate, delivery_lag, effective_safety
            ):
                qty = self._quantity_with_levels(
                    pid, position, rate, delivery_lag, effective_safety
                )
                if qty > 0:
                    desired[pid] = qty

        # Allocate subject to the shared capacity and cash pools.
        allocation = _allocate_two_pass_fair_share(
            desired=desired,
            unit_costs=dict(unit_costs),
            free_space=free_space,
            cash=balance,
            min_qty=self.min_qty,
            pilot_pids=pilot_pids,
        )

        # Emit flat prices for all active pids (no dynamic pricing in
        # the textbook policies — pricing is a separate dimension).
        price_out: dict[str, float] = {}
        for pid in active_pids:
            price_out[pid] = float(base_prices.get(pid, 0.0))

        return {
            # Strip zero-qty entries from the action dict so the Store
            # doesn't book trivial orders.
            "order": {pid: qty for pid, qty in allocation.items() if qty > 0},
            "price": price_out,
            "activate": [],
            "deactivate": [],
            "promotions": {},
        }

    # ── Delegation helpers ──────────────────────────────────────────────
    # These wrap the (s, S) computation so simple subclasses (OrderUpToPolicy,
    # ReorderPointPolicy) can override the bare ``_trigger`` / ``_quantity``
    # hooks without ever touching ``delivery_lag`` or ``effective_safety``.
    # Periodic subclasses that need extra context (the tick number, the review
    # cadence) override ``_trigger_with_safety`` directly to bypass the wrap.

    def _trigger_with_safety(
        self,
        pid: str,
        step: int,
        position: int,
        rate: float,
        delivery_lag: float,
        effective_safety: int,
    ) -> bool:
        """Compute s and delegate to subclass ``_trigger``."""
        # Reorder point in demand units: cover the lead time plus the
        # safety horizon at the current rate estimate.
        s = (delivery_lag + effective_safety) * rate
        return self._trigger(pid, step, position, s)

    def _quantity_with_levels(
        self,
        pid: str,
        position: int,
        rate: float,
        delivery_lag: float,
        effective_safety: int,
    ) -> int:
        """Compute s, S and delegate to subclass ``_quantity``."""
        # ``s`` recomputed here matches the trigger's value exactly —
        # the duplication is intentional so neither method has to thread
        # ``s`` through the call graph.
        s = (delivery_lag + effective_safety) * rate
        # Order-up-to level: reorder point + a full cover horizon's
        # worth of demand. ``S - s = cover_horizon_ticks × rate`` by
        # construction (subclasses rely on this identity).
        S = s + self.cover_horizon_ticks * rate
        return self._quantity(pid, position, s, S)


# ─────────────────────────────────────────────────────────────────────────────
# OrderUpToPolicy — (s,S) continuous review
# ─────────────────────────────────────────────────────────────────────────────


class _OrderUpToCore(TextbookReorderPolicy):
    """(s,S) continuous-review core — inner policy for OrderUpToPolicy.

    The canonical textbook policy: review every tick (continuous review),
    reorder whenever ``position < s``, and on each reorder bring the
    position up to ``S``. The order quantity is therefore ``S - position``,
    which varies with how far position has fallen below ``s``.

    Reorder levels (demand-units framing):

        effective_safety_ticks = round(safety_lead_pct_of_lag × delivery_lag[pid])
        s = (delivery_lag + effective_safety_ticks) × rate
        S = s + cover_horizon_ticks × rate

    Both levels are computed per tick from the current rate estimate and
    the per-pid delivery lag read from the observation, so they are
    scale-invariant in capacity by construction.

    This class carries the pure (s,S) math and is used as the inner policy
    in ``OrderUpToPolicy`` (Phase-2 re-rooting onto
    ``MultiSupplierTextbookPolicy``).  Direct use via the public name
    ``OrderUpToPolicy`` is defined below.

    No additional kwargs beyond those on ``TextbookReorderPolicy``.
    """

    def _trigger(self, pid: str, step: int, position: int, s: float) -> bool:
        """Reorder when inventory position falls below ``s``."""
        return position < s

    def _quantity(self, pid: str, position: int, s: float, S: float) -> int:
        """Order ``S - position`` units, clamped at 0 and rounded to int.

        The ``max(0, …)`` clamp is defensive — the trigger already
        guarantees ``position < s < S`` so the gap is strictly positive,
        but a rate spike between trigger and quantity computation could
        in principle push ``S`` below ``position`` (rate is shared but
        re-read; no concurrency, but float jitter exists).
        """
        return max(0, int(round(S - position)))


# ─────────────────────────────────────────────────────────────────────────────
# ReorderPointPolicy — (s,Q) continuous review
# ─────────────────────────────────────────────────────────────────────────────


class _ReorderPointCore(TextbookReorderPolicy):
    """(s,Q) continuous-review core — inner policy for ReorderPointPolicy.

    Like ``_OrderUpToCore``, this reviews every tick and reorders when
    ``position < s``. The difference is what it orders: a *fixed* quantity
    ``Q`` (or a rate-derived default), independent of how deep the position
    is below ``s``.

    Reorder levels (demand-units framing):

        effective_safety_ticks = round(safety_lead_pct_of_lag × delivery_lag[pid])
        s = (delivery_lag + effective_safety_ticks) × rate
        Q = cover_horizon_ticks × rate    when Q=None (default)
        Q = self.Q                         when Q is an explicit int

    The defining (s,Q) property: the order quantity does NOT depend on how
    far below ``s`` the position fell — it is always the configured fixed
    quantity (or the rate-derived default). This makes (s,Q) attractive
    when ordering is constrained to multiples of a pack size or truck
    load; the cost is occasional double-orders right around ``s``.

    Additional kwargs beyond those on ``TextbookReorderPolicy``:

        Q: int | None = None
            Fixed reorder quantity.  When ``None`` (default), Q is computed
            per tick as ``int(round(cover_horizon_ticks × rate))``, so each
            cycle orders one cover-horizon's worth of demand.  Explicit
            integer values are honoured verbatim.
    """

    def __init__(self, *, Q: int | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        # Fixed reorder quantity (or ``None`` to use the per-tick
        # rate-derived default — see ``_quantity``).
        self.Q = Q

    def _trigger(self, pid: str, step: int, position: int, s: float) -> bool:
        """Reorder when inventory position falls below ``s``."""
        return position < s

    def _quantity(self, pid: str, position: int, s: float, S: float) -> int:
        """Order a fixed quantity ``Q`` (or the rate-derived default)."""
        if self.Q is not None:
            # Explicit configured quantity — honoured verbatim, no clamp
            # (caller is responsible for passing a sensible value).
            return self.Q
        # Rate-derived default: one cover-horizon's worth of demand.
        # ``S - s = cover_horizon_ticks × rate`` by construction in the
        # base class, so this is exactly the "one cycle" quantity
        # without needing to re-read ``cover_horizon_ticks`` here.
        return max(0, int(round(S - s)))


# ─────────────────────────────────────────────────────────────────────────────
# PeriodicOrderUpToPolicy — (R,S) periodic review
# ─────────────────────────────────────────────────────────────────────────────


class _PeriodicOrderUpToCore(TextbookReorderPolicy):
    """(R,S) periodic-review core — inner policy for PeriodicOrderUpToPolicy.

    Reviews inventory only every ``review_interval`` ticks; on review ticks
    it orders enough to bring position up to ``S``. On non-review ticks no
    order fires regardless of position — this is the defining property of
    "periodic review" and the source of (R,S)'s lower decision cost
    relative to continuous-review variants.

    Reorder level (demand-units framing):

        effective_safety_ticks = round(safety_lead_pct_of_lag × delivery_lag[pid])
        S = (delivery_lag + effective_safety_ticks + cover_horizon_ticks) × rate

    Note this is the "effective S" that ``_quantity_with_levels``
    computes as ``s + cover_horizon_ticks × rate``, which expands to the
    formula above. The reorder point ``s`` is computed for symmetry with
    the other policies but is not used by the trigger here.

    Additional kwargs beyond those on ``TextbookReorderPolicy``:

        review_interval: int | None = None
            Review cadence in ticks.  When ``None`` (default),
            ``review_interval`` equals the per-pid ``delivery_lag`` read from
            the observation (review at lead-time cadence — a common
            textbook default).  Explicit positive integers are honoured
            verbatim.

    Note: the ``max(0, …)`` clamp on the quantity is essential here because
    the trigger fires unconditionally on review ticks — position may already
    exceed ``S`` (e.g. immediately after a pilot order) and we must not emit
    a negative order.
    """

    def __init__(self, *, review_interval: int | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        # Review cadence in ticks. ``None`` ⇒ use per-pid delivery_lag
        # at trigger time (read inside ``_trigger_with_safety`` because
        # the value isn't available at construction).
        self.review_interval = review_interval

    def _trigger_with_safety(
        self,
        pid: str,
        step: int,
        position: int,
        rate: float,
        delivery_lag: float,
        effective_safety: int,
    ) -> bool:
        """Fire on the periodic schedule, ignoring inventory position.

        Overrides the base ``_trigger_with_safety`` directly so we can
        consult ``delivery_lag`` for the default review cadence without
        touching the (position, s) signature of the simple hook.
        """
        # Resolve the review interval at trigger time so the per-pid
        # delivery_lag from the observation is in scope. ``max(1, …)``
        # guards a zero-lag edge case (would cause ``step % 0``).
        ri = self.review_interval if self.review_interval is not None else max(1, int(delivery_lag))
        # Purely schedule-based: position is intentionally ignored —
        # that's what makes this (R,S) and not (R,s,S).
        return step % ri == 0

    def _trigger(self, pid: str, step: int, position: int, s: float) -> bool:
        """Not used — ``_trigger_with_safety`` is overridden directly."""
        # Required by abstract base; ``_trigger_with_safety`` overrides
        # the call path so this stub is never reached.
        return False  # pragma: no cover

    def _quantity(self, pid: str, position: int, s: float, S: float) -> int:
        """Order up to ``S`` from current position; clamp at 0.

        The clamp matters because the trigger fires unconditionally on
        review ticks. Immediately after a pilot order, ``position`` can
        exceed ``S``, and we must not emit a negative qty.
        """
        return max(0, int(round(S - position)))


# ─────────────────────────────────────────────────────────────────────────────
# PeriodicReorderPolicy — (R,s,S) periodic review with reorder-point gate
# ─────────────────────────────────────────────────────────────────────────────


class _PeriodicReorderCore(TextbookReorderPolicy):
    """(R,s,S) periodic-review core — inner policy for PeriodicReorderPolicy.

    Hybrid of (R,S) periodic review and (s,S) reorder-point gating.
    Orders fire only when BOTH conditions hold:

    1. ``step % review_interval == 0``  (periodic schedule)
    2. ``position < s``                 (reorder-point gate)

    When both are true, the order brings position up to ``S``. The
    intuition: review on a fixed cadence (cheap operationally) but
    skip the order when there's nothing to do (don't burn ordering
    cost on a partly-full shelf just because the calendar says so).

    Reorder levels (demand-units framing):

        effective_safety_ticks = round(safety_lead_pct_of_lag × delivery_lag[pid])
        s = (delivery_lag + effective_safety_ticks) × rate
        S = s + cover_horizon_ticks × rate

    Additional kwargs beyond those on ``TextbookReorderPolicy``:

        review_interval: int | None = None
            Review cadence in ticks.  When ``None`` (default),
            ``review_interval`` equals the per-pid ``delivery_lag`` read from
            the observation.  Explicit positive integers are honoured verbatim.
            Same semantics as ``PeriodicOrderUpToPolicy``.

    No ``max(0, …)`` clamp is needed on the quantity because the trigger
    requires ``position < s < S``, guaranteeing ``S - position > 0`` —
    contrast with ``PeriodicOrderUpToPolicy``, where the trigger fires
    regardless of position and the clamp is essential.
    """

    def __init__(self, *, review_interval: int | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        # Review cadence in ticks. ``None`` ⇒ resolved per-pid from
        # ``delivery_lag`` at trigger time (same semantics as
        # ``PeriodicOrderUpToPolicy``).
        self.review_interval = review_interval

    def _trigger_with_safety(
        self,
        pid: str,
        step: int,
        position: int,
        rate: float,
        delivery_lag: float,
        effective_safety: int,
    ) -> bool:
        """Fire on review ticks AND only when ``position < s``.

        Overrides ``_trigger_with_safety`` directly so we can apply both
        gates (schedule + position) in one place. We recompute ``s``
        locally — the base class' delegation helper would have done so
        too, so the duplicate compute costs nothing.
        """
        # Resolve review cadence (same fallback rule as (R,S)).
        ri = self.review_interval if self.review_interval is not None else max(1, int(delivery_lag))
        # Reorder point in demand units. Recomputed inline to keep the
        # trigger self-contained; matches ``_quantity_with_levels`` exactly.
        s = (delivery_lag + effective_safety) * rate
        return step % ri == 0 and position < s

    def _trigger(self, pid: str, step: int, position: int, s: float) -> bool:
        """Not used — ``_trigger_with_safety`` is overridden directly."""
        # Required by abstract base; ``_trigger_with_safety`` overrides
        # the call path so this stub is never reached.
        return False  # pragma: no cover

    def _quantity(self, pid: str, position: int, s: float, S: float) -> int:
        """Order up to ``S`` from current position.

        No ``max(0, …)`` clamp — the trigger guaranteed ``position < s < S``,
        so the gap is provably positive. The ``int(round(...))`` only
        applies float-to-int conversion.
        """
        return int(round(S - position))


# =============================================================================
# Phase-1 graph-engine concrete policies  (issue 06)
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# StaticFactoryPolicy — produces at fixed capacity, lists at unit_cost
# ─────────────────────────────────────────────────────────────────────────────


class StaticFactoryPolicy(FactoryPolicy):
    """Simple factory policy that lists at ``unit_cost`` (zero-margin, per
    ADR 0013) and produces by one of two rules:

    - ``target_inventory is None`` (default): produce exactly
      ``capacity_per_tick`` units every tick, unconditionally. Preserves the
      original demo/test behaviour.
    - ``target_inventory`` set: a **base-stock** rule — produce
      ``clamp(target_inventory - current_inventory, 0, capacity_per_tick)``.
      Once on-hand reaches the target, production stops; as the downstream
      shop draws inventory down, the factory refills up to capacity. This
      self-limits factory inventory to ~``target_inventory`` so long-run
      production tracks units sold instead of piling up unsold stock at
      cost (which, under ADR 0013, drives factory/system cash deeply
      negative).

    Parameters
    ----------
    capacity_per_tick:
        Maximum number of units producible in a single tick.
    unit_cost:
        Manufacturing cost per unit. ``list_price`` is set to this value.
    target_inventory:
        Base-stock target. ``None`` ⇒ unconditional produce-at-capacity.
    policy_seed:
        RNG seed (unused in this deterministic policy; included for
        interface uniformity).
    """

    def __init__(
        self,
        *,
        capacity_per_tick: int,
        unit_cost: float,
        target_inventory: int | None = None,
        policy_seed: int | None = None,
    ) -> None:
        super().__init__(policy_seed=policy_seed)
        self.capacity_per_tick = capacity_per_tick
        self.unit_cost = unit_cost
        self.target_inventory = target_inventory

    def decide(self, obs_factory: Mapping[str, Any]) -> dict[str, Any]:  # type: ignore[override]
        """Return ``{"produce_qty": int, "list_price": unit_cost}``.

        ``produce_qty`` is ``capacity_per_tick`` when no target is set, else
        the base-stock refill ``clamp(target - inventory, 0, capacity)``.
        """
        if self.target_inventory is None:
            produce_qty = self.capacity_per_tick
        else:
            current = int(obs_factory.get("inventory", 0))
            shortfall = self.target_inventory - current
            produce_qty = max(0, min(shortfall, self.capacity_per_tick))
        return {
            "produce_qty": produce_qty,
            "list_price": self.unit_cost,
        }


# ─────────────────────────────────────────────────────────────────────────────
# DefaultDemandSinkPolicy — greedy cheapest-feasible buyer
# ─────────────────────────────────────────────────────────────────────────────


class DefaultDemandSinkPolicy(DemandSinkPolicy):
    """Greedy demand-sink policy: buy from the cheapest available direct
    supplier, constrained by the sink's cash balance.

    This policy mirrors the engine's built-in default greedy action
    (``_default_sink_action``), but as an explicit ``DemandSinkPolicy``
    subclass so it can be attached to a node, swapped in tests, and
    introspected.

    The buy plan is: sort available direct-supplier offers by ``list_price``
    ascending, allocate as many units as possible from the cheapest first,
    constrained by ``demand_target`` and ``sink.cash``.

    Parameters
    ----------
    policy_seed:
        RNG seed (unused — this policy is deterministic given a stable
        sort order).
    """

    def __init__(self, *, policy_seed: int | None = None) -> None:
        super().__init__(policy_seed=policy_seed)

    def decide(
        self,
        obs_sink: Mapping[str, Any],
        central_table: Any,
    ) -> dict[str, Any]:
        """Return ``{"buy": [(supplier_id, qty), ...]}``.

        Parameters
        ----------
        obs_sink:
            Observation dict from ``build_sink_obs``.  Must contain:
            ``product_id``, ``cash``, ``demand_target``.
        central_table:
            Live ``CentralTable`` — queried via ``snapshot_for_buyer``.
        """
        pid: str = obs_sink["product_id"]
        cash: float = float(obs_sink.get("cash", 0.0))
        demand_target: float = float(obs_sink.get("demand_target", 0.0))

        offers = central_table.snapshot_for_buyer(pid)
        # Restrict to the sink's direct suppliers when the runner supplies
        # them.  The central table lists every seller of ``pid`` — including
        # indirect upstream nodes (e.g. the factory two echelons up) — so
        # without this filter the sink would buy from a cheaper indirect
        # seller and bypass its own shop, starving the shop's demand signal.
        direct_supplier_ids = obs_sink.get("direct_supplier_ids")
        if direct_supplier_ids is not None:
            offers = [(sid, o) for sid, o in offers if sid in direct_supplier_ids]
        if not offers:
            return {"buy": []}

        # Sort by price ascending — cheapest first.
        sorted_offers = sorted(offers, key=lambda t: t[1].list_price)

        remaining = int(demand_target)
        buys: list[tuple[str, int]] = []
        for supplier_id, offer in sorted_offers:
            if remaining <= 0:
                break
            if offer.available_qty <= 0:
                continue
            list_price = offer.list_price
            affordable_qty = (
                int(cash / list_price) if list_price > 0 else offer.available_qty
            )
            qty = min(remaining, offer.available_qty, affordable_qty)
            if qty > 0:
                buys.append((supplier_id, qty))
                remaining -= qty
                cash -= qty * list_price

        return {"buy": buys}


# ─────────────────────────────────────────────────────────────────────────────
# IntermediatePolicy.SingleSupplierAdapter
# ─────────────────────────────────────────────────────────────────────────────
#
# Note: ``IntermediatePolicy`` is already defined above as the ABC.  We attach
# ``SingleSupplierAdapter`` as a nested class so callers can do
# ``IntermediatePolicy.SingleSupplierAdapter(supplier_id=..., ...)`` — a
# natural namespace that matches the issue spec.
#
# Python allows adding class attributes to a class after the original
# definition, so we assign the nested class here.
# ─────────────────────────────────────────────────────────────────────────────


class _SingleSupplierAdapter(IntermediatePolicy):
    """Wraps the ``OrderUpToPolicy`` (s,S) textbook math for a single named
    upstream supplier.

    This adapter bridges the ``IntermediatePolicy`` interface (which receives
    ``obs_intermediate, central_table``) to the ``TextbookReorderPolicy``
    pipeline (which expects a Store-style observation dict).  It:

    1. Translates the intermediate-node observation into the textbook format.
    2. Calls ``OrderUpToPolicy.decide()`` to get the per-pid desired qty.
    3. Emits the new ``{"order": {pid: [(supplier_id, qty)]}}`` format,
       routing all qty to ``self.supplier_id``.

    The rate-estimate / safety-horizon math from ``TextbookReorderPolicy``
    is preserved exactly.  Only the observation-translation and output-
    reformatting layers are added here.

    Parameters
    ----------
    supplier_id:
        The single upstream supplier to route all orders to.
    cover_horizon_ticks:
        Passed through to ``OrderUpToPolicy``.  Default 14.
    safety_lead_pct_of_lag:
        Passed through to ``OrderUpToPolicy``.  Default 1/3.
    delivery_lag:
        Fixed delivery lag (ticks) from this supplier.  Used to build
        the textbook observation's ``delivery_lags`` map.  Default 2.
    unit_cost:
        Cost per unit for the supplied product.  Used by the pilot-sizing
        and allocation pass inside the textbook policy.  Default 1.0.
    list_price_out:
        Selling price to advertise downstream.  Returned in ``list_price``
        key.  Default 0.0 (no pricing change this tick).
    policy_seed:
        RNG seed for the inner textbook policy.
    """

    def __init__(
        self,
        *,
        supplier_id: str,
        cover_horizon_ticks: int = 14,
        safety_lead_pct_of_lag: float = 1 / 3,
        delivery_lag: int = 2,
        unit_cost: float = 1.0,
        list_price_out: float = 0.0,
        policy_seed: int | None = None,
    ) -> None:
        super().__init__(policy_seed=policy_seed)
        self.supplier_id = supplier_id
        self.delivery_lag = delivery_lag
        self.unit_cost = unit_cost
        self.list_price_out = list_price_out
        # Inner textbook policy — owns the reorder-level math.
        self._inner = OrderUpToPolicy(
            cover_horizon_ticks=cover_horizon_ticks,
            safety_lead_pct_of_lag=safety_lead_pct_of_lag,
            policy_seed=policy_seed,
        )

    def decide(
        self,
        obs_intermediate: Mapping[str, Any],
        central_table: Any,
    ) -> dict[str, Any]:
        """Compute reorder decisions and route all orders to ``self.supplier_id``.

        Parameters
        ----------
        obs_intermediate:
            Observation dict from ``build_intermediate_obs``.  Must contain:
            ``tick``, ``inventory``, ``pending``, ``list_prices``,
            ``min_order_imposed``.
        central_table:
            Live ``CentralTable`` — not consumed by this policy (single-
            supplier routing ignores offer comparisons); included for
            interface uniformity.

        Returns
        -------
        dict with keys:
            ``order``             — ``{pid: [(supplier_id, qty)]}``
            ``list_price``        — ``{pid: float}``
            ``min_order_imposed`` — ``{pid: int}``
        """
        tick: int = int(obs_intermediate.get("tick", 0))
        inventory: dict[str, int] = dict(obs_intermediate.get("inventory", {}))
        pending_nested: dict[str, dict[str, int]] = dict(
            obs_intermediate.get("pending", {})
        )
        list_prices: dict[str, float] = dict(obs_intermediate.get("list_prices", {}))
        min_order_imposed: dict[str, int] = dict(
            obs_intermediate.get("min_order_imposed", {})
        )

        if not inventory:
            return {
                "order": {},
                "list_price": list_prices,
                "min_order_imposed": min_order_imposed,
            }

        # Flatten pending: {supplier: {pid: qty}} → {pid: total_qty}.
        pending_flat: dict[str, int] = {}
        for sup_pending in pending_nested.values():
            for pid, qty in sup_pending.items():
                pending_flat[pid] = pending_flat.get(pid, 0) + qty

        # The textbook policy needs "sales" to update its rolling log so the
        # rate estimate (and hence the reorder point ``s``) becomes non-zero.
        # Prefer the runner-injected ``observed_sales`` (exact units sold to
        # downstream buyers this tick under demand-pull ordering); fall back
        # to 0 only when it is absent.
        # Hardcoding 0 here left ``s == 0`` forever, so ``position < s`` never
        # held and the shop never reordered.
        observed_sales: dict[str, int] = dict(
            obs_intermediate.get("observed_sales", obs_intermediate.get("prev_tick_sales", {}))
        )
        sales_approx: dict[str, int] = {
            pid: observed_sales.get(pid, 0) for pid in inventory
        }

        # Build a Store-compatible observation for the textbook policy.
        # ``max_capacity`` comes from the node's ``capacity`` field if
        # passed in obs, otherwise we use a large sentinel so the capacity
        # clamp never binds.
        node_capacity = int(obs_intermediate.get("capacity", 0)) or 10_000

        textbook_obs: dict[str, Any] = {
            "current_sim_step": tick,
            "active_products": list(inventory.keys()),
            "inventory": inventory,
            "sales": sales_approx,
            "outstanding_orders": pending_flat,
            "max_capacity": node_capacity,
            "balance": float(obs_intermediate.get("cash", 10_000.0)),
            "unit_costs": {pid: self.unit_cost for pid in inventory},
            "base_prices": dict(list_prices),
            "delivery_lags": {
                pid: float(self.delivery_lag) for pid in inventory
            },
        }

        # Run the textbook pipeline.
        textbook_result = self._inner.decide(textbook_obs)
        textbook_orders: dict[str, int] = textbook_result.get("order", {})

        # Translate {pid: qty} → {pid: [(supplier_id, qty)]} routing to
        # the single named upstream supplier.
        new_order: dict[str, list[tuple[str, int]]] = {}
        for pid, qty in textbook_orders.items():
            if qty > 0:
                new_order[pid] = [(self.supplier_id, qty)]

        # Emit selling prices: keep existing prices unless list_price_out set.
        price_out: dict[str, float] = {}
        for pid in inventory:
            price_out[pid] = (
                self.list_price_out
                if self.list_price_out > 0
                else list_prices.get(pid, 0.0)
            )

        return {
            "order": new_order,
            "list_price": price_out,
            "min_order_imposed": min_order_imposed,
        }


# Attach as a nested class on IntermediatePolicy so callers can write
# ``IntermediatePolicy.SingleSupplierAdapter(...)``.
IntermediatePolicy.SingleSupplierAdapter = _SingleSupplierAdapter  # type: ignore[attr-defined]


# =============================================================================
# Phase-2 graph-engine policies  (issue 09)
#
# MultiSupplierTextbookPolicy is the new base for the textbook reorder-policy
# family.  It implements the IntermediatePolicy contract (multi-supplier
# routing via _split_across_suppliers) while delegating per-pid trigger /
# quantity decisions to an inner TextbookReorderPolicy instance.
#
# The four concrete public classes (OrderUpToPolicy, ReorderPointPolicy,
# PeriodicOrderUpToPolicy, PeriodicReorderPolicy) are re-rooted here as
# MultiSupplierTextbookPolicy subclasses.  They also preserve the legacy
# Store-engine interface (decide(observation)) by delegating to their inner
# _Core policy.
# =============================================================================


# ---------------------------------------------------------------------------
# Default routing strategy helper
# ---------------------------------------------------------------------------

def _default_cheapest_first_strategy(
    pid: str,
    qty_total: int,
    supplier_ids: list[str],
    central_table: Any,
    *,
    buyer_min_order_floor: int = 0,
) -> list[tuple[str, int]]:
    """Cheapest-first routing strategy (default for ``_split_across_suppliers``).

    Sorts suppliers by ``list_price`` ascending (cheapest first), then
    allocates greedily, respecting:

    - ``Offer.available_qty`` — never over-allocate from a supplier.
    - ``Offer.min_order``     — supplier-imposed minimum (line rejected if
      ``allocated_from_supplier < min_order``).
    - ``buyer_min_order_floor`` — buyer-side minimum per line (line
      rejected if ``allocated_from_supplier < buyer_min_order_floor``).

    Returns
    -------
    list of ``(supplier_id, qty)`` pairs. Only non-zero lines are included.
    """
    offers_raw = central_table.snapshot_for_buyer(pid)
    # Filter to the requested supplier set.
    offers = [(sid, o) for sid, o in offers_raw if sid in supplier_ids]
    if not offers:
        return []

    # Sort by price ascending — cheapest first.
    offers.sort(key=lambda t: t[1].list_price)

    result: list[tuple[str, int]] = []
    remaining = qty_total

    for sid, offer in offers:
        if remaining <= 0:
            break
        # Effective min: the higher of both layers.
        effective_min = max(offer.min_order, buyer_min_order_floor)
        if effective_min > 0 and remaining < effective_min:
            # Remaining qty falls below the effective minimum — skip.
            continue
        if offer.available_qty <= 0:
            continue

        qty = min(remaining, offer.available_qty)
        # After clamping to available, re-check effective minimum.
        if effective_min > 0 and qty < effective_min:
            continue

        result.append((sid, qty))
        remaining -= qty

    return result


def _fill_rate_weighted_strategy(
    pid: str,
    qty_total: int,
    supplier_ids: list[str],
    central_table: Any,
    *,
    buyer_min_order_floor: int = 0,
) -> list[tuple[str, int]]:
    """Fill-rate-weighted routing strategy for ``_split_across_suppliers``.

    Allocates ``qty_total`` proportionally to each supplier's recent
    ``fill_rate_recent`` EMA, then rounds to integers (water-fill mop-up).
    Falls back to cheapest-first when all suppliers have zero fill rate or
    no offers.

    Respects ``Offer.min_order`` and ``buyer_min_order_floor`` — lines that
    fall below the effective minimum are redistributed to other suppliers.

    Returns
    -------
    list of ``(supplier_id, qty)`` pairs. Only non-zero lines are included.
    """
    offers_raw = central_table.snapshot_for_buyer(pid)
    offers = [(sid, o) for sid, o in offers_raw if sid in supplier_ids]
    if not offers:
        return []

    # Build weight = fill_rate_recent × available_qty (never allocate beyond stock).
    weights = {
        sid: max(0.0, o.fill_rate_recent) * max(0, o.available_qty)
        for sid, o in offers
    }
    total_weight = sum(weights.values())

    if total_weight <= 0.0:
        # All suppliers empty or zero fill-rate — fall back to cheapest-first.
        return _default_cheapest_first_strategy(
            pid, qty_total, supplier_ids, central_table,
            buyer_min_order_floor=buyer_min_order_floor,
        )

    result: list[tuple[str, int]] = []
    remaining = qty_total
    offer_map = {sid: o for sid, o in offers}

    # First pass: proportional allocation.
    allocations: dict[str, int] = {}
    for sid, o in offers:
        w = weights[sid] / total_weight
        raw = w * qty_total
        qty = min(int(raw), o.available_qty)
        effective_min = max(o.min_order, buyer_min_order_floor)
        if effective_min > 0 and qty < effective_min:
            qty = 0  # reject line — below minimum
        allocations[sid] = qty

    allocated = sum(allocations.values())
    leftover = qty_total - allocated

    # Second pass: distribute leftover to cheapest available supplier.
    if leftover > 0:
        sorted_by_price = sorted(offers, key=lambda t: t[1].list_price)
        for sid, o in sorted_by_price:
            if leftover <= 0:
                break
            extra = min(leftover, o.available_qty - allocations.get(sid, 0))
            if extra > 0:
                allocations[sid] = allocations.get(sid, 0) + extra
                leftover -= extra

    return [(sid, qty) for sid, qty in allocations.items() if qty > 0]


# ---------------------------------------------------------------------------
# MultiSupplierTextbookPolicy — multi-supplier base class
# ---------------------------------------------------------------------------


class MultiSupplierTextbookPolicy(IntermediatePolicy):
    """Multi-supplier base for the textbook reorder-policy family.

    Lifts the textbook (s,S) rate-estimate / safety-horizon math onto the
    full ``IntermediatePolicy`` contract.  Subclasses override
    ``_make_inner_policy`` to inject their own trigger / quantity logic.

    ``_split_across_suppliers`` is a pure static routing function:
    ``(pid, qty_total, supplier_ids, central_table) -> [(supplier_id, qty)]``.

    Parameters
    ----------
    cover_horizon_ticks:
        Passed through to the inner policy.  Default 14.
    safety_lead_pct_of_lag:
        Passed through to the inner policy.  Default 1/3.
    delivery_lag:
        Fixed delivery lag (ticks) used to build the Store-style observation
        for the inner policy.  Default 2.
    unit_cost:
        Per-unit cost for inner-policy cash accounting.  Default 1.0.
    list_price_out:
        Selling price advertised downstream.  Default 0.0 (keep existing).
    per_supplier_min_order_floor:
        Buyer-side minimum order per line (default 0 = no constraint).
    routing_strategy:
        Optional callable replacing the default cheapest-first logic.
        Signature: ``(pid, qty_total, supplier_ids, central_table, *, buyer_min_order_floor) -> [(supplier_id, qty)]``.
    policy_seed:
        RNG seed for the inner policy.
    """

    def __init__(
        self,
        *,
        cover_horizon_ticks: int = 14,
        safety_lead_pct_of_lag: float = 1 / 3,
        delivery_lag: int = 2,
        unit_cost: float = 1.0,
        list_price_out: float = 0.0,
        per_supplier_min_order_floor: int = 0,
        routing_strategy: Any = None,
        policy_seed: int | None = None,
    ) -> None:
        super().__init__(policy_seed=policy_seed)
        self.delivery_lag = delivery_lag
        self.unit_cost = unit_cost
        self.list_price_out = list_price_out
        self.per_supplier_min_order_floor = per_supplier_min_order_floor
        self.routing_strategy = routing_strategy
        # Per-pid inventory snapshot from the previous decide call.  Used by
        # ``decide`` to compute approximate sales as the inventory decrease
        # between consecutive ticks (``max(0, prev_inv - curr_inv)``).
        # Inventory increases (deliveries) give 0 sales — conservative but
        # correct: the rate estimate self-corrects once depletion cycles begin.
        self._last_inventory: dict[str, int] = {}
        self._inner = self._make_inner_policy(
            cover_horizon_ticks=cover_horizon_ticks,
            safety_lead_pct_of_lag=safety_lead_pct_of_lag,
            policy_seed=policy_seed,
        )

    def _make_inner_policy(
        self,
        *,
        cover_horizon_ticks: int,
        safety_lead_pct_of_lag: float,
        policy_seed: int | None,
    ) -> "TextbookReorderPolicy":
        """Factory for the inner textbook policy.  Default: (s,S) ``_OrderUpToCore``."""
        return _OrderUpToCore(
            cover_horizon_ticks=cover_horizon_ticks,
            safety_lead_pct_of_lag=safety_lead_pct_of_lag,
            policy_seed=policy_seed,
        )

    # ------------------------------------------------------------------
    # Core routing function (static, pure — deep-module entry point)
    # ------------------------------------------------------------------

    @staticmethod
    def _split_across_suppliers(
        pid: str,
        qty_total: int,
        supplier_ids: list[str],
        central_table: Any,
        *,
        buyer_min_order_floor: int = 0,
        routing_strategy: Any = None,
    ) -> list[tuple[str, int]]:
        """Split ``qty_total`` across ``supplier_ids``.

        Default strategy: cheapest-first, both min-order layers honoured.
        Pluggable via ``routing_strategy``.

        Parameters
        ----------
        pid:
            Product ID being ordered.
        qty_total:
            Total units to source across all suppliers.
        supplier_ids:
            IDs of eligible upstream suppliers.
        central_table:
            Live ``CentralTable`` — queried for offers.
        buyer_min_order_floor:
            Buyer-side minimum per line (default 0 = no constraint).
        routing_strategy:
            Optional callable replacing the default cheapest-first logic.
            Receives ``(pid, qty_total, supplier_ids, central_table,
            buyer_min_order_floor=...)`` and returns a list of
            ``(supplier_id, qty)`` tuples.

        Returns
        -------
        list of ``(supplier_id, qty)`` pairs, non-zero lines only.
        """
        if qty_total <= 0 or not supplier_ids:
            return []
        strategy = routing_strategy if routing_strategy is not None else _default_cheapest_first_strategy
        return strategy(
            pid,
            qty_total,
            supplier_ids,
            central_table,
            buyer_min_order_floor=buyer_min_order_floor,
        )

    # Expose fill-rate-weighted routing as a named static method so
    # search_spaces.py can reference it as a Categorical choice.
    _routing_fill_rate_weighted = staticmethod(_fill_rate_weighted_strategy)

    # ------------------------------------------------------------------
    # IntermediatePolicy.decide
    # ------------------------------------------------------------------

    def decide(
        self,
        obs_intermediate: Mapping[str, Any],
        central_table: Any,
    ) -> dict[str, Any]:
        """Route reorder decisions across available upstream suppliers.

        Translates ``obs_intermediate`` to the inner textbook policy format,
        runs the inner policy to get per-pid desired quantities, then calls
        ``_split_across_suppliers`` to distribute those quantities across
        the direct upstream suppliers in ``central_table``.

        Returns
        -------
        dict with keys:
            ``order``             — ``{pid: [(supplier_id, qty), ...]}``
            ``list_price``        — ``{pid: float}``
            ``min_order_imposed`` — ``{pid: int}``
        """
        tick: int = int(obs_intermediate.get("tick", 0))
        inventory: dict[str, int] = dict(obs_intermediate.get("inventory", {}))
        pending_nested: dict[str, dict[str, int]] = dict(
            obs_intermediate.get("pending", {})
        )
        list_prices: dict[str, float] = dict(obs_intermediate.get("list_prices", {}))
        min_order_imposed: dict[str, int] = dict(
            obs_intermediate.get("min_order_imposed", {})
        )
        direct_supplier_ids: set[str] | None = obs_intermediate.get("direct_supplier_ids")

        if not inventory:
            return {
                "order": {},
                "list_price": list_prices,
                "min_order_imposed": min_order_imposed,
            }

        # Flatten pending: {supplier: {pid: qty}} → {pid: total_qty}.
        pending_flat: dict[str, int] = {}
        for sup_pending in pending_nested.values():
            for pid, qty in sup_pending.items():
                pending_flat[pid] = pending_flat.get(pid, 0) + qty

        # Use runner-injected observed_sales when available (preferred).
        # Under demand-pull ordering (ADR 0018) the runner populates
        # ``observed_sales`` with current-tick sales that are complete by the
        # time this intermediate is processed.
        # Also accepts the old ``prev_tick_sales`` key for backward compat.
        # Fallback: estimate sales as inventory decrease from previous tick —
        # conservative (delivery ticks give 0) but self-correcting.
        _sales_signal: dict[str, int] = (
            obs_intermediate.get("observed_sales")
            or obs_intermediate.get("prev_tick_sales")
            or {}
        )
        if _sales_signal:
            sales_approx: dict[str, int] = {
                pid: _sales_signal.get(pid, 0) for pid in inventory
            }
        else:
            # Inventory-delta fallback (used when runner doesn't inject sales).
            sales_approx = {
                pid: max(0, self._last_inventory.get(pid, 0) - inventory.get(pid, 0))
                for pid in inventory
            }
        # Snapshot the current inventory for the fallback path in the next call.
        self._last_inventory = dict(inventory)

        node_capacity = int(obs_intermediate.get("capacity", 0)) or 10_000

        textbook_obs: dict[str, Any] = {
            "current_sim_step": tick,
            "active_products": list(inventory.keys()),
            "inventory": inventory,
            "sales": sales_approx,
            "outstanding_orders": pending_flat,
            "max_capacity": node_capacity,
            "balance": float(obs_intermediate.get("cash", 10_000.0)),
            "unit_costs": {pid: self.unit_cost for pid in inventory},
            "base_prices": dict(list_prices),
            "delivery_lags": {pid: float(self.delivery_lag) for pid in inventory},
        }

        textbook_result = self._inner.decide(textbook_obs)
        textbook_orders: dict[str, int] = textbook_result.get("order", {})

        new_order: dict[str, list[tuple[str, int]]] = {}
        for pid, qty in textbook_orders.items():
            if qty <= 0:
                continue
            if direct_supplier_ids is not None:
                sids = list(direct_supplier_ids)
            else:
                sids = [sid for sid, _ in central_table.snapshot_for_buyer(pid)]
            if not sids:
                continue
            splits = self._split_across_suppliers(
                pid=pid,
                qty_total=qty,
                supplier_ids=sids,
                central_table=central_table,
                buyer_min_order_floor=self.per_supplier_min_order_floor,
                routing_strategy=self.routing_strategy,
            )
            if splits:
                new_order[pid] = splits

        price_out: dict[str, float] = {}
        for pid in inventory:
            price_out[pid] = (
                self.list_price_out
                if self.list_price_out > 0
                else list_prices.get(pid, 0.0)
            )

        return {
            "order": new_order,
            "list_price": price_out,
            "min_order_imposed": min_order_imposed,
        }


# ---------------------------------------------------------------------------
# Public textbook policy family — re-rooted on MultiSupplierTextbookPolicy
# (Phase-2, issue 09).
#
# Each public class is a MultiSupplierTextbookPolicy subclass that ALSO
# exposes the legacy Store-engine interface (decide(observation)) by
# delegating to its _inner _Core policy.
# ---------------------------------------------------------------------------


def _make_textbook_decide(cls_name: str):
    """Factory for the dual-dispatch decide used by re-rooted textbook policies."""
    def decide(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        f"""Route {cls_name}.decide to Store or graph engine based on arg count."""
        if len(args) == 1 and not kwargs:
            # Single positional arg → Store-style observation dict.
            return self._inner.decide(args[0])
        if len(args) == 2:
            # Two positional args → (obs_intermediate, central_table)
            return MultiSupplierTextbookPolicy.decide(self, args[0], args[1])
        # Named args or other — forward to inner for Store compat.
        return self._inner.decide(*args, **kwargs)
    decide.__name__ = "decide"
    return decide


def _textbook_properties(inner_attrs: list[str]):
    """Return a dict of property descriptors that delegate to self._inner."""
    props: dict = {}
    for attr in inner_attrs:
        def _make_prop(a=attr):
            @property
            def _prop(self):
                return getattr(self._inner, a)
            return _prop
        props[attr] = _make_prop()
    return props


class OrderUpToPolicy(MultiSupplierTextbookPolicy):
    """(s,S) continuous-review policy, re-rooted on MultiSupplierTextbookPolicy.

    The canonical CRN comparison anchor for RL.  Review every tick,
    reorder whenever ``position < s``, bring position up to ``S``.

    Reorder levels::

        effective_safety_ticks = round(safety_lead_pct_of_lag × delivery_lag[pid])
        s = (delivery_lag + effective_safety_ticks) × rate
        S = s + cover_horizon_ticks × rate

    Supports BOTH the legacy Store-engine interface
    (``decide(observation)``) AND the graph-engine interface
    (``decide(obs_intermediate, central_table)``).
    """

    def __init__(
        self,
        *,
        policy_seed: int | None = None,
        cover_horizon_ticks: int = 14,
        safety_lead_pct_of_lag: float = 1 / 3,
        opening_budget_pct: float = 0.50,
        stockout_safety_bonus_pct_of_lag: float = 0.0,
        min_qty: int = 0,
        delivery_lag: int = 2,
        unit_cost: float = 1.0,
        list_price_out: float = 0.0,
        per_supplier_min_order_floor: int = 0,
        routing_strategy: Any = None,
    ) -> None:
        self._obu_opening_budget_pct = opening_budget_pct
        self._obu_stockout_safety_bonus = stockout_safety_bonus_pct_of_lag
        self._obu_min_qty = min_qty
        super().__init__(
            policy_seed=policy_seed,
            cover_horizon_ticks=cover_horizon_ticks,
            safety_lead_pct_of_lag=safety_lead_pct_of_lag,
            delivery_lag=delivery_lag,
            unit_cost=unit_cost,
            list_price_out=list_price_out,
            per_supplier_min_order_floor=per_supplier_min_order_floor,
            routing_strategy=routing_strategy,
        )

    def _make_inner_policy(
        self,
        *,
        cover_horizon_ticks: int,
        safety_lead_pct_of_lag: float,
        policy_seed: int | None,
    ) -> _OrderUpToCore:
        return _OrderUpToCore(
            policy_seed=policy_seed,
            cover_horizon_ticks=cover_horizon_ticks,
            safety_lead_pct_of_lag=safety_lead_pct_of_lag,
            opening_budget_pct=getattr(self, "_obu_opening_budget_pct", 0.50),
            stockout_safety_bonus_pct_of_lag=getattr(
                self, "_obu_stockout_safety_bonus", 0.0
            ),
            min_qty=getattr(self, "_obu_min_qty", 0),
        )

    decide = _make_textbook_decide("OrderUpToPolicy")

    @property
    def cover_horizon_ticks(self) -> int:
        return self._inner.cover_horizon_ticks

    @property
    def safety_lead_pct_of_lag(self) -> float:
        return self._inner.safety_lead_pct_of_lag

    @property
    def opening_budget_pct(self) -> float:
        return self._inner.opening_budget_pct

    @property
    def stockout_safety_bonus_pct_of_lag(self) -> float:
        return self._inner.stockout_safety_bonus_pct_of_lag

    @property
    def min_qty(self) -> int:
        return self._inner.min_qty

    @property
    def sales_log(self) -> dict:
        return self._inner.sales_log

    @property
    def inv_before_settle_log(self) -> dict:
        return self._inner.inv_before_settle_log


class ReorderPointPolicy(MultiSupplierTextbookPolicy):
    """(s,Q) continuous-review policy, re-rooted on MultiSupplierTextbookPolicy.

    Reviews every tick and reorders when ``position < s``.  Orders a *fixed*
    quantity ``Q`` (or rate-derived default), independent of how far below
    ``s`` position fell.

    Supports BOTH the legacy Store-engine interface and the graph-engine
    interface.
    """

    def __init__(
        self,
        *,
        Q: int | None = None,
        policy_seed: int | None = None,
        cover_horizon_ticks: int = 14,
        safety_lead_pct_of_lag: float = 1 / 3,
        opening_budget_pct: float = 0.50,
        stockout_safety_bonus_pct_of_lag: float = 0.0,
        min_qty: int = 0,
        delivery_lag: int = 2,
        unit_cost: float = 1.0,
        list_price_out: float = 0.0,
        per_supplier_min_order_floor: int = 0,
        routing_strategy: Any = None,
    ) -> None:
        self._rpp_Q = Q
        self._rpp_opening_budget_pct = opening_budget_pct
        self._rpp_stockout_safety_bonus = stockout_safety_bonus_pct_of_lag
        self._rpp_min_qty = min_qty
        super().__init__(
            policy_seed=policy_seed,
            cover_horizon_ticks=cover_horizon_ticks,
            safety_lead_pct_of_lag=safety_lead_pct_of_lag,
            delivery_lag=delivery_lag,
            unit_cost=unit_cost,
            list_price_out=list_price_out,
            per_supplier_min_order_floor=per_supplier_min_order_floor,
            routing_strategy=routing_strategy,
        )

    def _make_inner_policy(
        self,
        *,
        cover_horizon_ticks: int,
        safety_lead_pct_of_lag: float,
        policy_seed: int | None,
    ) -> _ReorderPointCore:
        return _ReorderPointCore(
            Q=getattr(self, "_rpp_Q", None),
            policy_seed=policy_seed,
            cover_horizon_ticks=cover_horizon_ticks,
            safety_lead_pct_of_lag=safety_lead_pct_of_lag,
            opening_budget_pct=getattr(self, "_rpp_opening_budget_pct", 0.50),
            stockout_safety_bonus_pct_of_lag=getattr(
                self, "_rpp_stockout_safety_bonus", 0.0
            ),
            min_qty=getattr(self, "_rpp_min_qty", 0),
        )

    decide = _make_textbook_decide("ReorderPointPolicy")

    @property
    def Q(self) -> int | None:
        return self._inner.Q  # type: ignore[attr-defined]

    @property
    def cover_horizon_ticks(self) -> int:
        return self._inner.cover_horizon_ticks

    @property
    def safety_lead_pct_of_lag(self) -> float:
        return self._inner.safety_lead_pct_of_lag

    @property
    def opening_budget_pct(self) -> float:
        return self._inner.opening_budget_pct

    @property
    def sales_log(self) -> dict:
        return self._inner.sales_log

    @property
    def inv_before_settle_log(self) -> dict:
        return self._inner.inv_before_settle_log


class PeriodicOrderUpToPolicy(MultiSupplierTextbookPolicy):
    """(R,S) periodic-review policy, re-rooted on MultiSupplierTextbookPolicy.

    Reviews only every ``review_interval`` ticks; on review ticks orders
    up to ``S``.  Supports BOTH the legacy Store-engine and graph-engine
    interfaces.
    """

    def __init__(
        self,
        *,
        review_interval: int | None = None,
        policy_seed: int | None = None,
        cover_horizon_ticks: int = 14,
        safety_lead_pct_of_lag: float = 1 / 3,
        opening_budget_pct: float = 0.50,
        stockout_safety_bonus_pct_of_lag: float = 0.0,
        min_qty: int = 0,
        delivery_lag: int = 2,
        unit_cost: float = 1.0,
        list_price_out: float = 0.0,
        per_supplier_min_order_floor: int = 0,
        routing_strategy: Any = None,
    ) -> None:
        self._poutp_review_interval = review_interval
        self._poutp_opening_budget_pct = opening_budget_pct
        self._poutp_stockout_safety_bonus = stockout_safety_bonus_pct_of_lag
        self._poutp_min_qty = min_qty
        super().__init__(
            policy_seed=policy_seed,
            cover_horizon_ticks=cover_horizon_ticks,
            safety_lead_pct_of_lag=safety_lead_pct_of_lag,
            delivery_lag=delivery_lag,
            unit_cost=unit_cost,
            list_price_out=list_price_out,
            per_supplier_min_order_floor=per_supplier_min_order_floor,
            routing_strategy=routing_strategy,
        )

    def _make_inner_policy(
        self,
        *,
        cover_horizon_ticks: int,
        safety_lead_pct_of_lag: float,
        policy_seed: int | None,
    ) -> _PeriodicOrderUpToCore:
        return _PeriodicOrderUpToCore(
            review_interval=getattr(self, "_poutp_review_interval", None),
            policy_seed=policy_seed,
            cover_horizon_ticks=cover_horizon_ticks,
            safety_lead_pct_of_lag=safety_lead_pct_of_lag,
            opening_budget_pct=getattr(self, "_poutp_opening_budget_pct", 0.50),
            stockout_safety_bonus_pct_of_lag=getattr(
                self, "_poutp_stockout_safety_bonus", 0.0
            ),
            min_qty=getattr(self, "_poutp_min_qty", 0),
        )

    decide = _make_textbook_decide("PeriodicOrderUpToPolicy")

    @property
    def review_interval(self) -> int | None:
        return self._inner.review_interval  # type: ignore[attr-defined]

    @property
    def cover_horizon_ticks(self) -> int:
        return self._inner.cover_horizon_ticks

    @property
    def safety_lead_pct_of_lag(self) -> float:
        return self._inner.safety_lead_pct_of_lag

    @property
    def opening_budget_pct(self) -> float:
        return self._inner.opening_budget_pct

    @property
    def sales_log(self) -> dict:
        return self._inner.sales_log

    @property
    def inv_before_settle_log(self) -> dict:
        return self._inner.inv_before_settle_log


class PeriodicReorderPolicy(MultiSupplierTextbookPolicy):
    """(R,s,S) periodic-review policy, re-rooted on MultiSupplierTextbookPolicy.

    Hybrid of (R,S) and (s,S): fires only when both the review schedule AND
    the reorder point gate are satisfied.  Supports BOTH the legacy
    Store-engine and graph-engine interfaces.
    """

    def __init__(
        self,
        *,
        review_interval: int | None = None,
        policy_seed: int | None = None,
        cover_horizon_ticks: int = 14,
        safety_lead_pct_of_lag: float = 1 / 3,
        opening_budget_pct: float = 0.50,
        stockout_safety_bonus_pct_of_lag: float = 0.0,
        min_qty: int = 0,
        delivery_lag: int = 2,
        unit_cost: float = 1.0,
        list_price_out: float = 0.0,
        per_supplier_min_order_floor: int = 0,
        routing_strategy: Any = None,
    ) -> None:
        self._prp_review_interval = review_interval
        self._prp_opening_budget_pct = opening_budget_pct
        self._prp_stockout_safety_bonus = stockout_safety_bonus_pct_of_lag
        self._prp_min_qty = min_qty
        super().__init__(
            policy_seed=policy_seed,
            cover_horizon_ticks=cover_horizon_ticks,
            safety_lead_pct_of_lag=safety_lead_pct_of_lag,
            delivery_lag=delivery_lag,
            unit_cost=unit_cost,
            list_price_out=list_price_out,
            per_supplier_min_order_floor=per_supplier_min_order_floor,
            routing_strategy=routing_strategy,
        )

    def _make_inner_policy(
        self,
        *,
        cover_horizon_ticks: int,
        safety_lead_pct_of_lag: float,
        policy_seed: int | None,
    ) -> _PeriodicReorderCore:
        return _PeriodicReorderCore(
            review_interval=getattr(self, "_prp_review_interval", None),
            policy_seed=policy_seed,
            cover_horizon_ticks=cover_horizon_ticks,
            safety_lead_pct_of_lag=safety_lead_pct_of_lag,
            opening_budget_pct=getattr(self, "_prp_opening_budget_pct", 0.50),
            stockout_safety_bonus_pct_of_lag=getattr(
                self, "_prp_stockout_safety_bonus", 0.0
            ),
            min_qty=getattr(self, "_prp_min_qty", 0),
        )

    decide = _make_textbook_decide("PeriodicReorderPolicy")

    @property
    def review_interval(self) -> int | None:
        return self._inner.review_interval  # type: ignore[attr-defined]

    @property
    def cover_horizon_ticks(self) -> int:
        return self._inner.cover_horizon_ticks

    @property
    def safety_lead_pct_of_lag(self) -> float:
        return self._inner.safety_lead_pct_of_lag

    @property
    def opening_budget_pct(self) -> float:
        return self._inner.opening_budget_pct

    @property
    def sales_log(self) -> dict:
        return self._inner.sales_log

    @property
    def inv_before_settle_log(self) -> dict:
        return self._inner.inv_before_settle_log
