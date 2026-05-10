"""Policy ABC + ``BaselinePolicy`` (issues 03 + 06).

A ``Policy`` is the decision-making brain attached to a ``Store``. It
receives an ``Observation`` dict each tick and returns an ``Action``
dict with four required keys (``order``, ``price``, ``activate``,
``deactivate``) plus optional ``promotions`` / ``promotion_cooldown``.

Each ``Policy`` instance owns its own ``policy_rng`` seeded from
``policy_seed``. World streams (``world_rng``) and policy streams never
share an RNG, so swapping a policy on a Scenario does not perturb the
world.

``BaselinePolicy`` is the heuristic baseline ported verbatim from the
deleted ``src/agents/policy.py``. The behaviour (cooldown gating,
capacity respect, price floors at unit cost, periodic catalog review)
is preserved; what changed is the hyperparameter pathway. The old
``init_params`` / ``live_params`` broadcasting dicts (encoded across
two methods) are gone in favour of plain keyword arguments. Stochastic
choices (``randint`` for promo duration / order-cooldown jitter,
``shuffle`` of the inactive product list during catalog review,
``promo_discount`` sampling) all flow through ``self.policy_rng`` —
never the global ``random`` module and never ``world_rng``.
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


class NoopPolicy(Policy):
    """Returns an empty action dict; consumes no policy_rng draws.

    Useful for determinism tests where we want to drive the runner
    loop without making any decisions.
    """

    def decide(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Return ``{}`` — no orders, no prices, no catalog changes."""
        return {}


def _maybe_sample(value: Any, rng: Random) -> Any:
    """Sample a ``Distribution`` value once; pass through scalars."""
    if isinstance(value, Distribution):
        return value.sample(rng)
    return value


class BaselinePolicy(Policy):
    """Heuristic baseline policy with kwargs hyperparameters.

    Observation contract (every step, supplied by ``Store.observe``):

    - ``current_sim_step`` (int): current simulation tick.
    - ``inventory`` (dict[pid, int]): on-hand units per product.
    - ``max_capacity`` (int | float): total store capacity.
    - ``outstanding_orders`` (dict[pid, int]): in-transit pending qty.
    - ``active_products`` (Iterable[pid]): products in the active assortment.
    - ``initial_order_needed`` (Iterable[pid]): newly activated products
      that have not yet received their first replenishment.
    - ``product_prices`` (dict[pid, float]): current base prices per pid.
    - ``unit_costs`` (dict[pid, float]): unit costs per pid (price floor).
    - ``related_products`` (dict[pid, list[(pid, float)]]): cross-product
      correlation graph used by the cross-price adjustment.
    - ``balance`` (float): store balance, divides into per-product budget.
    - ``sales`` (dict[pid, int]): units sold this step (for trend log).
    - ``promotions`` (dict[pid, dict]): currently active promotions.
    - ``promotion_cooldown`` (dict[pid, int]): per-product cooldown end-step.

    All stochastic choices (``randint``, ``shuffle``, ``promo_discount``
    sampling) consume ``self.policy_rng`` only.
    """

    def __init__(
        self,
        *,
        policy_seed: int | None = None,
        min_qty: int = 10,
        init_qty_factor: float = 0.3,
        min_promo_len: int = 3,
        max_promo_len: int = 10,
        promo_cd_len: int = 10,
        review_interval: int = 5,
        promo_threshold: float = 0.7,
        target_active_count: int = 10,
        slow_sales_limit: int = 5,
        stock_lo_ratio: float = 0.2,
        stock_hi_ratio: float = 0.6,
        price_up_factor: float = 1.1,
        price_down_factor: float = 0.9,
        history_window: int = 10,
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

        # min_promo_len <= max_promo_len; preserve the old guard that
        # silently swapped them if mis-ordered.
        self.min_promo_len = min(min_promo_len, max_promo_len)
        self.max_promo_len = max(min_promo_len, max_promo_len)

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

    def decide(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Build and return one action dict for the given observation."""
        # Update rolling history before consulting it.
        self._record_sales(observation)
        self._record_stock(observation)

        # Pull observation fields into locals — keeps the four planner
        # calls readable and avoids dict-lookup repetition.
        promos = dict(observation["promotions"])
        promo_cooldown = dict(observation["promotion_cooldown"])
        step = observation["current_sim_step"]
        inventory = observation["inventory"]
        capacity = observation["max_capacity"]
        pending = observation["outstanding_orders"]
        active_items = set(observation["active_products"])
        needs_init = set(observation["initial_order_needed"])
        prices = observation["product_prices"]
        related = observation["related_products"]
        balance = observation["balance"]
        costs = observation["unit_costs"]

        # Four decision sub-passes, in fixed order:
        promos, promo_cooldown = self._plan_promos(
            promos, promo_cooldown, step, inventory, capacity
        )
        orders = self._plan_orders(
            inventory, pending, capacity, active_items, needs_init, balance, costs, step
        )
        pricing = self._plan_prices(
            inventory, prices, capacity, active_items, promos, related, costs
        )
        activate, deactivate = self._review_catalog(step, active_items)

        return {
            "promotions": promos,
            "promotion_cooldown": promo_cooldown,
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
        promo_cooldown: dict[str, int],
        step: int,
        inventory: Mapping[str, int],
        capacity: float,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
        """Expire reached-duration promos, prune cooldowns, start new ones on heavy stock."""
        # Promotion windows are step-based; once duration is reached, expire.
        expired = [pid for pid, p in promos.items() if step - p["start_step"] >= p["duration"]]
        for pid in expired:
            del promos[pid]
            # Set the cooldown end so the same product can't go on
            # promo again until ``promo_cd_len`` steps have passed.
            promo_cooldown[pid] = step + self.promo_cd_len

        # Drop already-elapsed cooldowns.
        promo_cooldown = {pid: s for pid, s in promo_cooldown.items() if s > step}

        for pid, stock in inventory.items():
            # Only consider non-promoted, non-cooldown products.
            if pid not in promos and pid not in promo_cooldown:
                # "Heavy stock" gate: stock above promo_threshold × capacity.
                if stock > capacity * self.promo_threshold:
                    # Sample (or pass through) the promo discount.
                    discount = float(_maybe_sample(self.promo_discount, self.policy_rng))
                    # Promo duration sampled from [min, max] inclusive.
                    duration = self.policy_rng.randint(self.min_promo_len, self.max_promo_len)
                    promos[pid] = {
                        "discount": discount,
                        "duration": duration,
                        "start_step": step,
                    }
        return promos, promo_cooldown

    def _plan_orders(
        self,
        inventory: Mapping[str, int],
        pending: Mapping[str, int],
        capacity: float,
        active_items: set[str],
        needs_init: set[str],
        balance: float,
        costs: Mapping[str, float],
        step: int,
    ) -> dict[str, int]:
        """Decide replenishment qty per product, respecting capacity + cooldown + budget."""
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

                if pid in needs_init:
                    # First-time order after activation: opportunistic fill.
                    qty = self._initial_order(space, max_qty)
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

                qty = int(qty)
                orders[pid] = qty
                space = max(0, space - qty)
            else:
                orders[pid] = 0

        return orders

    def _initial_order(self, space: int, max_qty: int) -> int:
        """Opportunistic first-order qty: fill init_qty_factor × space, capped by budget."""
        qty = min(max_qty, int(space * self.init_qty_factor))
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
        prices: Mapping[str, float],
        capacity: float,
        active_items: set[str],
        promos: Mapping[str, Mapping[str, Any]],
        related: Mapping[str, list[tuple[str, float]]],
        costs: Mapping[str, float],
    ) -> dict[str, float]:
        """Per-product price decision: dynamic factor for actives, clearance for inactives."""
        pricing: dict[str, float] = {}
        for pid in inventory.keys():
            # Reference price = current base for this product.
            base = prices[pid]
            cost = costs[pid]
            if pid in active_items:
                # Combine stock/trend/cross/promo factors into one multiplier.
                factor = self._compute_price_factor(
                    pid, inventory, capacity, active_items, promos, related
                )
                decision = base * factor
            else:
                # Inactive ⇒ clearance pricing.
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
        # Stock ratio relative to total store capacity.
        ratio = inventory[pid] / capacity if capacity else 0.0

        if trend > self.trend_threshold and ratio < self.stock_lo_ratio:
            # Strong demand + low stock ⇒ raise price.
            factor = self.price_up_factor
        elif trend < -self.trend_threshold or ratio > self.stock_hi_ratio:
            # Cooling demand or heavy stock ⇒ markdown.
            factor = self.price_down_factor
        else:
            factor = 1.0

        # Per-related-product cross-adjustment. Active + in-inventory only.
        for rel_id, corr in related.get(pid, []):
            if rel_id in active_items and rel_id in inventory:
                rel_ratio = inventory[rel_id] / capacity if capacity else 0.0
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
        """Periodic activation/deactivation pass — runs every ``review_interval`` steps."""
        if step % self.review_interval != 0:
            return [], []

        to_activate: list[str] = []
        to_deactivate: list[str] = []

        # Drop at most one slow-mover this review pass (the ``break``).
        for pid in list(active_items):
            if self._is_slow_mover(pid):
                to_deactivate.append(pid)
                break

        # If the active set has slack, promote at most one inactive
        # candidate with growth potential.
        if len(active_items) < self.target_active_count:
            inactive = list(set(self.sales_log.keys()) - active_items)
            # Shuffle so the candidate isn't biased by iteration order.
            self.policy_rng.shuffle(inactive)
            for pid in inactive:
                if self._has_growth_potential(pid):
                    to_activate.append(pid)
                    break

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
            if sum(list(sales_log)[-self.history_window * 6:]) == 0:
                return True
        for s in recent_stock:
            # Consistently low stock implies supply-constrained, not slow.
            if s < (self.slow_sales_limit / len(recent_stock)):
                return False
        return sum(recent_sales) < self.slow_sales_limit

    def _has_growth_potential(self, pid: str) -> bool:
        """Heuristic: zero sales in the window ⇒ untested ⇒ worth activating."""
        log = self.sales_log.get(pid)
        if log is None:
            return False
        recent = list(log)[-self.history_window:]
        return sum(recent) == 0


__all__ = ["Policy", "NoopPolicy", "BaselinePolicy"]
