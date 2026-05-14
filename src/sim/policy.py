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

Textbook reorder-policy family (issue 01):

- ``TextbookReorderPolicy`` — abstract base. Implements the shared
  observe→pilot→trigger→allocate pipeline via a ``decide`` method.
  Subclasses implement ``_trigger`` and ``_quantity`` hooks.
- ``OrderUpToPolicy`` — (s,S) continuous review. Becomes the canonical
  CRN comparison anchor for RL.

Two private module-level helpers extracted for unit-testability:

- ``_estimate_rate`` — censored-sales rate estimator.
- ``_allocate_two_pass_fair_share`` — two-pass fair-share + water-fill
  + integer mop-up allocator.
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


class HeuristicPolicy(Policy):
    """Heuristic kitchen-sink demonstrator policy with kwargs hyperparameters.

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


class RLPolicy(Policy):
    """Minimal ``Policy`` shim for the RL environment.

    The env calls ``set_pending_action(action_dict)`` immediately before
    invoking ``store.decide(observation)``.  ``decide`` returns that dict
    and clears the pending slot so a second call without a fresh
    ``set_pending_action`` raises immediately (catching integration bugs).

    ``policy_rng`` is never consumed — all stochasticity in the RL env
    flows through ``world_rng`` exactly as in ``Runner``.

    Action dict format (same as ``HeuristicPolicy``)::

        {
            "order":      {pid: int},
            "price":      {pid: float},
            "activate":   [],
            "deactivate": [],
            "promotions": {},
        }
    """

    _SENTINEL = object()

    def __init__(self) -> None:
        # policy_seed=None so policy_rng is unseeded (never drawn from).
        super().__init__(policy_seed=None)
        self._pending: dict | object = self._SENTINEL

    def set_pending_action(self, action_dict: dict) -> None:
        """Store ``action_dict`` so the next ``decide`` call can return it."""
        self._pending = action_dict

    def decide(self, observation) -> dict:
        """Return the pending action dict and clear it.

        Raises ``RuntimeError`` if called without a preceding
        ``set_pending_action`` — this surfaces env wiring bugs immediately
        rather than returning a silent empty action.
        """
        if self._pending is self._SENTINEL:
            raise RuntimeError(
                "RLPolicy.decide() called without a preceding set_pending_action(). "
                "The RL env must call set_pending_action(action_dict) before "
                "store.decide(obs) on every tick."
            )
        action = self._pending
        self._pending = self._SENTINEL
        return action


__all__ = [
    "Policy",
    "NoopPolicy",
    "HeuristicPolicy",
    "RLPolicy",
    "TextbookReorderPolicy",
    "OrderUpToPolicy",
    "ReorderPointPolicy",
    "PeriodicOrderUpToPolicy",
]


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers (extracted for unit-testability)
# ─────────────────────────────────────────────────────────────────────────────


def _estimate_rate(
    sales_log: dict[str, list[int]],
    demand_window: int,
    inv_before_settle: dict[str, list[int]],
    stockout_safety_bonus_ticks: int = 0,
) -> dict[str, float]:
    """Estimate the per-pid demand rate from censored sales history.

    Args:
        sales_log: per-pid list of historical sales quantities (most
            recent last). May be shorter than ``demand_window``.
        demand_window: number of recent ticks to average over. If the
            log is shorter, uses the full log.
        inv_before_settle: per-pid list of inventory *before* demand was
            settled (= post-settle inventory + sales). Used only when
            ``stockout_safety_bonus_ticks > 0`` to detect censored ticks.
            Must have the same length as the corresponding ``sales_log``
            entry when supplied.
        stockout_safety_bonus_ticks: when > 0, the returned rate for a
            pid that had at least one stockout tick in the window is
            stored unchanged but ``_has_stockout`` context is tracked by
            the caller. The estimator itself returns the plain censored
            mean; bonus application is the caller's responsibility.

    Returns:
        dict mapping each pid to its estimated rate (float). Pids absent
        from ``sales_log`` are not included in the output. Pids with an
        empty log get rate 0.0.
    """
    result: dict[str, float] = {}
    for pid, log in sales_log.items():
        if not log:
            result[pid] = 0.0
            continue
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

    A stockout tick is one where sales[pid] == inv_before_settle[pid]
    AND sales[pid] > 0 (meaning all inventory was exhausted by demand).
    """
    sales = sales_log.get(pid, [])
    inv_bs = inv_before_settle.get(pid, [])
    if not sales or not inv_bs:
        return False
    n = min(len(sales), len(inv_bs), demand_window)
    for s, inv in zip(sales[-n:], inv_bs[-n:]):
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

    Pass 1 (fair-share):
        Each active SKU is allocated ``min(desired, space_block,
        cash_block_in_qty)`` where the pools are divided equally per SKU.
        Iteration-order-independent within pass 1.

    Pass 2 (water-fill):
        Leftover space/cash is redistributed to SKUs with remaining
        shortfall (desired > allocated). Bounded to ≤ K rounds.

    Mop-up:
        A greedy pass over remaining shortfalls collects the integer-
        rounding tail (≤ K-1 units).

    min_qty floor:
        Applied only to non-pilot allocations after both passes. Pilot
        pids bypass it.

    Args:
        desired: requested qty per pid. Zero entries are kept in the
            output as 0.
        unit_costs: per-pid unit cost (for cash accounting).
        free_space: total units of capacity headroom across all SKUs.
        cash: available cash for purchasing.
        min_qty: minimum order quantity for non-pilot SKUs. An allocation
            below this floor is set to 0 (textbook-pure: no trivial orders).
        pilot_pids: PIDs exempt from the min_qty floor (cold-start probes).

    Returns:
        dict mapping every pid in ``desired`` to its allocated qty (int ≥ 0).
    """
    K = len(desired)
    if K == 0:
        return {}

    # Work in floats internally; convert to int at the end.
    pids = list(desired.keys())
    allocated: dict[str, float] = {pid: 0.0 for pid in pids}
    remaining_space = float(free_space)
    remaining_cash = float(cash)

    def _cash_qty(pid: str, qty: float) -> float:
        """Convert desired qty to max qty affordable given remaining cash."""
        cost = unit_costs.get(pid, 1.0)
        if cost <= 0:
            return qty
        return min(qty, remaining_cash / cost)

    # ── Pass 1: fair-share ────────────────────────────────────────────────
    active = [pid for pid in pids if desired[pid] > 0]
    n_active = len(active)

    if n_active > 0:
        space_per_sku = remaining_space / n_active
        # Compute each SKU's pass-1 allocation independently of order.
        pass1: dict[str, float] = {}
        for pid in active:
            want = float(desired[pid])
            from_space = min(want, space_per_sku)
            cost = unit_costs.get(pid, 1.0)
            cash_cap = (remaining_cash / n_active) / cost if cost > 0 else from_space
            pass1[pid] = min(from_space, cash_cap)

        for pid in active:
            allocated[pid] = pass1[pid]
        remaining_space -= sum(pass1.values())
        remaining_cash -= sum(
            pass1[pid] * unit_costs.get(pid, 1.0) for pid in active
        )

    # ── Pass 2: water-fill ───────────────────────────────────────────────
    for _ in range(K):
        shortfall_pids = [
            pid for pid in pids if allocated[pid] < desired[pid]
        ]
        if not shortfall_pids or remaining_space <= 0 or remaining_cash <= 0:
            break
        n_sf = len(shortfall_pids)
        space_per_sku = remaining_space / n_sf
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
        for pid, gain in round_alloc.items():
            allocated[pid] += gain
        remaining_space -= sum(round_alloc.values())
        remaining_cash -= sum(
            gain * unit_costs.get(pid, 1.0) for pid, gain in round_alloc.items()
        )
        if not progress:
            break

    # ── Integer conversion ────────────────────────────────────────────────
    int_alloc: dict[str, int] = {pid: int(allocated[pid]) for pid in pids}
    remaining_space_int = free_space - sum(int_alloc.values())
    remaining_cash_float = cash - sum(
        int_alloc[pid] * unit_costs.get(pid, 1.0) for pid in pids
    )

    # ── Mop-up: greedy over shortfalls ───────────────────────────────────
    mop_shortfalls = [
        pid for pid in pids if int_alloc[pid] < desired[pid]
    ]
    for pid in mop_shortfalls:
        if remaining_space_int <= 0:
            break
        cost = unit_costs.get(pid, 1.0)
        if cost > 0 and remaining_cash_float < cost:
            continue
        gain = 1
        int_alloc[pid] += gain
        remaining_space_int -= gain
        remaining_cash_float -= cost

    # ── min_qty floor ─────────────────────────────────────────────────────
    for pid in pids:
        if pid not in pilot_pids and 0 < int_alloc[pid] < min_qty:
            int_alloc[pid] = 0

    return int_alloc


# ─────────────────────────────────────────────────────────────────────────────
# TextbookReorderPolicy abstract base
# ─────────────────────────────────────────────────────────────────────────────


class TextbookReorderPolicy(Policy):
    """Abstract base for the textbook reorder-policy family.

    Subclasses implement two hooks:

    - ``_trigger(pid, step, position, rate) -> bool``: should this SKU
      reorder this tick?
    - ``_quantity(pid, position, rate) -> int``: if triggered, how many
      units to order?

    ``decide`` implements the shared pipeline:

    1. Read ``obs["active_products"]`` as the universe.
    2. Pilot pass: for any active pid never observed (no row in
       ``sales_log``), schedule a pilot qty sized from
       ``opening_budget_pct × balance / K_active / unit_cost[pid]``,
       clamped to free space. Pilot pids bypass ``min_qty``.
    3. For every active pid: compute ``position`` and ``rate``.
    4. For non-pilot active pids: call ``_trigger``; if True, set
       ``desired[pid] = max(0, _quantity(pid, position, rate))``.
    5. Run ``_allocate_two_pass_fair_share`` for a feasible allocation.
    6. Emit the action dict (flat prices, no dynamic pricing/promotions).
    """

    def __init__(
        self,
        *,
        policy_seed: int | None = None,
        cover_horizon_ticks: int = 10,
        safety_lead_ticks: int = 2,
        opening_budget_pct: float = 0.50,
        stockout_safety_bonus_ticks: int = 0,
        min_qty: int = 0,
    ) -> None:
        super().__init__(policy_seed=policy_seed)
        self.cover_horizon_ticks = cover_horizon_ticks
        self.safety_lead_ticks = safety_lead_ticks
        self.opening_budget_pct = opening_budget_pct
        self.stockout_safety_bonus_ticks = stockout_safety_bonus_ticks
        self.min_qty = min_qty
        # Per-pid rolling sales history (most recent last).
        self.sales_log: dict[str, list[int]] = {}
        # Per-pid rolling inventory-before-settle history for stockout detection.
        self.inv_before_settle_log: dict[str, list[int]] = {}

    @abstractmethod
    def _trigger(self, pid: str, step: int, position: int, s: float) -> bool:
        """Return True if this SKU should reorder this tick.

        Args:
            pid: product ID.
            step: current simulation step.
            position: inventory position (on-hand + in-transit).
            s: the reorder point (delivery_lag + effective_safety) × rate,
               already computed by the base class.
        """

    @abstractmethod
    def _quantity(self, pid: str, position: int, s: float, S: float) -> int:
        """Return how many units to order (before allocation clamping).

        Args:
            pid: product ID.
            position: inventory position (on-hand + in-transit).
            s: the reorder point.
            S: the order-up-to level (s + cover_horizon × rate).
        """

    def _update_logs(self, observation: Mapping[str, Any]) -> None:
        """Append current-tick sales and inv_before_settle to per-pid logs."""
        inventory = observation["inventory"]
        sales = observation["sales"]
        for pid in inventory:
            # inv_before_settle = post-settle inventory + sales this tick
            inv_bs = inventory.get(pid, 0) + sales.get(pid, 0)
            self.inv_before_settle_log.setdefault(pid, []).append(inv_bs)
            self.sales_log.setdefault(pid, []).append(sales.get(pid, 0))

    def _get_delivery_lag(self, pid: str, observation: Mapping[str, Any]) -> float:
        """Return the delivery lag for ``pid`` from the observation."""
        # Store.observe() populates ``delivery_lags`` as of issue 01.
        delivery_lags = observation.get("delivery_lags", {})
        if pid in delivery_lags:
            return float(delivery_lags[pid])
        # Fallback: store-level scalar (older observations without per-pid lags).
        return float(observation.get("delivery_lag", 2))

    def _free_space(self, observation: Mapping[str, Any]) -> int:
        """Remaining capacity headroom = capacity - (on-hand + in-transit)."""
        inv = observation["inventory"]
        pending = observation.get("outstanding_orders", {})
        capacity = observation["max_capacity"]
        return max(0, int(capacity - sum(inv.values()) - sum(pending.values())))

    def decide(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Build one action dict for the given observation."""
        # Snapshot which pids are "already known" BEFORE updating logs.
        # A pid is cold-start if we have never completed a decide call for it.
        known_pids: set[str] = set(self.sales_log.keys())

        # Update rolling logs with current-tick data so rate estimates and
        # stockout detection use the freshest observation.
        self._update_logs(observation)

        active_pids: list[str] = list(observation["active_products"])
        K_active = max(1, len(active_pids))
        step: int = observation["current_sim_step"]
        inventory: Mapping[str, int] = observation["inventory"]
        pending: Mapping[str, int] = observation.get("outstanding_orders", {})
        balance: float = observation["balance"]
        unit_costs: Mapping[str, float] = observation["unit_costs"]
        base_prices: Mapping[str, float] = observation.get(
            "base_prices", observation.get("product_prices", {})
        )
        free_space = self._free_space(observation)

        desired: dict[str, int] = {}
        pilot_pids: set[str] = set()

        for pid in active_pids:
            cost = unit_costs.get(pid, 1.0)
            position = inventory.get(pid, 0) + pending.get(pid, 0)
            delivery_lag = self._get_delivery_lag(pid, observation)

            # Pilot pass: cold-start probe for never-previously-observed pids.
            # We use ``known_pids`` (snapshot before this tick's log update) so
            # that the very first decide call for a pid triggers the pilot even
            # though the log now has one entry (from the current tick's update).
            if pid not in known_pids:
                if self.opening_budget_pct > 0 and cost > 0:
                    pilot_qty = int(
                        self.opening_budget_pct * balance / K_active / cost
                    )
                    # Clamp to free space (shared pool — will be further
                    # adjusted by the allocator).
                    pilot_qty = min(pilot_qty, free_space)
                    if pilot_qty > 0:
                        desired[pid] = pilot_qty
                        pilot_pids.add(pid)
                continue

            # Rate estimate for steady-state pids.
            demand_window = max(1, int(delivery_lag))
            rates = _estimate_rate(
                sales_log={pid: self.sales_log[pid]},
                demand_window=demand_window,
                inv_before_settle={pid: self.inv_before_settle_log.get(pid, [])},
                stockout_safety_bonus_ticks=self.stockout_safety_bonus_ticks,
            )
            rate = rates.get(pid, 0.0)

            # Stockout-adaptive safety bump (opt-in).
            effective_safety = self.safety_lead_ticks
            if self.stockout_safety_bonus_ticks > 0 and _has_stockout(
                pid,
                self.sales_log,
                self.inv_before_settle_log,
                demand_window,
            ):
                effective_safety += self.stockout_safety_bonus_ticks

            # Trigger and quantity hooks — override in subclass.
            if self._trigger_with_safety(
                pid, step, position, rate, delivery_lag, effective_safety
            ):
                qty = self._quantity_with_levels(
                    pid, position, rate, delivery_lag, effective_safety
                )
                if qty > 0:
                    desired[pid] = qty

        # Allocate subject to capacity and cash pools.
        allocation = _allocate_two_pass_fair_share(
            desired=desired,
            unit_costs=dict(unit_costs),
            free_space=free_space,
            cash=balance,
            min_qty=self.min_qty,
            pilot_pids=pilot_pids,
        )

        # Emit flat prices for all active pids.
        price_out: dict[str, float] = {}
        for pid in active_pids:
            price_out[pid] = float(base_prices.get(pid, 0.0))

        return {
            "order": {pid: qty for pid, qty in allocation.items() if qty > 0},
            "price": price_out,
            "activate": [],
            "deactivate": [],
            "promotions": {},
        }

    # Delegation helpers so subclasses override the simple (_trigger, _quantity)
    # hooks without needing to know about delivery_lag or effective_safety.

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
        s = (delivery_lag + effective_safety) * rate
        S = s + self.cover_horizon_ticks * rate
        return self._quantity(pid, position, s, S)


# ─────────────────────────────────────────────────────────────────────────────
# OrderUpToPolicy — (s,S) continuous review
# ─────────────────────────────────────────────────────────────────────────────


class OrderUpToPolicy(TextbookReorderPolicy):
    """(s,S) continuous-review policy.

    Reorders whenever ``position < s`` and brings position up to ``S``.

    Reorder levels (demand-units framing):

        s = (delivery_lag + safety_lead_ticks) × rate
        S = s + cover_horizon_ticks × rate

    Both levels are computed per tick from the current rate estimate and
    the per-pid delivery lag read from the observation, so they are
    scale-invariant in capacity by construction.

    No additional kwargs beyond those on ``TextbookReorderPolicy``.
    """

    def _trigger(self, pid: str, step: int, position: int, s: float) -> bool:
        """Reorder when inventory position falls below s."""
        return position < s

    def _quantity(self, pid: str, position: int, s: float, S: float) -> int:
        """Order up to S — current position."""
        return max(0, int(round(S - position)))


# ─────────────────────────────────────────────────────────────────────────────
# ReorderPointPolicy — (s,Q) continuous review
# ─────────────────────────────────────────────────────────────────────────────


class ReorderPointPolicy(TextbookReorderPolicy):
    """(s,Q) continuous-review policy.

    Reorders whenever ``position < s`` and places a fixed quantity ``Q``.

    Reorder levels (demand-units framing):

        s = (delivery_lag + safety_lead_ticks) × rate
        Q = cover_horizon_ticks × rate  (if Q=None, computed per tick per pid)

    The defining (s,Q) property: the order quantity does NOT depend on how
    far below ``s`` the position fell — it is always the configured fixed
    quantity (or the rate-derived default when ``Q=None``).

    Additional kwargs beyond those on ``TextbookReorderPolicy``:

        Q: int | None = None
            Fixed reorder quantity.  When ``None`` (default), Q is computed
            per tick as ``int(round(cover_horizon_ticks × rate))``, so each
            cycle orders one cover-horizon's worth of demand.  Explicit
            integer values are honoured verbatim.
    """

    def __init__(self, *, Q: int | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.Q = Q

    def _trigger(self, pid: str, step: int, position: int, s: float) -> bool:
        """Reorder when inventory position falls below s."""
        return position < s

    def _quantity(self, pid: str, position: int, s: float, S: float) -> int:
        """Order a fixed quantity Q (or rate-derived default when Q is None)."""
        if self.Q is not None:
            return self.Q
        # Rate-derived default: one cover-horizon's worth of demand.
        # S - s = cover_horizon_ticks × rate (by construction in the base class).
        return max(0, int(round(S - s)))


# ─────────────────────────────────────────────────────────────────────────────
# PeriodicOrderUpToPolicy — (R,S) periodic review
# ─────────────────────────────────────────────────────────────────────────────


class PeriodicOrderUpToPolicy(TextbookReorderPolicy):
    """(R,S) periodic-review policy.

    Reviews inventory only every ``review_interval`` ticks; on review ticks
    it orders enough to bring position up to ``S``.  On non-review ticks no
    order fires regardless of position.

    Reorder level (demand-units framing):

        S = (delivery_lag + safety_lead_ticks + cover_horizon_ticks) × rate

    Additional kwargs beyond those on ``TextbookReorderPolicy``:

        review_interval: int | None = None
            Review cadence in ticks.  When ``None`` (default),
            ``review_interval`` equals the per-pid ``delivery_lag`` read from
            the observation (review at lead-time cadence).  Explicit positive
            integers are honoured verbatim.

    Note: the ``max(0, …)`` clamp on the quantity is essential here because
    the trigger fires unconditionally on review ticks — position may already
    exceed ``S`` (e.g. immediately after a pilot order) and we must not emit
    a negative order.
    """

    def __init__(self, *, review_interval: int | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
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
        """Fire on periodic schedule, ignoring position."""
        ri = self.review_interval if self.review_interval is not None else max(1, int(delivery_lag))
        return step % ri == 0

    def _trigger(self, pid: str, step: int, position: int, s: float) -> bool:
        """Not used — _trigger_with_safety is overridden directly."""
        # Required by abstract base; _trigger_with_safety overrides the call path.
        return False  # pragma: no cover

    def _quantity(self, pid: str, position: int, s: float, S: float) -> int:
        """Order up to S, clamped to zero (position may exceed S on review tick)."""
        return max(0, int(round(S - position)))
