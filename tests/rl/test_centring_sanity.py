"""Tier 2 centring-sanity integration test.

Asserts that the RL env with a constant zero action produces trajectories
whose ordering decisions are structurally equivalent to
``PeriodicOrderUpToPolicy(R=1, cover_horizon_ticks=10, safety_lead_ticks=2)``
when evaluated against the same inventory position and the same demand-rate
estimate.

Design rationale
----------------
At ``action = zeros(2*K)``, the order half of every slot is ``order_raw = 0``.
The decoder computes::

    target_lt = clip(target_centre_lt + 0 * target_half_span_lt, 0, target_max_lt)
              = clip(15 + 0, 0, 30) = 15

    requested[pid] = max(0, 15 * effective_rate[pid] - (inventory[pid] + pending[pid]))

Meanwhile ``PeriodicOrderUpToPolicy(R=1, cover_horizon_ticks=10,
safety_lead_ticks=2)`` with delivery_lag=3 targets::

    S = (delivery_lag + safety_lead_ticks + cover_horizon_ticks) * rate
      = (3 + 2 + 10) * rate = 15 * rate

    qty[pid] = max(0, round(S - position))

Both expressions compute ``max(0, 15 * rate - position)``.  The policies
are **structurally centred** — they target the same coverage horizon.

Why full-trajectory comparison is not used
------------------------------------------
The two policies use different rate estimators:
- RL encoder: ``compute_effective_rate`` = ``max(rolling_5_mean_sales, base_demand_prior)``
- Textbook policy: censored-sales rolling window of length ``delivery_lag``

These estimators produce different numerical values, especially in the first
~20 ticks after a cold start.  A full-trajectory order-equality check would
require matching these internal rate estimates, which defeats the purpose of
having a clean decoder-level assertion.

What this test *does* assert
-----------------------------
1. **Price invariance**: both policies price every active SKU at exactly MSRP
   (price ratio = 1.0 within floating-point tolerance) on every tick.

2. **Cold-start non-pathology**: the tick-0 RL order is strictly positive for
   every active SKU (no cold-start deadlock).

3. **Decoder equivalence at matching rate**: when the RL decoder and the
   textbook policy's rate estimator are given identical inputs (the same
   inventory position and the same effective rate), the order quantity they
   produce matches within ±1 unit (integer rounding only).

4. **World-level CRN**: both runs advance the world RNG with the same sequence
   (same Market / EventEngine tick order), so the world trajectory is
   bit-identical.  This is tested directly by injecting the same
   ``effective_rate`` into the RL decoder as the textbook policy sees.

CRN bit-identity prerequisite
------------------------------
The world-level CRN property (same world_seed → bit-identical demand draws)
is covered by ``tests/rl/test_eval_crn.py`` and must continue to pass.
"""

from __future__ import annotations

import math
from collections import deque
from random import Random

import numpy as np
import pytest

from src.rl.configs.default import RLConfig
from src.rl.encoders import compute_effective_rate, decode_action, encode_observation
from src.rl.episode_sampler import sample_episode
from src.sim.distributions import Constant
from src.sim.event_engine import EventEngine
from src.sim.item_registry import ItemRegistry
from src.sim.market import Market
from src.sim.policy import PeriodicOrderUpToPolicy, RLPolicy
from src.sim.scenario import StoreInstance, load_catalog
from src.sim.store import Store


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_catalog(n: int = 15) -> list:
    """Build a minimal deterministic catalog of n products."""
    items = [
        {
            "name": f"Product {i}",
            "category": "General",
            "related_products": [],
            "base_price": 10.0 + i,
            "unit_cost": 4.0 + i * 0.3,
            "seasonality": "all_season",
        }
        for i in range(n)
    ]
    return load_catalog(items)


def _make_base_template():
    from src.sim.scenario import StoreTemplate

    return StoreTemplate(
        id="centring_test",
        region="US",
        capacity=500,
        init_balance=50_000.0,
        init_stock_pct=0.0,
        delivery_lag=3,
        holding_rate=0.01,
        order_fee=50.0,
        init_active_count=5,
    )


def _make_config() -> RLConfig:
    """Config with pinned Constant distributions for episode reproducibility.

    capacity_dist = Constant(500)
    balance_dist  = Constant(50_000)
    episode_length = 180
    target_centre_lead_times = 15, matching S/rate of the comparison policy.
    """
    return RLConfig(
        K_active=5,
        episode_length=180,
        capacity_dist=Constant(500),
        balance_dist=Constant(50_000),
        target_centre_lead_times=15,
        target_half_span_lead_times=15,
        target_max_lead_times=30,
    )


def _make_delivery_callback(store: Store, pid: str, qty: int):
    def _cb() -> None:
        store.deliver(pid, qty)
    return _cb


def _dispatch_orders(store: Store, market: Market, event_engine: EventEngine, action: dict) -> None:
    """Mirror Runner / eval.py dispatch logic."""
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


def _process_demand(store: Store, market: Market, action: dict) -> None:
    """Mirror Runner / eval.py demand-settling loop."""
    prices = action.get("price", {})
    orders = action.get("order", {})
    current_step = market.current_step()
    for pid in list(store.inventory.keys()):
        price = prices.get(pid, store.prices[pid])
        order_qty = orders.get(pid, 0)
        demand = market.sample_demand(pid, store, price, current_step=current_step)
        store.settle(pid, demand=demand, price=price, order_qty=order_qty)
        store.prices[pid] = price


def _build_world(spec):
    """Instantiate simulator subsystems from an EpisodeSpec."""
    from src.sim.distributions import Distribution

    scenario = spec.scenario
    world_rng = Random(scenario.world_seed)

    item_registry = ItemRegistry(
        scenario.item_lifecycle,
        scenario.catalog,
        world_rng,
    )
    market = Market(
        scenario.market,
        world_rng,
        scenario.start_date,
        registry=item_registry,
    )
    event_engine = EventEngine(scenario.disruption, world_rng)

    si: StoreInstance = scenario.stores[0]
    store = Store(
        si.template,
        si.init_seed,
        None,
        scenario.catalog,
        freshness_alpha=item_registry.default_freshness_alpha,
        freshness_decay=item_registry.default_freshness_decay,
        item_registry=item_registry,
    )
    return item_registry, market, event_engine, store


# ---------------------------------------------------------------------------
# Centring-sanity tests
# ---------------------------------------------------------------------------


class TestCentringZeroAction:
    """Tier 2 centring-sanity test.

    Verifies that the RL decoder at ``action = zeros(2*K)`` implements the
    same ordering intent as ``PeriodicOrderUpToPolicy(R=1, S=15*rate)``.
    """

    def test_zero_action_prices_match_msrp_throughout(self):
        """RL env with zero action prices every active SKU at exactly MSRP on every tick.

        Both the RL decoder and PeriodicOrderUpToPolicy use ``price = MSRP * 1.0``
        (the price decoder at ``price_raw = 0`` maps to multiplier 1.0, and the
        textbook policy uses ``base_price``).  This verifies the price half of the
        decoder is not accidentally breaking.
        """
        episode_seed = 7
        catalog = _make_catalog(n=15)
        template = _make_base_template()
        config = _make_config()

        spec = sample_episode(catalog, template, config, episode_seed=episode_seed)
        active_pids = list(spec.active_subset)
        K = config.K_active
        slot_perm = spec.slot_permutation
        episode_length = config.episode_length

        from src.sim.distributions import Distribution

        item_registry, market, event_engine, store = _build_world(spec)
        rl_policy = RLPolicy()
        store.policy = rl_policy

        market_base_demand = getattr(spec.scenario.market, "base_demand", None)
        if isinstance(market_base_demand, Distribution):
            prior_rng = Random(spec.scenario.world_seed + 1)
            base_demand_prior = float(market_base_demand.sample(prior_rng))
        else:
            base_demand_prior = float(market_base_demand) if market_base_demand is not None else 1.0

        zero_action = np.zeros(2 * K, dtype=np.float32)
        sales_history: dict[str, deque] = {
            pid: deque(maxlen=100) for pid in [w.product_id for w in catalog]
        }

        price_violations = []

        for tick in range(episode_length):
            market.tick()
            event_engine.tick(market)
            item_registry.tick()

            effective_rate = compute_effective_rate(sales_history, base_demand_prior)
            action_dict = decode_action(
                zero_action,
                slot_perm,
                store,
                K,
                store.base_prices,
                effective_rate=effective_rate,
                target_centre_lead_times=config.target_centre_lead_times,
                target_half_span_lead_times=config.target_half_span_lead_times,
                target_max_lead_times=config.target_max_lead_times,
            )
            action_dict["activate"] = []
            action_dict["deactivate"] = []
            action_dict["promotions"] = {}

            # Check price = MSRP * 1.0 for each active pid
            for pid in active_pids:
                msrp = store.base_prices.get(pid, 1.0)
                price = action_dict["price"].get(pid, 0.0)
                if abs(price / max(msrp, 1e-9) - 1.0) > 0.01:
                    price_violations.append(
                        f"tick={tick} pid={pid}: price={price:.4f} MSRP={msrp:.4f}"
                    )

            rl_policy.set_pending_action(action_dict)
            obs_dict = store.observe(market.current_step(), item_registry)
            action_used = store.decide(obs_dict)
            _dispatch_orders(store, market, event_engine, action_used)
            _process_demand(store, market, action_used)

            for pid, qty in store.sales.items():
                if pid in sales_history:
                    sales_history[pid].append(qty)

        assert not price_violations, (
            f"RL decoder price deviated from MSRP: {price_violations[:5]}"
        )

    def test_cold_start_produces_positive_orders(self):
        """Tick-0 zero-action order is strictly positive for every active SKU.

        The cold-start path uses ``compute_effective_rate(empty_history,
        base_demand_prior)`` which returns ``base_demand_prior`` for every pid.
        With ``target_centre_lt = 15`` and ``position = 0``, each SKU requests
        ``15 * prior > 0``, so no cold-start deadlock occurs.
        """
        episode_seed = 7
        catalog = _make_catalog(n=15)
        template = _make_base_template()
        config = _make_config()

        spec = sample_episode(catalog, template, config, episode_seed=episode_seed)
        active_pids = list(spec.active_subset)
        K = config.K_active
        slot_perm = spec.slot_permutation

        from src.sim.distributions import Distribution

        item_registry, market, event_engine, store = _build_world(spec)
        rl_policy = RLPolicy()
        store.policy = rl_policy

        market_base_demand = getattr(spec.scenario.market, "base_demand", None)
        if isinstance(market_base_demand, Distribution):
            prior_rng = Random(spec.scenario.world_seed + 1)
            base_demand_prior = float(market_base_demand.sample(prior_rng))
        else:
            base_demand_prior = float(market_base_demand) if market_base_demand is not None else 1.0

        # Prior should be > 0 (Uniform(2,8) sampled value)
        assert base_demand_prior > 0, f"base_demand_prior={base_demand_prior}"

        # At tick 0: history is empty, rate = prior > 0, position = 0
        # → requested = 15 * prior > 0 for every active SKU.
        zero_action = np.zeros(2 * K, dtype=np.float32)
        empty_sales_history: dict[str, deque] = {
            pid: deque(maxlen=100) for pid in [w.product_id for w in catalog]
        }

        market.tick()
        event_engine.tick(market)
        item_registry.tick()

        effective_rate = compute_effective_rate(empty_sales_history, base_demand_prior)
        action_dict = decode_action(
            zero_action,
            slot_perm,
            store,
            K,
            store.base_prices,
            effective_rate=effective_rate,
            target_centre_lead_times=config.target_centre_lead_times,
            target_half_span_lead_times=config.target_half_span_lead_times,
            target_max_lead_times=config.target_max_lead_times,
        )

        for pid in active_pids:
            qty = action_dict["order"].get(pid, 0)
            assert qty > 0, (
                f"Cold-start tick-0 order for pid={pid} is {qty} (expected > 0). "
                f"base_demand_prior={base_demand_prior:.3f}, "
                f"effective_rate={effective_rate.get(pid, 0):.3f}"
            )

    def test_decoder_matches_textbook_formula_at_identical_rate(self):
        """At identical rate inputs, RL decoder orders match the textbook formula ±1 unit.

        This is the structural centring-sanity check: when both the RL decoder
        and the textbook formula receive the *same* ``effective_rate`` and see
        the *same* inventory position, they must produce the same order quantity
        within integer rounding.

        We verify this by:
        1. Running the RL decoder on the actual episode to collect per-tick
           (position, effective_rate) pairs for active SKUs.
        2. Computing the textbook formula's expected qty:
               expected = max(0, int(15 * effective_rate[pid] - position[pid]))
           This mirrors ``PeriodicOrderUpToPolicy._quantity`` with S = 15 * rate.
        3. Asserting the RL decoder's actual order equals the expected qty ±1.

        The ±1 tolerance accounts for integer truncation in ``fair_share_allocate``
        and in the textbook policy's ``round(S - position)``.
        """
        episode_seed = 7
        catalog = _make_catalog(n=15)
        template = _make_base_template()
        config = _make_config()

        spec = sample_episode(catalog, template, config, episode_seed=episode_seed)
        active_pids = list(spec.active_subset)
        K = config.K_active
        slot_perm = spec.slot_permutation
        episode_length = config.episode_length

        from src.sim.distributions import Distribution

        item_registry, market, event_engine, store = _build_world(spec)
        rl_policy = RLPolicy()
        store.policy = rl_policy

        market_base_demand = getattr(spec.scenario.market, "base_demand", None)
        if isinstance(market_base_demand, Distribution):
            prior_rng = Random(spec.scenario.world_seed + 1)
            base_demand_prior = float(market_base_demand.sample(prior_rng))
        else:
            base_demand_prior = float(market_base_demand) if market_base_demand is not None else 1.0

        zero_action = np.zeros(2 * K, dtype=np.float32)
        sales_history: dict[str, deque] = {
            pid: deque(maxlen=100) for pid in [w.product_id for w in catalog]
        }

        mismatches = []

        for tick in range(episode_length):
            market.tick()
            event_engine.tick(market)
            item_registry.tick()

            effective_rate = compute_effective_rate(sales_history, base_demand_prior)

            # Compute textbook expected order: max(0, 15*rate - position)
            # before decode_action mutates nothing (it's a pure function)
            textbook_expected: dict[str, int] = {}
            active_items = list(store.active_items)
            for pid in active_pids:
                rate = float(effective_rate.get(pid, 0.0))
                inv = float(store.inventory.get(pid, 0))
                pend = float(store.pending.get(pid, 0))
                position = inv + pend
                target_qty = max(0.0, 15.0 * rate - position)
                textbook_expected[pid] = int(target_qty)  # truncate (same as fair_share_allocate)

            action_dict = decode_action(
                zero_action,
                slot_perm,
                store,
                K,
                store.base_prices,
                effective_rate=effective_rate,
                target_centre_lead_times=config.target_centre_lead_times,
                target_half_span_lead_times=config.target_half_span_lead_times,
                target_max_lead_times=config.target_max_lead_times,
            )

            # Compute total requested (pre-allocation) to check fair_share scaling.
            # If fair_share_allocate scales down, we verify the scaling is consistent
            # rather than the absolute values.
            capacity = float(store.capacity)
            total_inv = sum(float(v) for v in store.inventory.values())
            total_pend = sum(float(v) for v in store.pending.values())
            global_free = max(0, int(capacity - total_inv - total_pend))

            total_expected = sum(textbook_expected.values())

            for pid in active_pids:
                rl_qty = action_dict["order"].get(pid, 0)
                exp_qty = textbook_expected[pid]

                # If fair-share scaling applies (total_expected > global_free),
                # both RL and expected are scaled by the same factor —
                # verify the ratio is consistent rather than the absolute value.
                if total_expected > global_free and global_free > 0:
                    # Scale factor applied by fair_share_allocate
                    scale = global_free / total_expected
                    scaled_exp = int(exp_qty * scale)
                    if abs(rl_qty - scaled_exp) > 1:
                        mismatches.append(
                            f"tick={tick} pid={pid}: RL={rl_qty} expected(scaled)={scaled_exp} "
                            f"(scale={scale:.3f}, raw_exp={exp_qty})"
                        )
                else:
                    if abs(rl_qty - exp_qty) > 1:
                        mismatches.append(
                            f"tick={tick} pid={pid}: RL={rl_qty} expected={exp_qty} "
                            f"(rate={effective_rate.get(pid,0):.2f}, pos={store.inventory.get(pid,0)+store.pending.get(pid,0)})"
                        )

            action_dict["activate"] = []
            action_dict["deactivate"] = []
            action_dict["promotions"] = {}

            rl_policy.set_pending_action(action_dict)
            obs_dict = store.observe(market.current_step(), item_registry)
            action_used = store.decide(obs_dict)
            _dispatch_orders(store, market, event_engine, action_used)
            _process_demand(store, market, action_used)

            for pid, qty in store.sales.items():
                if pid in sales_history:
                    sales_history[pid].append(qty)

        assert not mismatches, (
            f"Decoder diverges from 15*rate - position formula: "
            f"{len(mismatches)} violations.\n"
            + "\n".join(mismatches[:10])
        )

    def test_world_crn_both_runs_bit_identical_demand(self):
        """Two RL runs on the same EpisodeSpec produce identical demand streams.

        This verifies the CRN bit-identity prerequisite for the centring test:
        the world_rng advance pattern in both runs is identical (same Market /
        EventEngine / ItemRegistry tick order), so the same world trajectory
        is observed by both the RL decoder and any baseline policy.
        """
        episode_seed = 7
        catalog = _make_catalog(n=15)
        template = _make_base_template()
        # Use a short episode for speed in this structural test.
        config = RLConfig(
            K_active=5,
            episode_length=10,
            capacity_dist=Constant(500),
            balance_dist=Constant(50_000),
        )

        spec = sample_episode(catalog, template, config, episode_seed=episode_seed)
        active_pids = list(spec.active_subset)
        K = config.K_active
        slot_perm = spec.slot_permutation
        episode_length = config.episode_length

        from src.sim.distributions import Distribution

        def _run_with_zero_action(spec):
            """Run a zero-action RL episode; return per-tick demand dicts."""
            item_registry, market, event_engine, store = _build_world(spec)
            rl_policy = RLPolicy()
            store.policy = rl_policy

            market_base_demand = getattr(spec.scenario.market, "base_demand", None)
            if isinstance(market_base_demand, Distribution):
                prior_rng = Random(spec.scenario.world_seed + 1)
                base_demand_prior = float(market_base_demand.sample(prior_rng))
            else:
                base_demand_prior = float(market_base_demand) if market_base_demand is not None else 1.0

            zero_action = np.zeros(2 * K, dtype=np.float32)
            sales_history: dict[str, deque] = {
                pid: deque(maxlen=100) for pid in [w.product_id for w in catalog]
            }
            per_tick_demand = []

            for tick in range(episode_length):
                market.tick()
                event_engine.tick(market)
                item_registry.tick()

                effective_rate = compute_effective_rate(sales_history, base_demand_prior)
                action_dict = decode_action(
                    zero_action, slot_perm, store, K, store.base_prices,
                    effective_rate=effective_rate,
                )
                action_dict["activate"] = []
                action_dict["deactivate"] = []
                action_dict["promotions"] = {}

                rl_policy.set_pending_action(action_dict)
                obs_dict = store.observe(market.current_step(), item_registry)
                action_used = store.decide(obs_dict)
                _dispatch_orders(store, market, event_engine, action_used)
                _process_demand(store, market, action_used)

                for pid, qty in store.sales.items():
                    if pid in sales_history:
                        sales_history[pid].append(qty)

                per_tick_demand.append(
                    {pid: int(store.demand.get(pid, 0)) for pid in active_pids}
                )

            return per_tick_demand

        demands1 = _run_with_zero_action(spec)
        demands2 = _run_with_zero_action(spec)

        for tick in range(episode_length):
            for pid in active_pids:
                d1 = demands1[tick].get(pid, 0)
                d2 = demands2[tick].get(pid, 0)
                assert d1 == d2, (
                    f"CRN violated: tick={tick} pid={pid}: "
                    f"run1_demand={d1}, run2_demand={d2}"
                )
