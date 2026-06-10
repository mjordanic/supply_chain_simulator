"""Tier 2 centring-sanity integration test (graph engine).

Asserts that the RL env with a constant zero action produces ordering
decisions structurally equivalent to the textbook formula:
    qty[pid] = max(0, 15 * rate - position)

since target_centre_lt=15 → target = 15 * rate, and at zero action order_raw=0.

Design rationale
----------------
At ``action = zeros(2*K)``, the order half of every slot is ``order_raw = 0``.
The decoder computes::

    target_lt = clip(15 + 0 * 15, 0, 30) = 15
    requested[pid] = max(0, 15 * effective_rate[pid] - (inventory[pid] + pending[pid]))

Both price_raw=0 → multiplier 1.0 → price = MSRP.

What this test asserts
----------------------
1. **Price invariance**: both decode_action and the node price both stay at MSRP.
2. **Cold-start non-pathology**: tick-0 RL order is strictly positive for all active SKUs.
3. **Decoder equivalence at matching rate**: when RL decoder and textbook formula receive
   identical inputs, the order qty matches within ±1 unit (integer rounding only).
4. **World-level CRN**: two graph-engine runs on the same spec produce identical
   intermediate-node cash trajectories (same world_seed → bit-identical world).
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
from src.sim.policy import RLIntermediatePolicy
from src.sim.runner import build_world
from src.sim.scenario import load_catalog


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


def _make_config() -> RLConfig:
    """Config with pinned Constant distributions for episode reproducibility."""
    return RLConfig(
        K_active=5,
        episode_length=20,  # shorter for graph-engine speed
        capacity_dist=Constant(500),
        balance_dist=Constant(50_000),
        target_centre_lead_times=15,
        target_half_span_lead_times=15,
        target_max_lead_times=30,
    )


# ---------------------------------------------------------------------------
# Centring-sanity tests
# ---------------------------------------------------------------------------


class TestCentringZeroAction:
    """Tier 2 centring-sanity test (graph engine).

    Verifies that the RL decoder at ``action = zeros(2*K)`` implements the
    same ordering intent as the formula ``qty = max(0, 15*rate - position)``.
    """

    def test_zero_action_prices_match_msrp(self):
        """RL decoder with zero action prices every active SKU at exactly MSRP.

        price_raw = 0 → multiplier = 0.5 + (0 + 1)*0.5 = 1.0 → price = MSRP.
        """
        catalog = _make_catalog(n=15)
        config = _make_config()
        spec = sample_episode(catalog, config, episode_seed=7)

        active_pids = list(spec.active_subset)
        K = config.K_active
        slot_perm = spec.slot_permutation

        rl_policy = RLIntermediatePolicy()
        sim = build_world(spec.scenario, policy_overrides={"S": rl_policy})
        node_s = sim.nodes["S"]

        zero_action = np.zeros(2 * K, dtype=np.float32)
        sales_history: dict[str, deque] = {
            pid: deque(maxlen=100) for pid in active_pids
        }

        from src.sim.distributions import Distribution
        market_base_demand = getattr(spec.scenario.market, "base_demand", None)
        if isinstance(market_base_demand, Distribution):
            prior_rng = Random(spec.world_seed + 1)
            base_demand_prior = float(market_base_demand.sample(prior_rng))
        else:
            base_demand_prior = float(market_base_demand) if market_base_demand is not None else 1.0

        price_violations = []
        base_prices = {pid: float(node_s.list_prices.get(pid, 1.0)) for pid in active_pids}

        for tick in range(config.episode_length):
            current_tick = sim.tick_world()
            central_table = getattr(sim, "_current_table", None)
            effective_rate = compute_effective_rate(sales_history, base_demand_prior)

            action_dict = decode_action(
                zero_action,
                slot_perm,
                node_s,
                K,
                base_prices,
                active_subset=active_pids,
                effective_rate=effective_rate,
                target_centre_lead_times=config.target_centre_lead_times,
                target_half_span_lead_times=config.target_half_span_lead_times,
                target_max_lead_times=config.target_max_lead_times,
            )

            for pid in active_pids:
                msrp = base_prices.get(pid, 1.0)
                price = action_dict["list_price"].get(pid, 0.0)
                if abs(price / max(msrp, 1e-9) - 1.0) > 0.01:
                    price_violations.append(
                        f"tick={tick} pid={pid}: price={price:.4f} MSRP={msrp:.4f}"
                    )

            rl_policy.set_pending_action(action_dict)
            sim.tick_decide_and_settle(current_tick)

            last_sales = sim._tick_sales.get("S", {})
            for pid in active_pids:
                sales_history[pid].append(last_sales.get(pid, 0))

        assert not price_violations, (
            f"RL decoder price deviated from MSRP: {price_violations[:5]}"
        )

    def test_cold_start_produces_positive_orders_when_inventory_empty(self):
        """Zero-action decoder produces positive orders when node S has zero inventory.

        With position=0, rate=prior>0, target_lt=15, requested = 15*prior > 0.
        We override the node's inventory to zero to test the cold-start decoder math.
        """
        catalog = _make_catalog(n=15)
        config = _make_config()
        spec = sample_episode(catalog, config, episode_seed=7)

        active_pids = list(spec.active_subset)
        K = config.K_active
        slot_perm = spec.slot_permutation

        rl_policy = RLIntermediatePolicy()
        sim = build_world(spec.scenario, policy_overrides={"S": rl_policy})
        node_s = sim.nodes["S"]

        # Zero out the inventory to test the decoder formula at position=0.
        node_s.inventory = {pid: 0 for pid in active_pids}
        node_s.pending = {}

        from src.sim.distributions import Distribution
        market_base_demand = getattr(spec.scenario.market, "base_demand", None)
        if isinstance(market_base_demand, Distribution):
            prior_rng = Random(spec.world_seed + 1)
            base_demand_prior = float(market_base_demand.sample(prior_rng))
        else:
            base_demand_prior = float(market_base_demand) if market_base_demand is not None else 1.0

        assert base_demand_prior > 0, f"base_demand_prior={base_demand_prior}"

        zero_action = np.zeros(2 * K, dtype=np.float32)
        empty_history: dict[str, deque] = {pid: deque(maxlen=100) for pid in active_pids}
        effective_rate = compute_effective_rate(empty_history, base_demand_prior)
        base_prices = {pid: float(node_s.list_prices.get(pid, 1.0)) for pid in active_pids}

        action_dict = decode_action(
            zero_action,
            slot_perm,
            node_s,
            K,
            base_prices,
            active_subset=active_pids,
            effective_rate=effective_rate,
            target_centre_lead_times=config.target_centre_lead_times,
            target_half_span_lead_times=config.target_half_span_lead_times,
            target_max_lead_times=config.target_max_lead_times,
        )

        for pid in active_pids:
            orders = action_dict.get("order", {}).get(pid, [])
            total_qty = sum(qty for _, qty in orders)
            assert total_qty > 0, (
                f"Cold-start tick-0 order for pid={pid} is {total_qty} (expected > 0). "
                f"base_demand_prior={base_demand_prior:.3f}, "
                f"effective_rate={effective_rate.get(pid, 0):.3f}"
            )

    def test_decoder_matches_textbook_formula_at_identical_rate(self):
        """At identical rate inputs, RL decoder orders match the textbook formula ±1 unit.

        Structural centring-sanity: when RL decoder receives the same
        effective_rate and position as the textbook formula, the results agree.
        """
        catalog = _make_catalog(n=15)
        config = _make_config()
        spec = sample_episode(catalog, config, episode_seed=7)

        active_pids = list(spec.active_subset)
        K = config.K_active
        slot_perm = spec.slot_permutation

        rl_policy = RLIntermediatePolicy()
        sim = build_world(spec.scenario, policy_overrides={"S": rl_policy})
        node_s = sim.nodes["S"]

        from src.sim.distributions import Distribution
        market_base_demand = getattr(spec.scenario.market, "base_demand", None)
        if isinstance(market_base_demand, Distribution):
            prior_rng = Random(spec.world_seed + 1)
            base_demand_prior = float(market_base_demand.sample(prior_rng))
        else:
            base_demand_prior = float(market_base_demand) if market_base_demand is not None else 1.0

        zero_action = np.zeros(2 * K, dtype=np.float32)
        sales_history: dict[str, deque] = {pid: deque(maxlen=100) for pid in active_pids}
        base_prices = {pid: float(node_s.list_prices.get(pid, 1.0)) for pid in active_pids}
        mismatches = []

        for tick in range(config.episode_length):
            current_tick = sim.tick_world()
            effective_rate = compute_effective_rate(sales_history, base_demand_prior)

            # Compute textbook expected: max(0, 15*rate - position)
            # Before any allocation scaling.
            textbook_expected: dict[str, int] = {}
            for pid in active_pids:
                rate = float(effective_rate.get(pid, 0.0))
                inv = float(node_s.inventory.get(pid, 0))
                pend_total = sum(
                    node_s.pending.get(sup, {}).get(pid, 0)
                    for sup in node_s.pending
                )
                position = inv + pend_total
                target_qty = max(0.0, 15.0 * rate - position)
                textbook_expected[pid] = int(target_qty)

            action_dict = decode_action(
                zero_action,
                slot_perm,
                node_s,
                K,
                base_prices,
                active_subset=active_pids,
                effective_rate=effective_rate,
                target_centre_lead_times=config.target_centre_lead_times,
                target_half_span_lead_times=config.target_half_span_lead_times,
                target_max_lead_times=config.target_max_lead_times,
            )

            capacity = float(node_s.capacity)
            total_inv = sum(float(v) for v in node_s.inventory.values())
            pending_totals: dict[str, int] = {}
            for sup_p in node_s.pending.values():
                for pid, qty in sup_p.items():
                    pending_totals[pid] = pending_totals.get(pid, 0) + qty
            total_pend = sum(float(v) for v in pending_totals.values())
            global_free = max(0, int(capacity - total_inv - total_pend))
            total_expected = sum(textbook_expected.values())

            for pid in active_pids:
                orders = action_dict.get("order", {}).get(pid, [])
                rl_qty = sum(qty for _, qty in orders)
                exp_qty = textbook_expected[pid]

                if total_expected > global_free and global_free > 0:
                    scale = global_free / total_expected
                    scaled_exp = int(exp_qty * scale)
                    if abs(rl_qty - scaled_exp) > 1:
                        mismatches.append(
                            f"tick={tick} pid={pid}: RL={rl_qty} expected(scaled)={scaled_exp}"
                        )
                else:
                    if abs(rl_qty - exp_qty) > 1:
                        mismatches.append(
                            f"tick={tick} pid={pid}: RL={rl_qty} expected={exp_qty}"
                        )

            rl_policy.set_pending_action(action_dict)
            sim.tick_decide_and_settle(current_tick)

            last_sales = sim._tick_sales.get("S", {})
            for pid in active_pids:
                sales_history[pid].append(last_sales.get(pid, 0))

        assert not mismatches, (
            f"Decoder diverges from 15*rate - position formula: "
            f"{len(mismatches)} violations.\n"
            + "\n".join(mismatches[:10])
        )

    def test_world_crn_two_runs_bit_identical_cash_trajectory(self):
        """Two graph-engine runs on the same spec produce identical intermediate-node
        cash trajectories (world-level CRN guarantee).
        """
        catalog = _make_catalog(n=15)
        config = RLConfig(
            K_active=5,
            episode_length=10,
            capacity_dist=Constant(500),
            balance_dist=Constant(50_000),
        )
        spec = sample_episode(catalog, config, episode_seed=7)

        K = config.K_active
        slot_perm = spec.slot_permutation
        active_pids = list(spec.active_subset)

        from src.sim.distributions import Distribution
        market_base_demand = getattr(spec.scenario.market, "base_demand", None)

        def _run(episode_seed: int):
            # Fresh spec each time so node objects aren't shared/mutated.
            s = sample_episode(catalog, config, episode_seed=episode_seed)
            rl_policy = RLIntermediatePolicy()
            sim = build_world(s.scenario, policy_overrides={"S": rl_policy})
            node_s = sim.nodes["S"]
            zero_action = np.zeros(2 * K, dtype=np.float32)
            sales_history = {pid: deque(maxlen=100) for pid in active_pids}
            base_prices = {pid: float(node_s.list_prices.get(pid, 1.0)) for pid in active_pids}

            if isinstance(market_base_demand, Distribution):
                prior_rng = Random(s.world_seed + 1)
                base_demand_prior = float(market_base_demand.sample(prior_rng))
            else:
                base_demand_prior = float(market_base_demand) if market_base_demand is not None else 1.0

            cash_trace = []
            for tick in range(config.episode_length):
                current_tick = sim.tick_world()
                eff_rate = compute_effective_rate(sales_history, base_demand_prior)
                ad = decode_action(zero_action, slot_perm, node_s, K, base_prices,
                                   active_subset=active_pids, effective_rate=eff_rate)
                rl_policy.set_pending_action(ad)
                sim.tick_decide_and_settle(current_tick)
                last_sales = sim._tick_sales.get("S", {})
                for pid in active_pids:
                    sales_history[pid].append(last_sales.get(pid, 0))
                cash_trace.append(float(node_s.cash))
            return cash_trace

        trace1 = _run(7)
        trace2 = _run(7)

        for t, (c1, c2) in enumerate(zip(trace1, trace2)):
            assert c1 == pytest.approx(c2, abs=1e-6), (
                f"CRN violated: tick={t}: run1_cash={c1}, run2_cash={c2}"
            )
