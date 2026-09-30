"""Fit a latent planner and a direct planner on the transition log.

Seams, from the issue acceptance criteria:

1. Fit to an order. A tiny log from ``Runner`` can be fit. Attached with
   ``policy_overrides``, the Policy returns ``list_price`` from ``base_price``,
   an empty promotion map, a zero minimum-order floor, and quantities that
   fit in free capacity and cash.
2. Features. The encoder sees on-hand, an arrival-age profile, a sales
   window whose length is its own knob, cash, capacity, and the delivery
   lead time. A market-multiplier column, a disruption column, and a
   calendar column are not features: two fits that differ only in those
   columns place the same orders.
3. Horizon. The predictor unrolls open-loop for ``H = delivery_lag`` with a
   stopped target embedding and an anti-collapse penalty. Loss values,
   latent width, and EMA decay are not asserted (PRD testing decision).
4. Probes. Fitting probes does not change encoder parameters. Profit is
   arithmetic on decoded quantities, not a learned reward head.
5. Retarget. On a fixture where every proposal still fits at
   ``holding_rate = 0.01`` and the till empties at ``0.05``, a frozen
   Policy places a different order at the higher rate, without a refit.
6. Search. The published budget is 64 scored proposals. Smoke tests
   use 4. The same ``policy_seed`` on the same episode repeats the same
   orders. The search follows imagined scores, so a different world can
   change later proposals. ``policy_rng`` is still the only randomness.
7. Direct predictor. Trained on the same log with no latent target, it is
   a second Policy with the same search and the same priority ration.
8. Fair-share arm. The latent model's covers can be rationed with textbook
   fair-share and still return a valid order dict.
9. The RL stack is not imported.
"""

from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import torch

from src.jepa.dynamics import sku_features
from src.jepa.log import collect_transition_log
from src.jepa.plan import DEFAULT_BUDGET, ReplenishmentPlanner, fit_direct, fit_latent
from src.sim.central_table import CentralTable, Offer
from src.sim.distributions import Constant
from src.sim.inspect import closing_inventory_frame, flow_frame, purchase_frame
from src.sim.metrics import profit_decomposition
from src.sim.runner import Runner
from src.tuning.config import TuningConfig
from src.tuning.episode import make_synthetic_catalog, sample_episode


def _tiny_log(lag: int = 2):
    return collect_transition_log(
        n_episodes_each=1,
        episode_length=lag + 3,
        k_active=2,
        delivery_lag=lag,
        seed_offset=14_000_000,
    )


def test_tiny_runner_log_fits_and_orders_within_capacity_and_cash():
    lag = 2
    frame = _tiny_log(lag)
    planner = fit_latent(
        frame,
        delivery_lag=lag,
        epochs=2,
        seed=0,
        policy_seed=3,
        budget=4,
    )

    config = TuningConfig(
        episode_length=3,
        K_active=2,
        delivery_lag=lag,
        capacity_dist=Constant(50_000),
        balance_dist=Constant(1_000_000),
        init_stock_pct_dist=Constant(0.0),
        holding_rate=0.01,
        order_fee=50.0,
    )
    spec = sample_episode(make_synthetic_catalog(8), config, episode_seed=14_000_100)
    catalog_price = {ware.product_id: float(ware.base_price) for ware in spec.scenario.catalog}
    log = Runner(spec.scenario, policy_overrides={"S": planner}).run()

    assert len(planner.proposals) == config.episode_length
    for tick, proposal in zip(log["ticks"], planner.proposals, strict=True):
        action = proposal["action"]
        pids = set(spec.active_subset)
        assert set(action["list_price"]) == pids
        for pid in pids:
            assert action["list_price"][pid] == catalog_price[pid]
        assert action["promotions"] == {}
        assert action["min_order_imposed"] == {pid: 0 for pid in pids}

        on_hand = {pid: int(tick["node_inventory"]["S"][pid]) for pid in pids}
        pending_after = tick["node_pending"]["S"]
        filled = {
            pid: sum(
                int(row["qty_filled"])
                for row in tick["purchases"]
                if row["buyer_id"] == "S" and row["pid"] == pid
            )
            for pid in pids
        }
        in_transit_before = {
            pid: int(pending_after.get(pid, 0)) - filled[pid] for pid in pids
        }
        free_space = 50_000 - sum(on_hand.values()) - sum(in_transit_before.values())
        requested = {
            pid: sum(int(qty) for _supplier, qty in action["order"].get(pid, []))
            for pid in pids
        }
        assert sum(requested.values()) <= max(0, free_space)
        assert tick["node_orders"]["S"] == {
            pid: qty for pid, qty in requested.items() if qty > 0
        }
        spent = sum(float(row["cash_paid"]) for row in tick["purchases"] if row["buyer_id"] == "S")
        assert spent <= 1_000_000.0

    profit = profit_decomposition(
        flow_frame(log),
        purchase_frame(log),
        closing_inventory_frame(log),
        spec.scenario,
    )
    net = float(profit.loc[profit["node_id"] == "S", "net_profit"].iloc[0])
    assert math.isfinite(net / spec.balance)


def _table(offers: dict[str, tuple[str, float]]) -> CentralTable:
    table = CentralTable()
    for pid, (supplier_id, price) in offers.items():
        table.publish(
            supplier_id,
            pid,
            Offer(available_qty=10_000, list_price=price, min_order=0),
        )
    return table


def _obs(
    *,
    inventory: dict[str, int],
    observed_sales: dict[str, int] | None = None,
    capacity: int = 100_000,
    cash: float = 1_000_000.0,
    base_prices: dict[str, float] | None = None,
    list_prices: dict[str, float] | None = None,
    pending: dict | None = None,
) -> dict:
    prices = base_prices or {pid: 10.0 for pid in inventory}
    obs: dict = {
        "tick": 0,
        "inventory": dict(inventory),
        "pending": pending or {},
        "observed_sales": observed_sales or {pid: 0 for pid in inventory},
        "list_prices": dict(list_prices or prices),
        "capacity": capacity,
        "cash": cash,
        "direct_supplier_ids": {f"F_{pid}" for pid in inventory},
    }
    if base_prices is not None:
        obs["base_prices"] = dict(base_prices)
    return obs


def _granted(action: dict, pid: str) -> int:
    return sum(qty for _supplier, qty in action["order"].get(pid, []))


def test_encoder_features_are_the_observation_window_without_market_or_disruption():
    lag = 2
    frame = _tiny_log(lag)
    other = frame.copy()
    other["market_multiplier"] = 3.5
    other["disruption"] = "port_closure"
    other["tick"] = 987654
    other["simulation_date"] = "2026-09-30"
    kwargs = dict(delivery_lag=lag, epochs=2, seed=0, policy_seed=1, budget=4)
    plain = fit_latent(frame, **kwargs)
    marked = fit_latent(other, **kwargs)
    obs = _obs(
        inventory={"A": 0, "B": 0},
        observed_sales={"A": 4, "B": 4},
        base_prices={"A": 12.0, "B": 9.0},
        list_prices={"A": 1.0, "B": 1.0},
    )
    table = _table({"A": ("F_A", 4.0), "B": ("F_B", 3.0)})
    sample = [
        sku_features(
            on_hand=1,
            in_transit=[0, 2],
            sales_window=[1, 2],
            cash=10,
            capacity=20,
            delivery_lag=lag,
        )
    ]
    assert plain.dynamics.predict_next(sample, [3.0]) == marked.dynamics.predict_next(sample, [3.0])
    assert plain.decide(obs, table) == marked.decide(obs, table)


def test_fitting_probes_does_not_change_the_encoder():
    lag = 2
    frame = _tiny_log(lag)
    planner = fit_latent(frame, delivery_lag=lag, epochs=2, seed=0, policy_seed=1, budget=4)
    encoder_before = [parameter.detach().clone() for parameter in planner.dynamics.encoder.parameters()]
    with torch.no_grad():
        for parameter in planner.dynamics.probes.parameters():
            parameter.add_(1.0)
    probe_before = [parameter.detach().clone() for parameter in planner.dynamics.probes.parameters()]

    planner.fit_probes(frame, epochs=5)

    for before, after in zip(encoder_before, planner.dynamics.encoder.parameters(), strict=True):
        assert torch.equal(before, after)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(probe_before, planner.dynamics.probes.parameters(), strict=True)
    )

    planner.dynamics.encoder.zero_grad(set_to_none=True)
    planner.decide(
        _obs(inventory={"A": 1}, observed_sales={"A": 2}, base_prices={"A": 8.0}),
        _table({"A": ("F_A", 4.0)}),
    )
    for parameter in planner.dynamics.encoder.parameters():
        assert parameter.grad is None


class _CashOracle:
    """Granted units arrive after ``delivery_lag`` steps and sell at ``demand``."""

    def __init__(self) -> None:
        self.delivery_lag = 2
        self.demand = 10.0

    def open_rollout(self, features: list[list[float]]) -> "_CashRollout":
        return _CashRollout(self, features)


class _CashRollout:
    def __init__(self, dynamics: _CashOracle, features: list[list[float]]) -> None:
        self.demand = dynamics.demand
        self.rows = [[float(feat[0]), [0.0] * dynamics.delivery_lag] for feat in features]

    def step(self, granted: list[float]) -> list[tuple[float, float, float]]:
        predicted: list[tuple[float, float, float]] = []
        for row, qty in zip(self.rows, granted, strict=True):
            on_hand, profile = row
            arriving = float(profile.pop(0))
            profile.append(float(qty))
            on_hand += arriving
            sales = min(on_hand, self.demand)
            row[0] = on_hand - sales
            predicted.append((row[0], float(sum(profile)), sales))
        return predicted


def test_frozen_policy_orders_less_when_the_higher_holding_rate_empties_the_till():
    """On-hand 10, unit cost 10, cash 12. The search wants a cover of 4.

    Holding on those 10 units is 1 at rate 0.01 and 5 at 0.05. After the
    charge the cheap till buys one unit and the expensive till buys none.
    """
    lag = 2
    low = ReplenishmentPlanner(
        _CashOracle(),
        policy_seed=0,
        delivery_lag=lag,
        budget=4,
        holding_rate=0.01,
        order_fee=0.0,
    )
    high = low.retarget(0.05)
    assert high.dynamics is low.dynamics

    obs = _obs(
        inventory={"A": 10},
        observed_sales={"A": 10},
        capacity=10_000,
        cash=12.0,
        base_prices={"A": 30.0},
        list_prices={"A": 1.0},
    )
    table = _table({"A": ("F_A", 10.0)})
    low_action = low.decide(obs, table)
    high_action = high.decide(obs, table)

    assert low.proposals[-1]["draws"] == high.proposals[-1]["draws"]
    assert _granted(low_action, "A") == 1
    assert _granted(high_action, "A") == 0
    assert low_action["list_price"] == {"A": 30.0}
    assert high_action["list_price"] == {"A": 30.0}


def test_published_budget_is_64_and_the_same_policy_seed_repeats_orders():
    lag = 2
    frame = _tiny_log(lag)
    fitted = fit_latent(frame, delivery_lag=lag, epochs=1, seed=0, policy_seed=0)
    assert fitted.budget == DEFAULT_BUDGET == 64
    obs = _obs(inventory={"A": 0, "B": 0}, observed_sales={"A": 3, "B": 3})
    table = _table({"A": ("F_A", 4.0), "B": ("F_B", 4.0)})
    fitted.decide(obs, table)
    assert len(fitted.proposals[-1]["draws"]) == 64

    config = TuningConfig(
        episode_length=3,
        K_active=2,
        delivery_lag=lag,
        capacity_dist=Constant(50_000),
        balance_dist=Constant(1_000_000),
        init_stock_pct_dist=Constant(0.0),
    )
    spec = sample_episode(make_synthetic_catalog(8), config, episode_seed=14_000_200)
    world_seed = spec.scenario.world_seed

    def run(seed_world: int, policy_seed: int):
        scenario = replace(spec.scenario, world_seed=seed_world)
        planner = ReplenishmentPlanner(
            fitted.dynamics,
            policy_seed=policy_seed,
            delivery_lag=lag,
            budget=8,
            holding_rate=0.01,
        )
        log = Runner(scenario, policy_overrides={"S": planner}).run()
        orders = [dict(tick["node_orders"]["S"]) for tick in log["ticks"]]
        return orders, [dict(draw) for step in planner.proposals for draw in step["draws"]]

    orders_a, draws_a = run(world_seed, 7)
    orders_b, draws_b = run(world_seed, 7)
    assert orders_a == orders_b
    assert draws_a == draws_b

    # Later waves follow imagined scores, so the world can change the
    # proposal sequence. A different policy seed still changes the draws.
    _orders_d, draws_d = run(world_seed, 8)
    assert draws_d != draws_a


def _positive_equal_cover_seed(fitter, frame, lag: int) -> int:
    obs = _obs(inventory={"A": 0, "B": 0}, observed_sales={"A": 6, "B": 6})
    table = _table({"A": ("F_A", 4.0), "B": ("F_B", 4.0)})
    fitted = fitter(frame, delivery_lag=lag, epochs=1, seed=0, budget=1, policy_seed=0)
    for seed in range(300):
        planner = ReplenishmentPlanner(
            fitted.dynamics,
            policy_seed=seed,
            delivery_lag=lag,
            budget=1,
            holding_rate=0.01,
        )
        planner.decide(obs, table)
        covers = planner.proposals[-1]["draws"][0]["covers"]
        if covers["A"] == covers["B"] and covers["A"] > 0:
            return seed
    raise AssertionError("no seed drew equal positive covers")


def test_direct_predictor_uses_the_same_priority_ration():
    lag = 2
    frame = _tiny_log(lag)
    seed = _positive_equal_cover_seed(fit_direct, frame, lag)
    sales = 6
    fitted = fit_direct(frame, delivery_lag=lag, epochs=1, seed=0, budget=1, policy_seed=0)
    drawer = ReplenishmentPlanner(
        fitted.dynamics,
        policy_seed=seed,
        delivery_lag=lag,
        budget=1,
        holding_rate=0.01,
    )
    loose = _obs(inventory={"A": 0, "B": 0}, observed_sales={"A": sales, "B": sales})
    table = _table({"A": ("F_A", 4.0), "B": ("F_B", 4.0)})
    drawer.decide(loose, table)
    cover = drawer.proposals[-1]["draws"][0]["covers"]["A"]
    desired = max(0, round(cover * float(sales) * lag))
    assert desired > 1

    tight = _obs(
        inventory={"A": 0, "B": 0},
        observed_sales={"A": sales, "B": sales},
        capacity=desired,
        cash=float(desired * 4),
        base_prices={"A": 12.0, "B": 9.0},
        list_prices={"A": 1.0, "B": 1.0},
    )
    planner = ReplenishmentPlanner(
        fitted.dynamics,
        policy_seed=seed,
        delivery_lag=lag,
        budget=1,
        holding_rate=0.01,
    )
    action = planner.decide(tight, table)
    top, second = planner.proposals[-1]["draws"][0]["priority"]
    assert _granted(action, top) == desired
    assert _granted(action, second) == 0
    assert action["list_price"] == {"A": 12.0, "B": 9.0}
    assert action["promotions"] == {}
    assert action["min_order_imposed"] == {"A": 0, "B": 0}
    spent = (_granted(action, "A") + _granted(action, "B")) * 4.0
    assert spent <= tight["cash"]


def test_fair_share_arm_splits_a_pool_priority_gives_to_one_sku():
    lag = 2
    frame = _tiny_log(lag)
    seed = _positive_equal_cover_seed(fit_latent, frame, lag)
    sales = 6
    fitted = fit_latent(frame, delivery_lag=lag, epochs=1, seed=0, budget=1, policy_seed=0)
    drawer = ReplenishmentPlanner(
        fitted.dynamics,
        policy_seed=seed,
        delivery_lag=lag,
        budget=1,
    )
    table = _table({"A": ("F_A", 4.0), "B": ("F_B", 4.0)})
    drawer.decide(
        _obs(inventory={"A": 0, "B": 0}, observed_sales={"A": sales, "B": sales}),
        table,
    )
    cover = drawer.proposals[-1]["draws"][0]["covers"]["A"]
    desired = max(0, round(cover * float(sales) * lag))
    tight = _obs(
        inventory={"A": 0, "B": 0},
        observed_sales={"A": sales, "B": sales},
        capacity=desired,
        cash=1_000_000.0,
    )
    priority = ReplenishmentPlanner(
        fitted.dynamics,
        policy_seed=seed,
        delivery_lag=lag,
        budget=1,
        ration="priority",
    )
    fair = ReplenishmentPlanner(
        fitted.dynamics,
        policy_seed=seed,
        delivery_lag=lag,
        budget=1,
        ration="fair_share",
    )
    priority_action = priority.decide(tight, table)
    fair_action = fair.decide(tight, table)
    top, second = priority.proposals[-1]["draws"][0]["priority"]
    assert _granted(priority_action, top) == desired
    assert _granted(priority_action, second) == 0
    assert _granted(fair_action, "A") > 0
    assert _granted(fair_action, "B") > 0
    assert _granted(fair_action, "A") + _granted(fair_action, "B") <= desired
    assert fair_action["order"] != priority_action["order"]


def test_direct_and_fair_share_order_through_policy_overrides():
    lag = 2
    frame = _tiny_log(lag)
    config = TuningConfig(
        episode_length=3,
        K_active=2,
        delivery_lag=lag,
        capacity_dist=Constant(50_000),
        balance_dist=Constant(1_000_000),
        init_stock_pct_dist=Constant(0.0),
        holding_rate=0.01,
        order_fee=50.0,
    )
    catalog = make_synthetic_catalog(8)
    arms = (
        fit_direct(frame, delivery_lag=lag, epochs=1, seed=0, policy_seed=3, budget=4),
        fit_latent(
            frame,
            delivery_lag=lag,
            epochs=1,
            seed=0,
            policy_seed=3,
            budget=4,
            ration="fair_share",
        ),
    )
    for planner in arms:
        spec = sample_episode(catalog, config, episode_seed=14_000_100)
        catalog_price = {ware.product_id: float(ware.base_price) for ware in spec.scenario.catalog}
        log = Runner(spec.scenario, policy_overrides={"S": planner}).run()
        assert len(planner.proposals) == config.episode_length
        for tick, proposal in zip(log["ticks"], planner.proposals, strict=True):
            action = proposal["action"]
            pids = set(spec.active_subset)
            assert set(action["list_price"]) == pids
            for pid in pids:
                assert action["list_price"][pid] == catalog_price[pid]
            assert action["promotions"] == {}
            assert action["min_order_imposed"] == {pid: 0 for pid in pids}
            requested = sum(
                qty for lines in action["order"].values() for _supplier, qty in lines
            )
            on_hand = sum(int(tick["node_inventory"]["S"][pid]) for pid in pids)
            pending_after = sum(int(tick["node_pending"]["S"].get(pid, 0)) for pid in pids)
            filled = sum(
                int(row["qty_filled"])
                for row in tick["purchases"]
                if row["buyer_id"] == "S"
            )
            free_space = 50_000 - on_hand - (pending_after - filled)
            assert requested <= max(0, free_space)
            spent = sum(
                float(row["cash_paid"]) for row in tick["purchases"] if row["buyer_id"] == "S"
            )
            assert spent <= 1_000_000.0


def test_plan_package_does_not_import_the_rl_stack():
    for path in Path("src/jepa").glob("*.py"):
        text = path.read_text()
        assert "src.rl" not in text
        assert "optuna" not in text

    script = (
        "import src.jepa.plan, sys\n"
        "bad = [name for name in sys.modules if name == 'src.rl' or name.startswith('src.rl.')]\n"
        "assert not bad, bad\n"
    )
    subprocess.check_call([sys.executable, "-c", script])
