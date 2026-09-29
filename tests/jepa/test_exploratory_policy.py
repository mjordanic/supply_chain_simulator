"""Exploratory replenishment policy.

Seams, from the issue acceptance criteria:

1. Cover formula. Every tick the policy draws a cover in ``{0, 1, 2, 4}``
   and, before the ration, orders
   ``max(0, round(cover × censored-sales rate × horizon) − position)``
   with position = on-hand + in-transit. ``horizon`` is ``delivery_lag``
   unless ``planning_depth`` is set. The rate is the textbook
   censored-sales mean over ``delivery_lag``, including the current tick.
2. Ordering only. ``list_price`` equals ``base_price``, promotions are
   empty, and ``min_order_imposed`` is zero.
3. Priority ration. Granted units fit in free capacity
   (capacity − on-hand − in-transit) and in cash. With room for one cover,
   the higher priority takes the whole pool.
4. Determinism. The same ``policy_seed`` and world seed place the same
   orders. Changing only the world seed leaves the proposal sequence
   (covers and priority) unchanged.
"""

from __future__ import annotations

from dataclasses import replace

from src.jepa.policy import ExploratoryPolicy
from src.sim.central_table import CentralTable, Offer
from src.sim.distributions import Constant
from src.sim.runner import Runner
from src.tuning.config import TuningConfig
from src.tuning.episode import make_synthetic_catalog, sample_episode


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
    pending: dict | None = None,
    observed_sales: dict[str, int] | None = None,
    capacity: int = 100_000,
    cash: float = 1_000_000.0,
    base_prices: dict[str, float] | None = None,
    list_prices: dict[str, float] | None = None,
    tick: int = 0,
) -> dict:
    prices = base_prices or {pid: 10.0 for pid in inventory}
    obs: dict = {
        "tick": tick,
        "inventory": dict(inventory),
        "pending": pending or {},
        "observed_sales": observed_sales or {pid: 0 for pid in inventory},
        "list_prices": dict(list_prices or prices),
        "capacity": capacity,
        "cash": cash,
        "direct_supplier_ids": {f"F_{pid}" for pid in inventory},
        "min_order_imposed": {pid: 5 for pid in inventory},
    }
    if base_prices is not None:
        obs["base_prices"] = dict(base_prices)
    return obs


def _granted(action: dict, pid: str) -> int:
    return sum(qty for _supplier, qty in action["order"].get(pid, []))


def test_order_before_ration_is_cover_in_lead_times_of_censored_sales():
    """Position is on-hand plus in-transit. The rate is this tick's sales
    when the log is shorter than the lead time. A loose till does not ration.
    """
    lag = 3
    policy = ExploratoryPolicy(policy_seed=0, delivery_lag=lag)
    obs = _obs(
        inventory={"A": 4, "B": 1},
        pending={"F_A": {"A": 2}},
        observed_sales={"A": 5, "B": 3},
        base_prices={"A": 12.0, "B": 9.0},
    )
    action = policy.decide(obs, _table({"A": ("F_A", 4.0), "B": ("F_B", 3.0)}))

    proposal = policy.proposals[-1]
    assert set(proposal["covers"]) == {"A", "B"}
    assert set(proposal["covers"].values()) <= {0, 1, 2, 4}

    expected_position = {"A": 4 + 2, "B": 1}
    expected_rate = {"A": 5.0, "B": 3.0}
    for pid in ("A", "B"):
        cover = proposal["covers"][pid]
        target = round(cover * expected_rate[pid] * lag)
        expected = max(0, target - expected_position[pid])
        assert _granted(action, pid) == expected


def test_rate_is_the_censored_mean_over_the_lead_time():
    """After the window fills, the rate is the mean of the last ``delivery_lag``
    sales, not the latest tick alone.
    """
    lag = 2
    policy = ExploratoryPolicy(policy_seed=1, delivery_lag=lag)
    sales_by_tick = ({"A": 2}, {"A": 4}, {"A": 6})
    table = _table({"A": ("F_A", 4.0)})
    action = None
    for tick, sales in enumerate(sales_by_tick):
        action = policy.decide(
            _obs(inventory={"A": 0}, observed_sales=sales, tick=tick),
            table,
        )
    assert action is not None
    cover = policy.proposals[-1]["covers"]["A"]
    # Window is the last two ticks: 4 and 6. Mean rate is 5.
    expected = max(0, round(cover * 5.0 * lag) - 0)
    assert _granted(action, "A") == expected


def test_prices_stay_at_base_price_with_no_promotion_or_minimum():
    policy = ExploratoryPolicy(policy_seed=2, delivery_lag=3)
    action = policy.decide(
        _obs(
            inventory={"A": 0, "B": 0},
            observed_sales={"A": 1, "B": 1},
            base_prices={"A": 12.0, "B": 9.0},
            list_prices={"A": 1.0, "B": 1.0},
        ),
        _table({"A": ("F_A", 4.0), "B": ("F_B", 3.0)}),
    )
    assert action["list_price"] == {"A": 12.0, "B": 9.0}
    assert action["promotions"] == {}
    assert action["min_order_imposed"] == {"A": 0, "B": 0}
    assert policy.min_order == 0


def _seed_with_equal_positive_covers(lag: int, sales: int) -> tuple[int, int]:
    """Return ``(policy_seed, cover)`` whose first tick draws the same cover > 0."""
    table = _table({"A": ("F_A", 4.0), "B": ("F_B", 4.0)})
    obs = _obs(
        inventory={"A": 0, "B": 0},
        observed_sales={"A": sales, "B": sales},
    )
    for seed in range(300):
        policy = ExploratoryPolicy(policy_seed=seed, delivery_lag=lag)
        policy.decide(obs, table)
        covers = policy.proposals[-1]["covers"]
        if covers["A"] == covers["B"] and covers["A"] > 0:
            return seed, covers["A"]
    raise AssertionError("no seed drew equal positive covers")


def test_planning_depth_changes_the_order_horizon():
    """A deeper horizon raises the target. The rate window stays the lag."""
    lag = 3
    sales = 4
    seed, cover = _seed_with_equal_positive_covers(lag, sales)
    obs = _obs(
        inventory={"A": 0, "B": 0},
        observed_sales={"A": sales, "B": sales},
    )
    table = _table({"A": ("F_A", 4.0), "B": ("F_B", 4.0)})
    short = ExploratoryPolicy(policy_seed=seed, delivery_lag=lag, planning_depth=4)
    deep = ExploratoryPolicy(policy_seed=seed, delivery_lag=lag, planning_depth=12)
    short_action = short.decide(obs, table)
    deep_action = deep.decide(obs, table)
    assert short.proposals[-1]["covers"] == deep.proposals[-1]["covers"] == {"A": cover, "B": cover}
    short_qty = _granted(short_action, "A") + _granted(short_action, "B")
    deep_qty = _granted(deep_action, "A") + _granted(deep_action, "B")
    assert deep_qty > short_qty
    assert short_qty == 2 * max(0, round(cover * sales * 4))
    assert deep_qty == 2 * max(0, round(cover * sales * 12))


def test_higher_priority_takes_the_whole_pool_when_only_one_cover_fits():
    lag = 2
    sales = 6
    unit_cost = 4.0
    seed, cover = _seed_with_equal_positive_covers(lag, sales)
    desired = max(0, round(cover * float(sales) * lag) - 0)
    assert desired > 0
    table = _table({"A": ("F_A", unit_cost), "B": ("F_B", unit_cost)})

    tight_space = _obs(
        inventory={"A": 0, "B": 0},
        observed_sales={"A": sales, "B": sales},
        capacity=desired,
        cash=1_000_000.0,
    )
    policy = ExploratoryPolicy(policy_seed=seed, delivery_lag=lag)
    action = policy.decide(tight_space, table)
    top, second = policy.proposals[-1]["priority"]
    assert _granted(action, top) == desired
    assert _granted(action, second) == 0
    assert _granted(action, "A") + _granted(action, "B") <= desired

    tight_cash = _obs(
        inventory={"A": 0, "B": 0},
        observed_sales={"A": sales, "B": sales},
        capacity=100_000,
        cash=float(desired * unit_cost),
    )
    policy = ExploratoryPolicy(policy_seed=seed, delivery_lag=lag)
    action = policy.decide(tight_cash, table)
    top, second = policy.proposals[-1]["priority"]
    assert _granted(action, top) == desired
    assert _granted(action, second) == 0
    spent = (_granted(action, "A") + _granted(action, "B")) * unit_cost
    assert spent <= tight_cash["cash"]


def _flow_sales(tick: dict, pid: str) -> int:
    for flow in tick["node_flows"]:
        if flow["node_id"] == "S" and flow["pid"] == pid:
            return int(flow["sales"])
    return 0


def _filled(tick: dict, pid: str) -> int:
    return sum(
        int(row["qty_filled"])
        for row in tick["purchases"]
        if row["buyer_id"] == "S" and row["pid"] == pid
    )


def test_attached_on_the_tuner_graph_orders_every_tick_from_the_cover_formula():
    lag = 3
    config = TuningConfig(
        episode_length=4,
        K_active=2,
        delivery_lag=lag,
        capacity_dist=Constant(50_000),
        balance_dist=Constant(1_000_000),
        init_stock_pct_dist=Constant(0.0),
    )
    spec = sample_episode(make_synthetic_catalog(8), config, episode_seed=14_000_000)
    policy = ExploratoryPolicy(policy_seed=11, delivery_lag=lag)
    log = Runner(spec.scenario, policy_overrides={"S": policy}).run()

    assert len(policy.proposals) == config.episode_length
    sales_hist: dict[str, list[int]] = {pid: [] for pid in spec.active_subset}
    for tick, proposal in zip(log["ticks"], policy.proposals, strict=True):
        assert set(proposal["covers"].values()) <= {0, 1, 2, 4}
        assert set(proposal["covers"]) == set(spec.active_subset)
        for pid in spec.active_subset:
            sales_hist[pid].append(_flow_sales(tick, pid))
            window = sales_hist[pid][-lag:]
            rate = sum(window) / len(window)
            on_hand = int(tick["node_inventory"]["S"][pid])
            in_transit = int(tick["node_pending"]["S"].get(pid, 0)) - _filled(tick, pid)
            position = on_hand + in_transit
            expected = max(0, round(proposal["covers"][pid] * rate * lag) - position)
            ordered = int(tick["node_orders"]["S"].get(pid, 0))
            assert ordered == expected


def test_same_policy_seed_repeats_orders_and_world_seed_does_not_change_proposals():
    config = TuningConfig(
        episode_length=5,
        K_active=3,
        delivery_lag=3,
        capacity_dist=Constant(800),
        balance_dist=Constant(50_000),
        init_stock_pct_dist=Constant(0.4),
    )
    spec = sample_episode(make_synthetic_catalog(12), config, episode_seed=14_000_042)
    world_seed = spec.scenario.world_seed

    def run(seed_world: int, policy_seed: int):
        scenario = replace(spec.scenario, world_seed=seed_world)
        policy = ExploratoryPolicy(policy_seed=policy_seed, delivery_lag=3)
        log = Runner(scenario, policy_overrides={"S": policy}).run()
        orders = [dict(tick["node_orders"]["S"]) for tick in log["ticks"]]
        return orders, policy.proposals

    orders_a, proposals_a = run(world_seed, 7)
    orders_b, proposals_b = run(world_seed, 7)
    assert orders_a == orders_b
    assert proposals_a == proposals_b

    _orders_c, proposals_c = run(world_seed + 999_983, 7)
    assert proposals_c == proposals_a
