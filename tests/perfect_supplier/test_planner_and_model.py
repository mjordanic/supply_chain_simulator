import numpy as np
import pytest
import pandas as pd
import torch

from src.perfect_supplier import data, jepa
from src.perfect_supplier.forecast import HORIZON, QUANTILES, OracleForecaster
from src.perfect_supplier.planner import Planner
from src.perfect_supplier.sim import Obs, simulate
from src.perfect_supplier.world import HISTORY_DAYS, sample_world


def _obs(on_hand, pipeline=None, lag=3, cash=1e6, capacity=10_000, holding_rate=0.001):
    P = len(on_hand)
    return Obs(
        day=10, weekday=3, delivery_lag=lag, on_hand=np.asarray(on_hand),
        pipeline=np.zeros((lag - 1, P), dtype=np.int64) if pipeline is None else pipeline,
        sales=np.full((HISTORY_DAYS + 11, P), 10), censored=np.zeros((HISTORY_DAYS + 11, P), bool),
        cash=cash, capacity=capacity, unit_cost=10.0, price=20.0, order_fee=50.0, holding_rate=holding_rate,
    )


class Flat:
    name = "flat"

    def __init__(self, level):
        self.level = level

    def __call__(self, obs):
        return np.full((HORIZON, obs.n_products, len(QUANTILES)), float(self.level))


def test_planner_orders_nothing_on_a_full_shelf():
    d = Planner(Flat(10)).decide(_obs([5000, 5000]))
    assert d.qty.sum() == 0 and not d.info["order_today"]


def test_planner_orders_on_an_empty_shelf():
    d = Planner(Flat(10)).decide(_obs([0, 0]))
    assert (d.qty > 0).all()


def test_imagined_equity_without_orders_is_margin_minus_holding():
    planner = Planner(Flat(0))
    obs = _obs([1000, 500])
    paths = np.full((planner.n_paths, HORIZON, 2), 3.0)
    score, first = planner.imagine(obs, paths)
    # Stock never falls to a reorder point (3 a day over the lag), so no candidate orders.
    assert (first == 0).all()
    on_hand = np.array([1000.0, 500.0])
    expected = -obs.holding_rate * 10 * on_hand.sum()  # today's holding
    for _ in range(1, HORIZON):
        sold = np.minimum(3.0, on_hand)
        on_hand -= sold
        expected += 10 * sold.sum() - obs.holding_rate * 10 * on_hand.sum()
    np.testing.assert_allclose(score, expected)


def test_oracle_planner_beats_never_ordering():
    world = sample_world(4, "eval")
    result = simulate(world, Planner(OracleForecaster(world)))
    assert result.summary["score"] > 0.9


def test_censored_pinball_ignores_over_prediction_on_censored_days():
    levels = torch.tensor([0.5])
    q = torch.tensor([[[5.0]]])
    y = torch.tensor([[2.0]])
    assert jepa.censored_pinball(q, y, torch.tensor([[True]]), levels) == 0
    assert jepa.censored_pinball(q, y, torch.tensor([[False]]), levels) == 1.5
    assert jepa.censored_pinball(torch.tensor([[[1.0]]]), y, torch.tensor([[True]]), levels) == 0.5


def test_jepa_forecaster_returns_monotone_quantiles():
    model = jepa.JepaModel(jepa.JepaConfig())
    q = jepa.JepaForecaster(model)(_obs([0, 0, 0]))
    assert q.shape == (HORIZON, 3, len(QUANTILES))
    assert (np.diff(q, axis=-1) >= 0).all() and (q >= 0).all()


def test_storage_keeps_columns_that_only_one_method_writes(tmp_path):
    a = pd.DataFrame({"world_seed": [1, 1], "day": [0, 1], "split": "eval", "method": "a", "x": [1, 2]})
    b = pd.DataFrame({"world_seed": [1, 2], "day": [0, 0], "split": "eval", "method": "b", "dec_y": [3, 4]})
    data.write_part(tmp_path, "eval", "a", 0, [a], [{"world_seed": 1, "score": 0.5}])
    data.write_part(tmp_path, "eval", "b", 0, [b], [{"world_seed": 1, "score": 0.7}])
    both = data.load_timeseries(tmp_path)
    assert {"x", "dec_y"} <= set(both.columns) and len(both) == 4
    assert len(data.load_timeseries(tmp_path, method="b", world_seed=2)) == 1
    assert set(data.load_summary(tmp_path)["method"]) == {"a", "b"}


def test_eoq_without_fee_orders_up_to_the_reorder_point():
    from src.perfect_supplier.textbook import EoqParams, EoqPolicy

    obs = _obs(on_hand=[0, 0], pipeline=np.zeros((2, 2)), cash=1e9, capacity=10**9, holding_rate=0.001)
    obs.order_fee = 0.0
    decision = EoqPolicy(EoqParams(service_z=0.0, demand_window=7)).decide(obs)
    np.testing.assert_allclose(decision.qty, np.rint(decision.info["s"]))
    assert (decision.info["S"] == decision.info["s"]).all()


def test_planner_ration_matches_the_sim():
    from src.perfect_supplier.planner import _ration
    from src.perfect_supplier.sim import ration

    rng = np.random.default_rng(0)
    want = rng.integers(0, 40, (6, 50, 4)).astype(float)
    free = rng.integers(-5, 120, (6, 50)).astype(float)
    budget = rng.uniform(-100, 1500, (6, 50))
    got = _ration(want, free, budget, 10.0)
    for c in range(6):
        for m in range(50):
            np.testing.assert_array_equal(got[c, m], ration(want[c, m].astype(np.int64), free[c, m], budget[c, m], 10.0))


@pytest.mark.parametrize("cash_scale,cap_scale", [(100, 100), (0.3, 1), (100, 0.6)])
def test_imagined_equity_matches_the_sim_on_the_realised_demand(cash_scale, cap_scale):
    """Each candidate plan, imagined on the true demand path, equals a real run of that plan."""
    import dataclasses

    from src.perfect_supplier.sim import Decision
    from src.perfect_supplier.world import demand_path

    w = sample_world(3000001, "eval")
    w = dataclasses.replace(w, initial_cash=w.initial_cash * cash_scale, capacity=int(w.capacity * cap_scale))
    demand = demand_path(w)
    future = demand[HISTORY_DAYS + 1 : HISTORY_DAYS + 1 + HORIZON].astype(float)

    class Capture:
        name = "capture"

        def decide(self, obs):
            self.obs = obs
            return Decision(np.zeros(obs.n_products, dtype=np.int64))

    capture = Capture()
    simulate(dataclasses.replace(w, n_days=1), capture, demand=demand)
    planner = Planner(None, name="x", n_paths=4)
    paths = np.repeat(future[None], 4, axis=0)
    score, first = planner.imagine(capture.obs, paths)
    reorder, targets = planner._levels(paths, w.delivery_lag)
    for c in range(0, len(planner.review), 5):

        class Rule:
            name = "rule"

            def decide(self, o):
                pos = o.position.astype(float)
                if (pos < reorder[c, o.day]).any():
                    return Decision(np.maximum(np.rint(targets[c, o.day] - pos), 0).astype(np.int64))
                return Decision(np.zeros(o.n_products, dtype=np.int64))

        day = simulate(dataclasses.replace(w, n_days=HORIZON), Rule(), demand=demand).frame.groupby("day")
        first_day = day.first().iloc[0]
        before_order = first_day["equity_close"] + first_day["fee_paid"] + first_day["holding_cost"]
        real = day.first()["equity_close"].iloc[-1] - before_order
        assert real == pytest.approx(score[c], abs=1e-6)
        np.testing.assert_array_equal(day.get_group(0)["filled"].to_numpy(), first[c])


def test_joint_eoq_orders_every_product_when_one_is_low():
    from src.perfect_supplier.textbook import EoqParams, EoqPolicy

    obs = _obs(on_hand=[0, 5000], cash=1e9, capacity=10**9)
    alone = EoqPolicy(EoqParams(demand_window=7)).decide(obs).qty
    joint = EoqPolicy(EoqParams(demand_window=7, joint=True)).decide(obs).qty
    assert alone[0] > 0 and alone[1] == 0
    assert joint[0] > 0 and joint[1] == 0  # product 1 is far above its order-up-to level
    obs2 = _obs(on_hand=[0, 40], cash=1e9, capacity=10**9)
    assert (EoqPolicy(EoqParams(demand_window=7, joint=True)).decide(obs2).qty > 0).all()
    assert EoqPolicy(EoqParams(demand_window=7)).decide(obs2).qty[1] == 0


def test_eval_offset_adds_to_earlier_worlds(tmp_path):
    from src.perfect_supplier.experiment import evaluate

    evaluate(tmp_path, 3, ["never_order"], "base", None, offset=0)
    evaluate(tmp_path, 2, ["never_order"], "base", None, offset=3)
    seeds = sorted(data.load_summary(tmp_path)["world_seed"])
    assert seeds == list(range(3_000_000, 3_000_005))


def test_loader_asks_each_method_only_for_columns_it_has(tmp_path):
    a = pd.DataFrame({"world_seed": [1], "day": [0], "split": "eval", "method": "a", "x": [1]})
    b = pd.DataFrame({"world_seed": [1], "day": [0], "split": "eval", "method": "b", "dec_y": [3]})
    data.write_part(tmp_path, "eval", "a", 0, [a], [{"world_seed": 1}])
    data.write_part(tmp_path, "eval", "b", 0, [b], [{"world_seed": 1}])
    both = data.load_timeseries(tmp_path, columns=["method", "day", "dec_y"])
    assert list(both.columns) == ["method", "day", "dec_y"] and both["dec_y"].isna().sum() == 1
