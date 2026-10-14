import numpy as np
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
