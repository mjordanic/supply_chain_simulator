"""Track T2: the planner's coherent path sampling and its tuned knobs."""

import dataclasses

import numpy as np
import pytest
from scipy.stats import norm

from src.perfect_supplier.forecast import HORIZON, QUANTILES, TAIL_QUANTILES, MovingAverageForecaster
from src.perfect_supplier.planner import SERVICE, Planner, sample_paths
from src.perfect_supplier.sim import simulate
from src.perfect_supplier.world import sample_world


@pytest.mark.parametrize("seed,variant,cash_scale,cap_scale,score,filled", [
    (4000003, "base", 1, 1, 0.881780238500852, 1345),
    (4000003, "base", 0.3, 0.6, 0.860868824531516, 1193),  # cash and shelf bind, so the ration runs
    (4000004, "no_fee", 1, 1, 0.9489150943396228, 1941),
])
def test_default_planner_episode_is_unchanged(seed, variant, cash_scale, cap_scale, score, filled):
    """Recorded from the planner before T2; the default knobs must keep every decision."""
    w = sample_world(seed, "dev", n_days=40, variant=variant)
    w = dataclasses.replace(w, initial_cash=w.initial_cash * cash_scale, capacity=int(w.capacity * cap_scale))
    r = simulate(w, Planner(MovingAverageForecaster()))
    assert r.summary["score"] == pytest.approx(score, abs=1e-12)
    assert int(r.frame["filled"].sum()) == filled


def _normal_quantiles(levels, mean=20.0, sd=4.0, H=HORIZON, P=2):
    q = mean + sd * norm.ppf(levels)
    return np.broadcast_to(q, (H, P, len(levels))).copy()


def test_coherent_paths_keep_each_days_marginal_and_widen_the_lead_time_sum():
    q = _normal_quantiles(QUANTILES)
    indep = sample_paths(q, 20_000, np.random.default_rng(0))
    coherent = sample_paths(q, 20_000, np.random.default_rng(0), rho=0.75)
    for paths in (indep, coherent):
        np.testing.assert_allclose(np.quantile(paths[:, 3, 0], [0.1, 0.5, 0.9]), q[3, 0, [1, 5, 9]], atol=0.6)
    ratio = coherent[:, :7, 0].sum(axis=1).std() / indep[:, :7, 0].sum(axis=1).std()
    # Equicorrelated normal scores: sqrt(1 + 6 * 0.75) = 2.35 for a normal marginal.
    assert 2.0 < ratio < 2.5


def test_tail_levels_let_imagined_demand_pass_the_old_cap():
    capped = sample_paths(_normal_quantiles(QUANTILES), 5_000, np.random.default_rng(1))
    tail = sample_paths(_normal_quantiles(TAIL_QUANTILES), 5_000, np.random.default_rng(1), levels=TAIL_QUANTILES)
    top = 20.0 + 4.0 * norm.ppf(0.98)
    assert capped.max() <= np.rint(top)
    assert tail.max() > top + 2


def test_service_cap_extends_the_candidate_grid():
    assert set(Planner(None, name="p").z) == set(SERVICE)
    assert set(Planner(None, name="p", service_max=0.999).z) == set(SERVICE) | {0.998, 0.999}
    assert max(Planner(None, name="p", service_max=0.9995).z) == 0.9995


def test_level_paths_resolve_a_reorder_point_above_the_64th_path():
    """With 64 paths every level above ~0.985 sits between the two largest sums; 4,096 paths resolve 0.999."""
    q = _normal_quantiles(TAIL_QUANTILES, P=1)
    true = 3 * 20.0 + np.sqrt(3) * 4.0 * norm.ppf(0.999)  # lag 3, independent days
    planner = Planner(None, name="p", service_max=0.999)
    k = np.flatnonzero(planner.z == 0.999)[0]
    few = sample_paths(q, 64, np.random.default_rng(2), levels=TAIL_QUANTILES)
    many = sample_paths(q, 4_096, np.random.default_rng(2), levels=TAIL_QUANTILES)
    reorder_many, _ = planner._levels(many, 3)
    assert reorder_many[k, 0, 0] == pytest.approx(true, abs=1.0)
    reorder_few, _ = planner._levels(few, 3)
    assert reorder_few[k, 0, 0] <= few[:, :3, 0].sum(axis=1).max()


def test_planner_uses_the_forecasters_levels_and_its_level_paths():
    w = sample_world(4000003, "dev", n_days=20)
    base = simulate(w, Planner(MovingAverageForecaster())).summary["score"]
    tail = Planner(MovingAverageForecaster(levels=TAIL_QUANTILES), rho=0.5, service_max=0.9995, level_paths=1024)
    r = simulate(w, tail)
    assert r.summary["score"] != base
    assert r.frame["filled"].sum() > 0


def test_a_jepa_model_trained_on_tail_levels_forecasts_them_and_round_trips(tmp_path):
    from src.perfect_supplier import jepa
    from src.perfect_supplier.sim import Obs
    from src.perfect_supplier.world import HISTORY_DAYS

    model = jepa.JepaModel(jepa.JepaConfig(quantiles=tuple(TAIL_QUANTILES), seed=1))
    jepa.save(model, tmp_path / "m.pt")
    f = jepa.JepaForecaster(jepa.load(tmp_path / "m.pt"))
    obs = Obs(day=10, weekday=3, delivery_lag=3, on_hand=np.zeros(2), pipeline=np.zeros((2, 2)),
              sales=np.full((HISTORY_DAYS + 11, 2), 10), censored=np.zeros((HISTORY_DAYS + 11, 2), bool), cash=0.0,
              capacity=0, unit_cost=10.0, price=20.0, order_fee=50.0, holding_rate=0.001)
    q = f(obs)
    assert q.shape == (HORIZON, 2, len(TAIL_QUANTILES))
    np.testing.assert_array_equal(f.levels, TAIL_QUANTILES)
    assert (np.diff(q, axis=-1) >= 0).all()
    # A model saved before levels were configurable keeps the original ones.
    assert jepa.JepaForecaster(jepa.JepaModel(jepa.JepaConfig())).levels.tolist() == QUANTILES.tolist()


def test_arm_configs_carry_the_t2_knobs_to_the_planner():
    from src.perfect_supplier.bench.arms import make_policy

    w = sample_world(4000003, "dev")
    knobs = {"rho": 0.4, "service_max": 0.9995, "level_paths": 512, "tail_quantiles": True}
    ets = make_policy({"name": "s", "kind": "seasonal_ets", "params": {"alpha": 0.2, "gamma": 0.1, **knobs}}, w)
    oracle = make_policy({"name": "o", "kind": "planner", "params": {"forecaster": "oracle", **knobs}}, w)
    for p in (ets, oracle):
        assert (p.rho, p.level_paths, max(p.z)) == (0.4, 512, 0.9995)
        np.testing.assert_array_equal(p.forecaster.levels, TAIL_QUANTILES)
    assert ets.forecaster.alpha == 0.2
    plain = make_policy({"name": "m", "kind": "planner", "params": {"forecaster": "moving_average"}}, w)
    np.testing.assert_array_equal(plain.forecaster.levels, QUANTILES)
