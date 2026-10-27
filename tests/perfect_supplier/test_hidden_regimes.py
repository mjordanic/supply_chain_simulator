"""Track T1: the hidden_regimes world family and the arms built for it."""

import numpy as np
import pytest

from src.perfect_supplier.forecast import QUANTILES
from src.perfect_supplier.world import HISTORY_DAYS, demand_mean_path, demand_path, regime_path, sample_world


def test_regime_world_keeps_the_base_draws_and_adds_one_shop_wide_regime():
    base = sample_world(4_000_003, "dev")
    reg = sample_world(4_000_003, "dev", family="hidden_regimes")
    for field in ("delivery_lag", "holding_rate", "cv", "weekly_amp", "base_mean", "order_fee"):
        assert getattr(reg, field) == getattr(base, field)
    assert reg.family == "hidden_regimes" and base.family == "perfect_supplier"
    r = regime_path(reg)
    assert r.shape == (HISTORY_DAYS + reg.n_days,) and set(np.unique(r)) <= {0, 1, 2}
    # One multiplier per day, shared by every product: the ratio to the base path.
    ratio = demand_mean_path(reg) / demand_mean_path(base)
    assert np.allclose(ratio, ratio[:, :1])
    assert np.allclose(ratio[:, 0], np.asarray(reg.regime_mult)[r])


def test_regimes_are_persistent_and_switch_at_the_world_rate():
    switches, days = 0, 0
    for i in range(40):
        w = sample_world(2_000_000 + i, "tune", family="hidden_regimes")
        assert 1 / 60 <= w.switch_prob <= 1 / 20
        r = regime_path(w)
        switches += int((np.diff(r) != 0).sum())
        days += len(r) - 1
    assert 1 / 60 < switches / days < 1 / 20


def test_regime_strength_scales_the_log_multipliers():
    w = sample_world(5, family="hidden_regimes", regimes={"strength": 0.5})
    assert np.allclose(w.regime_mult, np.sqrt([0.6, 1.0, 1.6]))
    with pytest.raises(ValueError, match="family"):
        sample_world(5, family="nope")


def test_regime_demand_is_noisier_in_the_noisier_regime():
    w = sample_world(11, family="hidden_regimes", regimes={"cv_scale": [2.0, 1.0, 0.25]}, n_days=2000)
    r = regime_path(w)
    rel = demand_path(w) / np.maximum(demand_mean_path(w), 1e-9)
    sd = [rel[r == k].std() for k in range(3) if (r == k).sum() > 100]
    assert sd == sorted(sd, reverse=True)


def _uncensored_obs(world, day):
    from types import SimpleNamespace

    d = demand_path(world)[: HISTORY_DAYS + day + 1]
    return SimpleNamespace(day=day, sales=d, censored=np.zeros_like(d, dtype=bool), weekday=(HISTORY_DAYS + day) % 7)


def test_bayes_filter_tracks_the_hidden_regime():
    from src.perfect_supplier.regimes import BayesRegimeForecaster

    w = sample_world(4_000_010, "dev", family="hidden_regimes")
    fc = BayesRegimeForecaster(w)
    r = regime_path(w)
    days = range(0, 360, 3)
    hits = [int(np.argmax(fc.belief(_uncensored_obs(w, t)))) == r[HISTORY_DAYS + t] for t in days]
    assert np.mean(hits) > 0.85


def test_bayes_paths_are_regime_coherent():
    """Under an unsure belief, lead-time sums spread far wider than independent days would."""
    from types import SimpleNamespace

    from src.perfect_supplier.regimes import BayesRegimeForecaster

    w = sample_world(4_000_010, "dev", family="hidden_regimes")
    fc = BayesRegimeForecaster(w)
    # Empty, sold-out shelves say nothing about demand: the belief is the prior.
    blind = SimpleNamespace(day=100, sales=np.zeros((HISTORY_DAYS + 101, w.n_products), dtype=np.int64),
                            censored=np.ones((HISTORY_DAYS + 101, w.n_products), dtype=bool))
    assert np.allclose(fc.belief(blind), 1 / 3, atol=0.01)
    paths = fc.paths(blind, 2000, np.random.default_rng(0))  # (M, H, P)
    assert paths.shape == (2000, 35, w.n_products)
    shuffled = np.stack([paths[np.random.default_rng(h).permutation(2000), h] for h in range(35)], axis=1)
    assert paths[:, 1:15].sum(axis=(1, 2)).std() > 1.5 * shuffled[:, 1:15].sum(axis=(1, 2)).std()


def test_planner_orders_from_paths_the_forecaster_draws():
    from src.perfect_supplier.planner import Planner
    from src.perfect_supplier.sim import simulate

    class Flat:
        name = "flat"

        def paths(self, obs, n, rng):
            return np.full((n, 35, obs.n_products), 8.0)

        def __call__(self, obs):
            raise AssertionError("the planner should sample from paths()")

    w = sample_world(3, family="hidden_regimes")
    res = simulate(w, Planner(Flat(), n_paths=16))
    assert res.summary["order_days"] > 0


def test_ets_rate_joint_eoq_follows_a_demand_jump_faster_with_a_larger_alpha():
    from types import SimpleNamespace

    from src.perfect_supplier.bench.arms import make_policy

    w = sample_world(4_000_002, "dev", family="hidden_regimes")
    sales = np.concatenate([np.full((HISTORY_DAYS, 2), 10), np.full((10, 2), 20)])
    obs = SimpleNamespace(day=9, delivery_lag=3, sales=sales, censored=np.zeros_like(sales, bool),
                          position=np.zeros(2), n_products=2, holding_rate=0.001, unit_cost=10.0, order_fee=50.0)
    rate = {}
    for alpha in (0.05, 0.5):
        config = {"name": "e", "kind": "ets_eoq_joint",
                  "params": {"service_z": 1.0, "demand_window": 14, "q_scale": 1.0, "alpha": alpha}}
        rate[alpha] = make_policy(config, w).decide(obs).info["rate"][0]
    assert 10 < rate[0.05] < 15 < rate[0.5] <= 20


@pytest.fixture
def proto(tmp_path):
    import yaml

    from src.perfect_supplier.bench.protocol import PROTOCOL_PATH, Protocol

    raw = yaml.safe_load(PROTOCOL_PATH.read_text())
    raw["budgets"] = {"learned": {"trials": 2, "worlds_per_trial": 1}, "textbook_multiplier": 4,
                      "textbook": {"trials": 4, "worlds_per_trial": 2}}
    raw["compute"]["max_workers"] = 1
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(raw))
    return Protocol.load(path)


def test_repo_protocol_declares_the_hidden_regime_world_and_t1_arms():
    from src.perfect_supplier.bench.protocol import Protocol

    p = Protocol.load()
    params = p.world_params("hidden_regimes")
    assert len(params["mult"]) == 3 and 1 / 60 <= params["switch_prob"][0] <= params["switch_prob"][1] <= 1 / 20
    assert p.world_params("perfect_supplier") == {}
    assert "ets_eoq_joint" in p.raw["search_spaces"]
    assert ["jepa", "hmm"] in p.raw["comparisons"]["secondary"]["pairs"]
    assert ["jepa", "bayes"] in p.raw["comparisons"]["secondary"]["pairs"]


def test_harness_runs_and_logs_arms_on_the_hidden_regime_family(proto, tmp_path):
    from src.perfect_supplier.bench import log
    from src.perfect_supplier.bench.evaluate import evaluate
    from src.perfect_supplier.bench.tuner import tune

    ets = tune("ets_eoq_joint", "tuned_ets_eoq_joint", "base", proto, "t", tmp_path / "tune",
               family="hidden_regimes", log_dir=tmp_path / "log")
    assert ets["world_family"] == "hidden_regimes"
    bayes = {"name": "bayes", "kind": "planner", "variant": "base", "world_family": "hidden_regimes",
             "params": {"forecaster": "bayes", "n_paths": 16}}
    frame = evaluate([ets, bayes], "dev", 2, "base", tmp_path / "out", proto, "t", family="hidden_regimes",
                     log_dir=tmp_path / "log")
    assert sorted(frame["arm"].unique()) == ["bayes", "tuned_ets_eoq_joint"]
    tune_row, eval_row = log.read("t", tmp_path / "log")
    assert tune_row["world_family"] == eval_row["world_family"] == "hidden_regimes"
    with pytest.raises(ValueError, match="world family"):
        evaluate([ets], "dev", 1, "base", tmp_path / "o2", proto, "t", log_dir=tmp_path / "log")


def test_baum_welch_recovers_a_known_gaussian_hmm():
    from src.perfect_supplier.regimes import fit_gaussian_hmm

    rng = np.random.default_rng(0)
    means, sds, p = np.array([-0.5, 0.0, 0.5]), np.array([0.1, 0.1, 0.1]), 0.03
    seqs = []
    for _ in range(60):
        k, xs = int(rng.integers(3)), []
        for t in range(400):
            if rng.random() < p:
                k = (k + 1 + int(rng.integers(2))) % 3
            xs.append(rng.normal(means[k], sds[k]))
        x = np.array(xs)
        x[rng.random(400) < 0.2] = np.nan  # missing (sold-out) days carry no emission
        seqs.append(x)
    hmm = fit_gaussian_hmm(seqs, 3, seed=1)
    order = np.argsort(hmm.means)
    np.testing.assert_allclose(hmm.means[order], means, atol=0.03)
    np.testing.assert_allclose(hmm.sds[order], sds, atol=0.02)
    assert 0.02 < 1 - np.diag(hmm.trans).mean() < 0.04
    # Filtering a fresh sequence finds its state most of the time.
    k, xs, ks = 0, [], []
    for t in range(300):
        if rng.random() < p:
            k = (k + 1 + int(rng.integers(2))) % 3
        xs.append(rng.normal(means[k], sds[k]))
        ks.append(k)
    post = hmm.filter(np.array(xs))
    assert np.mean(np.argsort(order)[post.argmax(axis=1)] == np.array(ks)) > 0.9


def test_hmm_forecaster_fitted_on_train_sales_tracks_true_demand():
    from src.perfect_supplier.regimes import HmmForecaster, fit_hmm_on_sales

    train = [demand_path(sample_world(1_000_000 + i, "train", family="hidden_regimes")) for i in range(30)]
    hmm = fit_hmm_on_sales([(d, np.zeros_like(d, dtype=bool)) for d in train], n_states=3)
    w = sample_world(4_000_010, "dev", family="hidden_regimes")
    fc = HmmForecaster(hmm)
    days = range(0, 320, 4)
    pred = [fc.paths(_uncensored_obs(w, t), 256, np.random.default_rng(t))[:, :7].mean(axis=0).sum() for t in days]
    true = [demand_mean_path(w)[HISTORY_DAYS + t + 1 : HISTORY_DAYS + t + 8].sum() for t in days]
    rolling = [demand_path(w)[HISTORY_DAYS + t - 27 : HISTORY_DAYS + t + 1].sum(axis=1).mean() * 7 for t in days]
    assert np.corrcoef(pred, true)[0, 1] > np.corrcoef(rolling, true)[0, 1] + 0.1
    assert 0.9 < np.mean(pred) / np.mean(true) < 1.1


def test_copula_sampler_keeps_the_marginals_and_correlates_the_days():
    from types import SimpleNamespace

    from src.perfect_supplier.forecast import MovingAverageForecaster
    from src.perfect_supplier.regimes import CopulaSampler

    obs = SimpleNamespace(sales=np.random.default_rng(0).poisson(10, (HISTORY_DAYS + 5, 2)))
    lead = {}
    for rho in (0.0, 0.8):
        p = CopulaSampler(MovingAverageForecaster(), rho).paths(obs, 4000, np.random.default_rng(1))
        lead[rho] = p[:, :10].sum(axis=1)  # (M, P)
        assert abs(p.mean() - obs.sales[-28:].mean()) < 0.3
    assert lead[0.8].std() > 2 * lead[0.0].std()


def _demand_bank(seeds, split):
    from src.perfect_supplier.joint import ShopBank

    d = np.stack([demand_path(sample_world(s, split, family="hidden_regimes")) for s in seeds]).astype(np.float32)
    return ShopBank(d, np.zeros(d.shape, bool), d, np.asarray(seeds))


@pytest.mark.parametrize("use_jepa", [True, False])
def test_joint_model_learns_the_shop_and_draws_coherent_paths(use_jepa):
    from src.perfect_supplier import joint

    train, val = _demand_bank(range(1_000_000, 1_000_016), "train"), _demand_bank(range(1_000_100, 1_000_104), "train")
    config = joint.JointConfig(jepa=use_jepa, steps=300, eval_every=150, batch_size=64, hidden=64, embed_dim=16)
    untrained = joint.evaluate(joint.JointModel(config), val)["nll"]
    model = joint.fit(train, val, config)
    assert joint.evaluate(model, val)["nll"] < untrained
    fc = joint.JointForecaster(model)
    obs = _uncensored_obs(sample_world(4_000_010, "dev", family="hidden_regimes"), 40)
    paths = fc.paths(obs, 300, np.random.default_rng(0))
    assert paths.shape == (300, 35, 4) and (paths >= 0).all() and np.allclose(paths, np.rint(paths))
    assert fc(obs).shape == (35, 4, len(QUANTILES))


def test_planner_arms_build_from_saved_joint_and_hmm_models(tmp_path):
    from src.perfect_supplier import joint
    from src.perfect_supplier.bench.arms import make_policy, model_spec
    from src.perfect_supplier.regimes import fit_hmm_on_sales, save_hmm

    w = sample_world(4_000_002, "dev", family="hidden_regimes")
    obs = _uncensored_obs(w, 5)
    from src.perfect_supplier.sim import Obs

    obs = Obs(day=5, weekday=obs.weekday, delivery_lag=w.delivery_lag, on_hand=np.zeros(4, int),
              pipeline=np.zeros((w.delivery_lag - 1, 4), int), sales=obs.sales, censored=obs.censored, cash=1e5,
              capacity=w.capacity, unit_cost=10.0, price=20.0, order_fee=50.0, holding_rate=w.holding_rate)
    joint.save(joint.JointModel(joint.JointConfig(hidden=16, embed_dim=8)), tmp_path / "j.pt")
    save_hmm(fit_hmm_on_sales([(demand_path(w), np.zeros((HISTORY_DAYS + w.n_days, 4), bool))], 3), tmp_path / "h.json")
    configs = [
        {"name": "j", "kind": "planner", "params": {"forecaster": "jepa", "n_paths": 8},
         "model": model_spec(tmp_path / "j.pt", "joint")},
        {"name": "jc", "kind": "planner", "params": {"forecaster": "direct", "n_paths": 8, "copula_rho": 0.5},
         "model": model_spec(tmp_path / "j.pt", "joint")},
        {"name": "h", "kind": "planner", "params": {"forecaster": "hmm", "n_paths": 8},
         "model": model_spec(tmp_path / "h.json", "hmm")},
        {"name": "h", "kind": "planner", "params": {"forecaster": "hmm", "n_paths": 8, "copula_rho": 0.3},
         "model": model_spec(tmp_path / "h.json", "hmm")},
    ]
    for c in configs:
        qty = make_policy(c, w).decide(obs).qty
        assert qty.shape == (4,) and (qty >= 0).all(), c["name"]


def test_t1_workflow_collects_fits_and_scores_the_g2_gate(proto, tmp_path, monkeypatch):
    from src.perfect_supplier import t1
    from src.perfect_supplier.bench import log

    monkeypatch.setenv("PS_WORKERS", "1")
    logs = tmp_path / "log"
    (tmp_path / "models").mkdir()
    t1.collect(tmp_path, proto, sizes={"train": 6, "tune": 2, "dev": 2}, log_dir=logs)
    t1.fit(tmp_path, [0], 60, proto, log_dir=logs)
    t1.fit_hmm(tmp_path, [2, 3], proto, log_dir=logs)
    text = t1.g2(tmp_path, [0], proto, 2, log_dir=logs)
    assert "G2 passes" in text or "G2 fails" in text
    rows = log.read(t1.TRACK, logs)
    arms = [r["arm"] for r in rows]
    assert arms[:3] == ["behaviour_log"] * 3 and sorted(arms[3:5]) == ["direct-s0", "jepa-s0"] and arms[5:] == ["hmm", "G2"]
    assert all(r["status"] == "kept" and r["world_family"] == "hidden_regimes" for r in rows)
    assert rows[-1]["result"]["gate"] == "G2" and "lead_time" in rows[-1]["result"]["jepa_minus_direct"]
