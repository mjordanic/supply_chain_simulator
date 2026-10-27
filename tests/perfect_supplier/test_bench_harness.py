import json

import numpy as np
import pandas as pd
import pytest
import yaml

from src.perfect_supplier.bench import log
from src.perfect_supplier.bench.arms import SeasonalEtsForecaster, cell_of, make_policy
from src.perfect_supplier.bench.calibration import calibrate
from src.perfect_supplier.bench.evaluate import evaluate, episode
from src.perfect_supplier.bench.protocol import PROTOCOL_PATH, GateError, Protocol, freeze
from src.perfect_supplier.bench import stats
from src.perfect_supplier.bench.report import write_pooled_primary, write_report
from src.perfect_supplier.bench.tuner import tune
from src.perfect_supplier.forecast import HORIZON, QUANTILES
from src.perfect_supplier.sim import Obs
from src.perfect_supplier.world import HISTORY_DAYS, sample_world

EOQ = {"name": "eoq_joint_a", "kind": "eoq_joint", "variant": "base",
       "params": {"service_z": 1.2, "demand_window": 20, "q_scale": 1.0}}


@pytest.fixture
def proto(tmp_path):
    """The repo protocol with small budgets and a stand-in sealed block (never the real test seeds)."""
    raw = yaml.safe_load(PROTOCOL_PATH.read_text())
    raw["budgets"] = {"learned": {"trials": 2, "worlds_per_trial": 1}, "textbook_multiplier": 4,
                      "textbook": {"trials": 4, "worlds_per_trial": 2}}
    raw["seed_blocks"]["test"] = {"first": 8_000_000, "size": 10, "sealed": True}
    raw["seed_blocks"]["tune"]["size"] = 64
    raw["compute"]["max_workers"] = 1
    raw["statistics"]["resamples"] = 500
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(raw))
    return Protocol.load(path)


def test_repo_protocol_holds_the_approved_constants():
    p = Protocol.load()
    assert p.raw["sesoi_pts"] == 0.10 and p.sesoi == pytest.approx(0.001)
    assert p.raw["seed_blocks"]["test"] == {"first": 9_000_000, "size": 1000, "sealed": True}
    assert p.raw["metrics"]["primary"] == "margin_capture" and p.raw["metrics"]["co_reported"] == "cost_reduction"
    assert p.budget("planner") == {"trials": 48, "worlds_per_trial": 32}
    tb = p.budget("eoq_joint")
    assert tb["trials"] * tb["worlds_per_trial"] >= 4 * 48 * 32
    assert p.raw["training_seeds"] == 3
    assert p.raw["comparisons"]["primary"]["arms"] == ["jepa", "direct"]
    assert p.workers() <= 3 or "PS_WORKERS" in __import__("os").environ
    assert p.burnt(3_000_123) and p.burnt(3_200_000) and not p.burnt(4_000_000)


def test_protocol_refuses_a_textbook_budget_below_the_multiplier(tmp_path):
    raw = yaml.safe_load(PROTOCOL_PATH.read_text())
    raw["budgets"]["textbook"] = {"trials": 48, "worlds_per_trial": 32}
    (tmp_path / "p.yaml").write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="textbook budget"):
        Protocol.load(tmp_path / "p.yaml")


def test_seeds_stay_inside_their_block_and_off_burnt_ranges(proto, tmp_path):
    assert proto.seeds("dev", 3, offset=2) == [4_000_002, 4_000_003, 4_000_004]
    with pytest.raises(GateError, match="holds 500"):
        proto.seeds("dev", 10, offset=495)
    raw = dict(proto.raw, seed_blocks={**proto.raw["seed_blocks"], "dev": {"first": 3_000_490, "size": 20}})
    (tmp_path / "burnt.yaml").write_text(yaml.safe_dump(raw))
    with pytest.raises(GateError, match="burnt seed 3000490"):
        Protocol.load(tmp_path / "burnt.yaml").seeds("dev", 5)


def test_sealed_run_without_a_frozen_config_is_refused_and_logged(proto, tmp_path):
    logs, frozen = tmp_path / "log", tmp_path / "frozen.json"
    with pytest.raises(GateError, match="not frozen"):
        evaluate([EOQ], "test", 2, "base", tmp_path / "out", proto, "t", log_dir=logs, frozen_path=frozen)
    (row,) = log.read("t", logs)
    assert row["status"] == "refused" and row["kind"] == "test-eval"
    assert not (tmp_path / "out").exists()
    first = json.loads(log.log_path("t", logs).read_text().splitlines()[0])
    assert first == {"kind": "protocol", "track": "t", "protocol_sha256": proto.sha256,
                     "protocol_path": str(proto.path)}


def test_frozen_config_runs_once_then_needs_a_rerun_reason(proto, tmp_path):
    logs, frozen = tmp_path / "log", tmp_path / "frozen.json"
    freeze(EOQ, "test", proto, frozen)
    frame = evaluate([EOQ], "test", 2, "base", tmp_path / "out", proto, "t", log_dir=logs, frozen_path=frozen)
    assert sorted(frame["world_seed"]) == [8_000_000, 8_000_001]
    with pytest.raises(GateError, match="rerun reason"):
        evaluate([EOQ], "test", 2, "base", tmp_path / "out2", proto, "t", log_dir=logs, frozen_path=frozen)
    evaluate([EOQ], "test", 2, "base", tmp_path / "out3", proto, "t", log_dir=logs, frozen_path=frozen,
             rerun_reason="checking reproducibility")
    kept = [r for r in log.read("t", logs) if r["status"] == "kept"]
    assert [r["rerun_reason"] for r in kept] == [None, "checking reproducibility"]
    # The same arm under a new name is still the same arm.
    renamed = {**EOQ, "name": "eoq_joint_b"}
    freeze(renamed, "test", proto, frozen)
    with pytest.raises(GateError, match="rerun reason"):
        evaluate([renamed], "test", 2, "base", tmp_path / "o5", proto, "t", log_dir=logs, frozen_path=frozen)
    # A changed config is a different hash, so it is not frozen.
    with pytest.raises(GateError, match="not frozen"):
        evaluate([{**EOQ, "params": {**EOQ["params"], "service_z": 2.0}}], "test", 2, "base", tmp_path / "o4",
                 proto, "t", log_dir=logs, frozen_path=frozen)


def test_config_frozen_under_another_protocol_is_refused(proto, tmp_path):
    frozen = tmp_path / "frozen.json"
    freeze(EOQ, "old", Protocol.load(), frozen)
    with pytest.raises(GateError, match="another protocol"):
        evaluate([EOQ], "test", 2, "base", tmp_path / "o", proto, "t", log_dir=tmp_path / "log", frozen_path=frozen)


def test_failed_run_is_logged(proto, tmp_path):
    bad = {**EOQ, "kind": "no_such_kind"}
    with pytest.raises(Exception):
        evaluate([bad], "dev", 1, "base", tmp_path / "o", proto, "t", log_dir=tmp_path / "log")
    (row,) = log.read("t", tmp_path / "log")
    assert row["status"] == "failed" and "no_such_kind" in row["note"]


def test_dev_eval_writes_paired_scores_and_a_report(proto, tmp_path):
    configs = [{**EOQ, "tune_score": 0.95},
               {"name": "eoq_b", "kind": "eoq", "variant": "base", "tune_score": 0.94,
                "params": {"service_z": 1.5, "demand_window": 21, "q_scale": 0.8}},
               {"name": "never_order", "kind": "never_order", "variant": "base", "params": {}}]
    frame = evaluate(configs, "dev", 4, "base", tmp_path / "out", proto, "t", log_dir=tmp_path / "log")
    assert frame.groupby("arm").size().to_dict() == {"eoq_b": 4, "eoq_joint_a": 4, "never_order": 4}
    (row,) = log.read("t", tmp_path / "log")
    assert row["kind"] == "dev-eval" and row["seeds"] == {"first": 4_000_000, "count": 4}
    assert set(row["config"]) == {"eoq_joint_a", "eoq_b", "never_order"}
    text = write_report(tmp_path / "out", proto, tmp_path / "r.md", "t", primary=["eoq_b"], strata=["holding_rate"],
                        log_dir=tmp_path / "log")
    assert "reference: `eoq_joint_a` (best tune score" in text
    assert "| eoq_b − eoq_joint_a |" in text and "cost reduction" in text
    assert "## Primary family (exploratory" in text
    report_row = log.read("t", tmp_path / "log")[-1]
    assert report_row["kind"] == "report" and report_row["families"] == "exploratory"
    assert report_row["primary"] == ["eoq_b"]


def test_param_eoq_with_zero_slope_is_global_eoq_and_cells_pick_their_params():
    w = sample_world(4_000_001, "dev")
    flat = {"name": "p", "kind": "param_eoq_joint",
            "params": {"z_a": 1.2, "z_b": 0.0, "demand_window": 20, "q_scale": 1.0}}
    assert episode(flat, w.world_seed, "dev", "base")["score"] == episode(EOQ, w.world_seed, "dev", "base")["score"]
    ctx = {"name": "c", "kind": "eoq_joint", "cell_keys": ["holding_rate"],
           "cells": {cell_of(w, ["holding_rate"]): {"service_z": 2.5, "demand_window": 10, "q_scale": 2.0}}}
    assert make_policy(ctx, w).params.service_z == 2.5


def test_every_kind_builds_a_policy_that_decides():
    w = sample_world(4_000_002, "dev")
    P = w.n_products
    obs = Obs(day=3, weekday=(HISTORY_DAYS + 3) % 7, delivery_lag=w.delivery_lag, on_hand=np.zeros(P, int),
              pipeline=np.zeros((w.delivery_lag - 1, P), int), sales=np.full((HISTORY_DAYS + 4, P), 8),
              censored=np.zeros((HISTORY_DAYS + 4, P), bool), cash=1e5, capacity=w.capacity, unit_cost=10.0,
              price=20.0, order_fee=50.0, holding_rate=w.holding_rate)
    configs = [{"name": k, "kind": k, "params": {}} for k in
               ("order_up_to", "reorder_point", "periodic_order_up_to", "periodic_reorder", "eoq", "eoq_joint",
                "never_order")]
    configs += [{"name": "pe", "kind": "param_eoq_joint", "params": {"z_a": 0, "z_b": 1, "demand_window": 14,
                                                                     "q_scale": 1}},
                {"name": "o", "kind": "planner", "params": {"forecaster": "oracle", "n_paths": 8}},
                {"name": "m", "kind": "planner", "params": {"forecaster": "moving_average", "n_paths": 8}},
                {"name": "s", "kind": "seasonal_ets", "params": {"n_paths": 8, "alpha": 0.2, "gamma": 0.1}}]
    for c in configs:
        qty = make_policy(c, w).decide(obs).qty
        assert qty.shape == (P,) and (qty >= 0).all(), c["kind"]


def test_seasonal_ets_recovers_a_noiseless_weekly_pattern():
    pattern = np.array([10, 12, 14, 9, 8, 15, 20], float)
    n = HISTORY_DAYS + 30
    sales = pattern[np.arange(n) % 7][:, None].repeat(2, axis=1)
    obs = Obs(day=29, weekday=(n - 1) % 7, delivery_lag=3, on_hand=np.zeros(2), pipeline=np.zeros((2, 2)),
              sales=sales, censored=np.zeros_like(sales, bool), cash=0.0, capacity=0, unit_cost=10.0,
              price=20.0, order_fee=50.0, holding_rate=0.001)
    q = SeasonalEtsForecaster(alpha=0.3, gamma=0.2)(obs)
    assert q.shape == (HORIZON, 2, len(QUANTILES))
    median = q[:, 0, list(QUANTILES).index(0.5)]
    np.testing.assert_allclose(median, pattern[(n + np.arange(HORIZON)) % 7], atol=1e-6)


def test_tuner_spends_the_protocol_budget_and_records_the_curve(proto, tmp_path):
    config = tune("eoq_joint", "tuned_eoq_joint", "base", proto, "t", tmp_path / "tune", log_dir=tmp_path / "log")
    assert set(config["params"]) == {"service_z", "demand_window", "q_scale"}
    assert config["budget"]["trials"] == 4 and config["budget"]["worlds_per_trial"] == 2
    trials = pd.read_parquet(tmp_path / "tune" / "tuned_eoq_joint-trials.parquet")
    assert len(trials) == 4 and trials["best_so_far"].is_monotonic_increasing
    assert trials["best_so_far"].iloc[-1] == pytest.approx(config["tune_score"])
    (row,) = log.read("t", tmp_path / "log")
    assert row["kind"] == "tune" and row["budget"]["episodes"] == 8
    assert json.loads((tmp_path / "tune" / "tuned_eoq_joint.json").read_text()) == config


def test_contextual_tuner_tunes_each_cell(proto, tmp_path):
    config = tune("eoq_joint", "ctx_eoq_joint", "base", proto, "t", tmp_path / "tune",
                  cell_keys=["holding_rate"], log_dir=tmp_path / "log")
    assert len(config["cells"]) == 3 and config["cell_keys"] == ["holding_rate"]
    w = sample_world(4_000_000, "dev")
    assert make_policy(config, w).params.service_z == config["cells"][cell_of(w, ["holding_rate"])]["service_z"]


def test_calibration_of_the_oracle_planner(proto, tmp_path):
    arm = {"name": "planner_oracle", "kind": "planner", "params": {"forecaster": "oracle", "n_paths": 64}}
    table = calibrate(arm, "dev", 1, "base", proto, "t", tmp_path / "cal", every=10, log_dir=tmp_path / "log")
    assert set(table.index) == {"lag", "lag+7"}
    # The oracle knows the true distribution: its 0.90 exceedance is near nominal.
    assert table.iloc[:, 0].between(0.0, 0.25).all()
    (row,) = log.read("t", tmp_path / "log")
    assert row["kind"] == "diagnostic" and row["status"] == "kept"
    with pytest.raises(ValueError, match="sealed"):
        calibrate(arm, "test", 1, "base", proto, "t", tmp_path / "cal", log_dir=tmp_path / "log")


def test_report_averages_training_seeds_and_adds_a_two_level_section(proto, tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    ref = rng.uniform(0.9, 0.97, n)
    arms = {"ref": ref, **{f"jepa_s{k}": ref + 0.002 + 0.0005 * k + rng.normal(0, 0.002, n) for k in range(3)},
            "planner_oracle": ref + 0.004 + rng.normal(0, 0.002, n), "seasonal_ets": ref + rng.normal(0, 0.002, n)}
    rows = [{"world_seed": 4_000_000 + i, "arm": a, "score": s[i], "variant": "base", "split": "dev",
             "holding_rate": 0.01, "cv": 0.3, "total_demand": 1000.0, "price": 20.0, "unit_cost": 10.0,
             "lost_units": 1, "fees": 50.0, "holding_cost": 30.0, "order_days": 50, "fill_rate_demand": 0.99,
             "mean_on_hand": 70.0} for a, s in arms.items() for i in range(n)]
    (tmp_path / "s").mkdir()
    pd.DataFrame(rows).to_parquet(tmp_path / "s" / "scores.parquet")
    configs = [{"name": "ref", "kind": "eoq_joint", "params": {}, "tune_score": 0.95}]
    configs += [{"name": f"jepa_s{k}", "kind": "planner", "seed_group": "jepa", "params": {"forecaster": "jepa"}}
                for k in range(3)]
    configs += [{"name": "planner_oracle", "kind": "planner", "params": {"forecaster": "oracle"}},
                {"name": "seasonal_ets", "kind": "seasonal_ets", "params": {}}]
    (tmp_path / "s" / "arms.json").write_text(json.dumps(configs))
    text = write_report(tmp_path / "s", proto, tmp_path / "r.md", "t", strata=[], log_dir=tmp_path / "log")
    primary, secondary = text.split("## Primary family")[1].split("## Secondary family")
    assert "provisional" in primary and "superior" not in primary and "inconclusive" not in primary
    assert "| jepa − ref |" in primary and "planner_oracle" not in primary and "seasonal_ets" not in primary
    assert "| jepa − seasonal_ets |" in secondary and "| jepa − planner_oracle |" in secondary
    assert "pre-registered" in text and "exploratory" not in text and "## Training seeds" in text
    (row,) = log.read("t", tmp_path / "log")
    assert row["kind"] == "report" and row["families"] == "pre-registered" and row["primary"] == ["jepa"]
    assert "jepa_s0: +0.182, jepa_s1: +0.254, jepa_s2: +0.292" in text


def _scores_dir(path, arms):
    """A synthetic eval directory: ``arms`` maps name -> (config, scores)."""
    path.mkdir(parents=True)
    rows = [{"world_seed": 9_000_000 + i, "arm": a, "score": s[i], "variant": "base", "split": "test",
             "total_demand": 1000.0, "price": 20.0, "unit_cost": 10.0, "lost_units": 1, "fees": 50.0,
             "holding_cost": 30.0, "order_days": 50, "fill_rate_demand": 0.99, "mean_on_hand": 70.0}
            for a, (_, s) in arms.items() for i in range(len(s))]
    pd.DataFrame(rows).to_parquet(path / "scores.parquet")
    (path / "arms.json").write_text(json.dumps([c for c, _ in arms.values()]))
    return [{"path": str(path / f), "sha256": ""} for f in ("scores.parquet", "arms.json")]


def _with_cells(proto, tmp_path, cells):
    """``proto`` with these declared primary cells, each a (track, variant) of the perfect-supplier family."""
    raw = yaml.safe_load(proto.path.read_text())
    raw["comparisons"]["primary"]["cells"] = [{"track": t, "world_family": "perfect_supplier", "variant": v}
                                              for t, v in cells]
    path = tmp_path / f"protocol-{len(cells)}.yaml"
    path.write_text(yaml.safe_dump(raw))
    return Protocol.load(path)


def _log_test_eval(track, variant, arms, proto, tmp_path):
    artifacts = _scores_dir(tmp_path / f"{track}-{variant}", arms)
    log.append(track, {"kind": "test-eval", "world_family": "perfect_supplier", "variant": variant,
                       "artifacts": artifacts, "status": "kept"}, proto, tmp_path / "log")


def test_pooled_primary_runs_holm_once_over_the_declared_cells(proto, tmp_path):
    rng = np.random.default_rng(1)
    n = 100
    diffs = []
    for track, variant, shift in (("T1", "base", 0.0005), ("T2", "base", 0.001), ("T2", "no_fee", 0.003)):
        ref = rng.uniform(0.9, 0.97, n)
        jepa, direct = ref + shift + rng.normal(0, 0.003, n), ref + shift / 2 + rng.normal(0, 0.003, n)
        if variant == "base":
            diffs += [jepa - ref, direct - ref]
        _log_test_eval(track, variant, {
            "ref": ({"name": "ref", "kind": "eoq_joint", "params": {}, "tune_score": 0.95}, ref),
            "jepa": ({"name": "jepa", "kind": "planner", "params": {"forecaster": "jepa"}}, jepa),
            "direct": ({"name": "direct", "kind": "planner", "params": {"forecaster": "direct"}}, direct)},
            proto, tmp_path)
    declared = _with_cells(proto, tmp_path, [("T1", "base"), ("T2", "base")])
    text = write_pooled_primary(declared, tmp_path / "pooled.md", "T2", log_dir=tmp_path / "log")
    assert "over all 4 comparisons" in text and "no_fee" not in text
    row = log.read("T2", tmp_path / "log")[-1]
    assert row["kind"] == "report" and row["cells"] == ["T1/perfect_supplier/base", "T2/perfect_supplier/base"]
    p_holm = [row["result"][f"{t}/perfect_supplier/base: {a}"]["p_holm"] for t in ("T1", "T2") for a in ("jepa", "direct")]
    assert p_holm == pytest.approx(list(stats.holm([stats.ttest_p(d) for d in diffs])))
    with_t3 = _with_cells(proto, tmp_path, [("T1", "base"), ("T2", "base"), ("T3", "base")])
    with pytest.raises(ValueError, match="T3/perfect_supplier/base: no test eval"):
        write_pooled_primary(with_t3, tmp_path / "p3.md", "T2", log_dir=tmp_path / "log")


def test_pooled_primary_refuses_a_cell_without_a_primary_arm(proto, tmp_path):
    ref = np.linspace(0.9, 0.95, 20)
    _log_test_eval("T2", "base", {
        "ref": ({"name": "ref", "kind": "eoq_joint", "params": {}, "tune_score": 0.95}, ref),
        "jepa": ({"name": "jepa", "kind": "planner", "params": {"forecaster": "jepa"}}, ref + 0.001)},
        proto, tmp_path)
    with pytest.raises(ValueError, match="T2/perfect_supplier/base: direct"):
        write_pooled_primary(_with_cells(proto, tmp_path, [("T2", "base")]), tmp_path / "p.md", "T2",
                             log_dir=tmp_path / "log")


def test_git_state_hashes_untracked_files(tmp_path, monkeypatch):
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "a.py").write_text("x = 1\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "a.py"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "a"],
                   check=True)
    monkeypatch.setattr(log, "REPO", tmp_path)
    assert log.git_state()["git_dirty_diff_sha256"] is None
    (tmp_path / "new_forecaster.py").write_text("y = 2\n")
    first = log.git_state()["git_dirty_diff_sha256"]
    assert first is not None
    (tmp_path / "new_forecaster.py").write_text("y = 3\n")
    assert log.git_state()["git_dirty_diff_sha256"] not in (None, first)


def test_report_with_an_explicit_reference_needs_no_tuned_textbook_arm(proto, tmp_path):
    rng = np.random.default_rng(2)
    ref = rng.uniform(0.9, 0.97, 50)
    _scores_dir(tmp_path / "s", {
        "direct": ({"name": "direct", "kind": "planner", "params": {"forecaster": "direct"}}, ref),
        "jepa": ({"name": "jepa", "kind": "planner", "params": {"forecaster": "jepa"}},
                 ref + 0.001 + rng.normal(0, 0.002, 50))})
    text = write_report(tmp_path / "s", proto, tmp_path / "r.md", "t", reference="direct", primary=["jepa"],
                        secondary=[], strata=[], log_dir=tmp_path / "log")
    assert "reference: `direct`\n" in text and "| jepa − direct |" in text
