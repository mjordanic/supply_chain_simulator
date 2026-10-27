"""Track T2's harness additions: training seeds, learned contextual budgets, reference selection."""

import json

import numpy as np
import pytest
import yaml

from src.perfect_supplier.bench import log
from src.perfect_supplier.bench.protocol import PROTOCOL_PATH, GateError, Protocol


@pytest.fixture
def proto(tmp_path):
    raw = yaml.safe_load(PROTOCOL_PATH.read_text())
    raw["budgets"] = {"learned": {"trials": 6, "worlds_per_trial": 1}, "textbook_multiplier": 4,
                      "textbook": {"trials": 24, "worlds_per_trial": 1}}
    raw["seed_blocks"]["train"]["size"] = 4
    raw["seed_blocks"]["tune"]["size"] = 64
    raw["compute"]["max_workers"] = 1
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(raw))
    return Protocol.load(path)


def test_fit_trains_one_seed_logs_it_and_writes_its_arm_config(proto, tmp_path, monkeypatch):
    from src.perfect_supplier.bench.arms import make_policy
    from src.perfect_supplier.bench.fit import fit_model
    from src.perfect_supplier.experiment import collect
    from src.perfect_supplier.forecast import TAIL_QUANTILES
    from src.perfect_supplier.world import sample_world

    monkeypatch.setenv("PS_WORKERS", "1")
    collect(tmp_path / "runs", 4, "base")
    config = fit_model(tmp_path / "runs", "direct", 2, proto, "t", tmp_path / "models", steps=3,
                       log_dir=tmp_path / "log")
    assert config["seed_group"] == "direct" and config["name"] == "direct_s2"
    assert config["model"]["sha256"] == log.file_sha256(tmp_path / "models" / "direct-s2.pt")
    (row,) = log.read("t", tmp_path / "log")
    assert row["kind"] == "fit" and row["status"] == "kept" and row["config"]["seed"] == 2
    w = sample_world(4_000_000, "dev")
    policy = make_policy({**config, "variant": "base"}, w)
    np.testing.assert_array_equal(policy.forecaster.levels, TAIL_QUANTILES)


def test_fit_refuses_a_behaviour_log_that_is_not_the_train_block(proto, tmp_path, monkeypatch):
    from src.perfect_supplier.bench.fit import fit_model
    from src.perfect_supplier.experiment import collect

    monkeypatch.setenv("PS_WORKERS", "1")
    collect(tmp_path / "runs", 3, "base")
    with pytest.raises(GateError, match="train block"):
        fit_model(tmp_path / "runs", "jepa", 0, proto, "t", tmp_path / "models", steps=2, log_dir=tmp_path / "log")
