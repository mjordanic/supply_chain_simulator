"""Tests for the Distribution ABC and concrete subclasses (T2)."""

import random

import pytest

from src.sim.distributions import (
    Choice,
    Constant,
    Distribution,
    Normal,
    Uniform,
    distribution_from_dict,
)


# ---------- Sampling-bounds tests ----------


def test_uniform_samples_within_bounds():
    rng = random.Random(123)
    dist = Uniform(low=2.0, high=5.0)
    for _ in range(500):
        x = dist.sample(rng)
        assert 2.0 <= x <= 5.0


def test_constant_returns_value():
    rng = random.Random(0)
    dist = Constant(value=42)
    for _ in range(50):
        assert dist.sample(rng) == 42


def test_choice_only_returns_declared_options():
    rng = random.Random(1)
    options = ["a", "b", "c"]
    dist = Choice(options=options)
    seen = set()
    for _ in range(200):
        seen.add(dist.sample(rng))
    assert seen <= set(options)
    assert seen == set(options)  # all options should appear with enough draws


def test_choice_with_weights_respects_weights():
    rng = random.Random(7)
    dist = Choice(options=["x", "y"], weights=[0.0, 1.0])
    for _ in range(100):
        assert dist.sample(rng) == "y"


def test_normal_clip_bounds():
    rng = random.Random(99)
    dist = Normal(mean=0.0, std=10.0, clip=(-1.0, 1.0))
    for _ in range(500):
        x = dist.sample(rng)
        assert -1.0 <= x <= 1.0


def test_normal_unclipped_runs():
    rng = random.Random(99)
    dist = Normal(mean=0.0, std=1.0)
    samples = [dist.sample(rng) for _ in range(50)]
    # Sanity: not all the same value
    assert len(set(samples)) > 1


# ---------- Determinism tests ----------


@pytest.mark.parametrize(
    "dist",
    [
        Uniform(low=0.0, high=1.0),
        Normal(mean=0.0, std=1.0),
        Normal(mean=0.0, std=2.0, clip=(-1.0, 1.0)),
        Choice(options=["a", "b", "c", "d"]),
        Choice(options=[1, 2, 3], weights=[0.2, 0.3, 0.5]),
        Constant(value=7.5),
    ],
)
def test_seeded_rng_reproduces_sequence(dist):
    rng_a = random.Random(2026)
    rng_b = random.Random(2026)
    seq_a = [dist.sample(rng_a) for _ in range(20)]
    seq_b = [dist.sample(rng_b) for _ in range(20)]
    assert seq_a == seq_b


def test_separate_rngs_do_not_interact():
    """Consuming RNG A must never alter RNG B's state."""
    rng_world = random.Random(2026)
    rng_policy = random.Random(2026)

    # Take baseline sequence from rng_policy alone
    baseline = [rng_policy.random() for _ in range(10)]

    # Reset and interleave with consumption from rng_world via a Distribution
    rng_world = random.Random(99)
    rng_policy = random.Random(2026)
    dist = Uniform(low=0.0, high=1.0)
    interleaved = []
    for _ in range(10):
        dist.sample(rng_world)  # Consumes world RNG only
        interleaved.append(rng_policy.random())

    assert baseline == interleaved


# ---------- JSON round-trip tests ----------


@pytest.mark.parametrize(
    "dist",
    [
        Uniform(low=-1.5, high=2.5),
        Normal(mean=0.0, std=1.0),
        Normal(mean=10.0, std=2.0, clip=(8.0, 12.0)),
        Choice(options=["red", "green", "blue"]),
        Choice(options=[1, 2, 3], weights=[0.5, 0.25, 0.25]),
        Constant(value=3.14),
    ],
)
def test_to_dict_from_dict_round_trip(dist):
    d = dist.to_dict()
    assert "type" in d
    rebuilt = distribution_from_dict(d)
    assert rebuilt == dist


def test_uniform_json_shape():
    d = Uniform(low=1.0, high=2.0).to_dict()
    assert d == {"type": "uniform", "low": 1.0, "high": 2.0}


def test_normal_json_shape_no_clip():
    d = Normal(mean=0.0, std=1.0).to_dict()
    assert d == {"type": "normal", "mean": 0.0, "std": 1.0}


def test_normal_json_shape_with_clip():
    d = Normal(mean=0.0, std=1.0, clip=(-2.0, 2.0)).to_dict()
    assert d == {"type": "normal", "mean": 0.0, "std": 1.0, "clip": [-2.0, 2.0]}


def test_choice_json_shape_no_weights():
    d = Choice(options=["a", "b"]).to_dict()
    assert d == {"type": "choice", "options": ["a", "b"]}


def test_choice_json_shape_with_weights():
    d = Choice(options=["a", "b"], weights=[0.3, 0.7]).to_dict()
    assert d == {"type": "choice", "options": ["a", "b"], "weights": [0.3, 0.7]}


def test_constant_json_shape():
    d = Constant(value=42).to_dict()
    assert d == {"type": "constant", "value": 42}


def test_from_dict_rejects_unknown_type():
    with pytest.raises(ValueError, match="unknown distribution type"):
        distribution_from_dict({"type": "exponential", "rate": 1.0})


def test_from_dict_rejects_missing_type():
    with pytest.raises(ValueError, match="missing 'type'"):
        distribution_from_dict({"low": 0.0, "high": 1.0})


def test_distribution_is_abstract():
    with pytest.raises(TypeError):
        Distribution()  # type: ignore[abstract]
