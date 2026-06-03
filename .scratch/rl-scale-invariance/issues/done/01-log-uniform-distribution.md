# 01 — `LogUniform(lo, hi)` Distribution subclass

Status: ready-for-agent

## Parent

PRD: `.scratch/rl-scale-invariance/PRD.md`
ADR: `docs/adr/0007-rl-scale-invariance-package.md`

## What to build

A new `LogUniform(lo, hi)` `Distribution` subclass in `src/sim/distributions.py`, alongside the existing `Constant` / `Uniform` / `Normal` / `Choice` shapes. `sample(rng)` returns `exp(rng.uniform(log(lo), log(hi)))` — a value drawn uniformly in log-space, supported on `[lo, hi]`. JSON round-trip via the existing `_REGISTRY` discriminator pattern (tag `"log_uniform"`). Validation: `lo > 0` and `hi > lo`, enforced at construction so a typo in a hand-edited scenario JSON fails loudly.

This is the load-bearing primitive for slice 6's domain-randomisation defaults (`LogUniform(100, 10_000)` for capacity, `LogUniform(10_000, 1_000_000)` for opening balance), but is independently useful for anything multiplicatively distributed.

PRD user story 14: "As a simulator user authoring a scenario, I want `LogUniform(lo, hi)` available as a first-class `Distribution` subclass alongside `Constant` / `Uniform` / `Normal` / `Choice`, so that log-scale draws can be expressed without ad-hoc lambdas."

### Shape

`LogUniform(Distribution)`, frozen dataclass:
- Fields: `low: float`, `high: float`.
- `sample(rng) -> float` returns `math.exp(rng.uniform(math.log(self.low), math.log(self.high)))`.
- `to_dict() -> {"type": "log_uniform", "low": ..., "high": ...}`.
- `__post_init__` raises `ValueError` if `low <= 0` or `high <= low`.
- Register in `_REGISTRY` under `"log_uniform"`.

### Tests added

Tests live in `tests/sim/test_distributions.py` (extend if present; create if not — follow `Constant` / `Uniform` / `Normal` / `Choice` test style):

- `test_log_uniform_sample_in_range` — many samples all lie in `[lo, hi]`.
- `test_log_uniform_geometric_mean_property` — mean of N=10_000 samples from `LogUniform(1, 100)` falls between the geometric mean (`10`) and the arithmetic mean (`50.5`); covers the "is it actually log-distributed" check without depending on exact statistical tail behaviour.
- `test_log_uniform_determinism` — same `Random(seed)` produces same sample across two calls.
- `test_log_uniform_json_round_trip` — `distribution_from_dict(LogUniform(2, 8).to_dict())` reconstructs an equal `LogUniform`.
- `test_log_uniform_rejects_non_positive_low` — `LogUniform(0, 10)` raises `ValueError`; `LogUniform(-1, 10)` raises `ValueError`.
- `test_log_uniform_rejects_inverted_range` — `LogUniform(10, 1)` raises `ValueError`; `LogUniform(5, 5)` raises `ValueError`.

## Acceptance criteria

- [ ] `src/sim/distributions.py` exports `LogUniform` (importable via `from src.sim.distributions import LogUniform`).
- [ ] `LogUniform(100, 10_000)` instantiates; `.sample(Random(42))` returns a float in `[100, 10_000]`.
- [ ] `LogUniform(100, 10_000).to_dict()` returns `{"type": "log_uniform", "low": 100, "high": 10_000}`.
- [ ] `distribution_from_dict({"type": "log_uniform", "low": 100, "high": 10_000})` returns an equal `LogUniform`.
- [ ] `LogUniform(0, 10)`, `LogUniform(-1, 10)`, `LogUniform(10, 1)`, and `LogUniform(5, 5)` all raise `ValueError` at construction.
- [ ] All six new tests pass under `uv run pytest tests/sim/test_distributions.py`.
- [ ] No regression in `uv run pytest tests/sim/`.

## Blocked by

None — can start immediately.
