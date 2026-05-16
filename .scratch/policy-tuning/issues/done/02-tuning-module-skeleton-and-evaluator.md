# 02 — `src/tuning/` skeleton + `TuningConfig` + `evaluate_policy_normalised`

Status: ready-for-agent

## Parent

PRD: `.scratch/policy-tuning/PRD.md`
ADR: `docs/adr/0009-policy-hyperparameter-tuning-tool.md`

## What to build

Add the new `src/tuning/` package with the configuration dataclass and the deep, testable single-policy evaluator. This slice is the seam where most of the tuning module's real logic lives; the later `study.py` orchestration sits on top of this.

New optional dependency: `uv add optuna` (single direct dep; transitive deps `alembic`, `colorlog`, `packaging`, `PyYAML`, `scipy`, `sqlalchemy`, `tqdm` are accepted).

### Module layout

```
src/tuning/
├── __init__.py        # public API exports (see below; partial — only TuningConfig and evaluate_policy_normalised in this slice)
├── config.py          # TuningConfig dataclass
└── evaluator.py       # evaluate_policy_normalised() — deep module
```

`search_spaces.py` and `study.py` are added in later slices.

### `TuningConfig` (`config.py`)

Frozen dataclass with all defaults documented:

```python
@dataclass(frozen=True)
class TuningConfig:
    n_trials: int = 150
    n_search_seeds: int = 16
    n_holdout_seeds: int = 32
    episode_length: int = 365
    seed_offset: int = 12_000_000      # disjoint from training (0) and from two-scale eval (10_000_000 + 1_000_000)
    holdout_seed_offset: int = 13_000_000
    sampler_seed: int = 42
    top_k_for_holdout: int = 5
```

### `evaluate_policy_normalised` (`evaluator.py`)

Pure-ish single-policy CRN evaluator. Takes pre-built `EpisodeSpec` objects, returns a flat dict. No Optuna import, no file I/O.

```python
def evaluate_policy_normalised(
    policy_factory: Callable[[], Policy],
    eval_specs: list[EpisodeSpec],
    *,
    config: RLConfig,
) -> dict[str, Any]:
    """Run a single policy on eval_specs, return normalised mean + per-seed KPI lists.

    Returns a dict with keys:
        mean_normalised_return: float          # mean of (net_profit / initial_cash) across seeds
        per_seed_net_profit: list[float]
        per_seed_initial_cash: list[float]
        per_seed_capacity: list[float]
        per_seed_service_level: list[float]
        per_seed_stockout_rate: list[float>
        per_seed_inventory_turnover: list[float]
        per_seed_revenue: list[float]
        per_seed_mean_price_pct_of_msrp: list[float]
        mean_*: float for each per_seed_* key
    """
```

Internally reuses the world-construction machinery from `src/rl/eval.py`. The `_build_world()` helper is suitable for module-internal reuse; lift it to a shared private helper (e.g. `src/rl/eval.py::_build_world` becoming importable from a private name, or moved to a tiny shared internal module under `src/rl/`) only if needed to avoid duplication. The public `evaluate()` in `src/rl/eval.py` is unchanged — it continues to mean "paired CRN".

The single-policy rollout uses the same `policy.act` / `Store.tick` loop as `evaluate()`'s baseline arm. `Policy` instances built by the factory consume *only* `policy_rng`; `world_rng` and `init_rng` are CRN-disjoint per the existing eval contract.

### Public API in `src/tuning/__init__.py` (partial)

```python
from src.tuning.config import TuningConfig
from src.tuning.evaluator import evaluate_policy_normalised

__all__ = [
    "TuningConfig",
    "evaluate_policy_normalised",
]
```

`search_spaces` exports are added in issue 03; `study` exports in issue 04 / 05.

### Tests added

`tests/tuning/` is created in this slice (mirrors source layout):

- `tests/tuning/test_config.py::test_seed_offsets_are_disjoint` — Assert `TuningConfig.seed_offset` and `TuningConfig.holdout_seed_offset` are disjoint from `RLConfig.eval_seed_offset` and from the two-scale eval ranges. Invariant test that prevents accidental tuning on training seeds.
- `tests/tuning/test_config.py::test_tuning_config_defaults` — Sanity check the documented defaults exist with the documented types.
- `tests/tuning/test_evaluator.py::test_evaluate_policy_normalised_2_seeds` — Build 2 `EpisodeSpec` objects with known seeds; call `evaluate_policy_normalised(lambda: OrderUpToPolicy(), specs, config=RLConfig())`; assert returned dict has all documented keys, all `per_seed_*` lists have length 2, `mean_normalised_return == mean(per_seed_normalised_return)` to float precision.
- `tests/tuning/test_evaluator.py::test_evaluate_policy_normalised_crn_determinism` — Same call twice on identical inputs returns bit-identical numbers.
- `tests/tuning/test_evaluator.py::test_normalisation_is_dimensionless` — Build two specs with capacity ratio 10× and balance ratio 10×; assert that on a no-op policy (returns zero orders) the per-seed `normalised_return` is approximately equal between the two specs (within 1% relative). Confirms the normalisation actually cancels scale.

## Acceptance criteria

- [ ] `optuna` is in `pyproject.toml` dependencies after `uv add optuna`.
- [ ] `from src.tuning import TuningConfig, evaluate_policy_normalised` works.
- [ ] `TuningConfig()` constructs with the documented defaults; all fields are immutable (`frozen=True`).
- [ ] `evaluate_policy_normalised(lambda: OrderUpToPolicy(), specs, config=RLConfig())` returns a dict with all keys listed in the docstring.
- [ ] `evaluate_policy_normalised` does not import `optuna` (the deep module is Optuna-free).
- [ ] All five tests in `tests/tuning/test_config.py` and `tests/tuning/test_evaluator.py` pass.
- [ ] `uv run pytest tests/tuning/ tests/rl/` is green (the eval contract is unchanged).

## Blocked by

- `01-reparameterise-safety-horizons.md` — the evaluator builds `OrderUpToPolicy` in tests with default kwargs, which only line up with ADR 0009's `safety_lead_pct_of_lag` framing after Bundle A lands.
