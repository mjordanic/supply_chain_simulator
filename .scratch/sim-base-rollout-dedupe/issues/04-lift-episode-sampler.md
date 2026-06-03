# 04 — Lift episode sampling to `src/sim/episode_sampler.py`; introduce `RLEpisodeSpec`; retire `TuningEpisodeSpec`

Status: ready-for-agent

## Parent

PRD: `.scratch/sim-base-rollout-dedupe/PRD.md`
ADR: `docs/adr/0004-rl-training-env.md` (slot-permutation invariant stays RL-specific)

## What to build

Make `src/sim/episode_sampler.py` the canonical home for the shared `(scenario, active_subset)` sampler, the 4-stream seed split (assortment / capacity / balance / world), and the three default-params factories. RL retains its 5th `slot` stream and the `slot_permutation` field via a composition wrapper `RLEpisodeSpec`. Tuning's parallel `TuningEpisodeSpec` name is retired in favour of `EpisodeSpec` from sim.

`src/tuning/episode.py::sample_episode` and `src/rl/episode_sampler.py::sample_episode` are ~95% identical today; their byte-equivalence between `src/rl/episode_sampler.py:113–176` and `src/tuning/episode.py:77–135` (the `default_*_params` factories) is the cleanest spot to lift.

### New sim module

`src/sim/episode_sampler.py` exposes:

- `EpisodeSpec(scenario, active_subset)` — frozen dataclass. No `slot_permutation` (RL-observation-encoder concern; see CONTEXT.md "Slot-shuffled observation" + ADR 0004 Decision 2).
- `sample_episode(catalog, base_template, *, K_active, episode_length, capacity_dist, balance_dist, episode_seed, market_params=None, disruption_params=None, lifecycle_params=None, start_date=None) -> EpisodeSpec`. Broken-out primitives — no `Config` parameter, so sim stays Config-agnostic.
- 4-stream seed split using the existing `_SUB_SEED_PARAMS` constants (assortment / capacity / balance / world). The slot stream stays in RL.
- `default_market_params()`, `default_disruption_params()`, `default_lifecycle_params()` — public functions (leading underscore dropped). Consumed by `sample_episode`'s `is None` fallbacks and by `src/sim/world_loader.py`'s synthetic-fallback path (introduced in issue 05).

### Tuning-side shrink

`src/tuning/episode.py` shrinks from ~240 lines to ~15: a Config-adapter that unpacks `TuningConfig` and calls `src.sim.episode_sampler.sample_episode`. The `TuningEpisodeSpec` name is retired — tuning imports `EpisodeSpec` from sim directly. `src/tuning/__init__.py` re-exports `EpisodeSpec` (transitional, to keep notebook 08 working with one import-line update) and `sample_episode`.

### RL-side rewrite (composition)

`src/rl/episode_sampler.py` defines `RLEpisodeSpec` as a composition of sim's `EpisodeSpec`:

```python
from dataclasses import dataclass
from src.sim.episode_sampler import EpisodeSpec

@dataclass(frozen=True)
class RLEpisodeSpec:
    spec: EpisodeSpec
    slot_permutation: tuple[int, ...]
```

`sample_episode(...)` body: call `src.sim.episode_sampler.sample_episode(...)` for the `(scenario, active_subset)` pair, then derive `slot_seed` separately (5th `_SUB_SEED_PARAMS["slot"]` constant stays in RL) and compute `slot_permutation`. Wrap in `RLEpisodeSpec`.

The existing `EpisodeSpec` symbol in `src/rl/episode_sampler.py` is renamed to `RLEpisodeSpec` via mechanical grep through `src/rl/` and `tests/rl/`. Expected cost: ~10 import-line updates.

### Sim-side tests

`tests/sim/test_episode_sampler.py` covers:

- Determinism: `sample_episode(episode_seed=N)` called twice returns equal `EpisodeSpec`.
- Cross-seed difference: different seeds produce different active subsets / capacities / balances.
- 4-stream independence: perturbing the `(prime, offset)` pair for one sub-seed purpose does not change the other three outputs.
- `default_*_params()` invoked when `None` passed; explicit overrides honoured.

### CONTEXT.md update (inline)

Add new "Episode sampler" architecture entry pointing at `src.sim.episode_sampler` and noting the `default_*_params` factories live there as synthetic-fallback defaults. Note that the `slot` stream and `slot_permutation` stay in `src/rl/episode_sampler.py::RLEpisodeSpec`.

## Acceptance criteria

- [ ] `src/sim/episode_sampler.py` exists; exports `EpisodeSpec`, `sample_episode`, `default_market_params`, `default_disruption_params`, `default_lifecycle_params`.
- [ ] `src/sim/episode_sampler.py` has no imports from `src.tuning`, `src.rl`, or `src.llm`.
- [ ] `src/tuning/episode.py` is ≤30 lines; its `sample_episode` is a Config-adapter calling sim's sampler.
- [ ] `TuningEpisodeSpec` name is gone; tuning code uses `EpisodeSpec` (from sim).
- [ ] `src/rl/episode_sampler.py` exports `RLEpisodeSpec` (composing sim's `EpisodeSpec` + `slot_permutation: tuple[int, ...]`); old `EpisodeSpec` name no longer used inside `src/rl/`.
- [ ] `src/tuning/__init__.py` re-exports `EpisodeSpec` and `sample_episode` from sim (transitional).
- [ ] `tests/sim/test_episode_sampler.py` exists with the four tests above; passes.
- [ ] `tests/tuning/test_evaluator.py::test_no_rl_import` continues to pass.
- [ ] CONTEXT.md has new "Episode sampler" entry.
- [ ] `uv run pytest tests/sim/ tests/tuning/ tests/rl/` is green.

## Blocked by

- `01-lift-metrics-to-sim.md` — shares the `src/sim/` ownership pattern; tuning/RL shrink touches the same files that import metrics, easier to land in order.
- `03-simulation-build-world-tick-result.md` — `EpisodeSpec` is consumed alongside `Simulation` in the shrunken shims; landing the runner refactor first keeps the import surface coherent.
