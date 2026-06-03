# 09 — CONTEXT.md consolidation + grep audit + end-to-end smoke

Status: ready-for-agent

## Parent

PRD: `.scratch/sim-base-rollout-dedupe/PRD.md`

## What to build

Final-gate slice. The bulk of the CONTEXT.md updates lands inline in issues 02–05 (per the PRD's "inline during execution" pattern); this slice does the consolidation pass — verifies coherence, tightens cross-references, and updates the entries that depend on the whole refactor being complete. It also runs the cross-cutting verifications: the grep audit and the four end-to-end smoke checks.

### CONTEXT.md consolidation

Verify and tighten:

- "Simulation" entry (added in issue 03) reads coherently with "Runner", "Scenario", and "CRN-paired eval".
- "Episode sampler" entry (added in issue 04) is consistent with the "Slot-shuffled observation" entry and notes that `RLEpisodeSpec` wraps sim's `EpisodeSpec`.
- "World loader" entry (added in issue 05) cross-references "WorldBuilder" and "World".
- "World" / "WorldBuilder" split (added in issue 02) reads clean.
- "CRN-paired eval" entry: point at `src/sim/metrics.py::RunSlice` and `aggregate_episode` for the run-projection types, and at `Simulation` / `build_world` for the rollout machinery (today the entry implies the location; after this PRD it should name it).
- "Policy tuning study" entry: remove the implicit "from `src.rl`" sourcing; point at sim modules.

Any inline edits from earlier issues that conflict (e.g. the "Runner" entry update from issue 03 vs the "Policy tuning study" entry update needed now) get reconciled.

### Grep audit (final gate)

- `grep -r "src.rl" src/tuning/` returns at most the `test_no_rl_import` guard string.
- `grep -r "src.tuning" src/rl/` returns nothing.
- `grep -r "from src.llm.world_builder import World" src/ tests/` returns nothing (the transitional re-export from issue 02 stays, but internal call sites use `src.sim.world` directly).

### End-to-end smoke checks

All four must succeed:

1. **Tuning study**: `uv run python -m src.tuning.study --world fashion_retail_250 --n-trials 5` runs to completion and produces `trials.parquet` + `per_seed.parquet`. (Numeric equality vs the pre-refactor baseline is already gated by issue 06's smoke; this run is just a re-check that the end-to-end command works after all subsequent slices land.)
2. **RL eval**: a short RL eval run completes; per-eval mean reward matches the pre-refactor number to within `1e-9` on the same seed. (Already gated by issue 07; re-check end-to-end.)
3. **RL training**: `uv run python -m src.rl.train --total-env-steps 5000 --n-envs 2` runs to completion; reward curve at the first eval point matches the pre-refactor number to within `1e-9` on the same `--seed`. (Already gated by issue 08; re-check end-to-end.)
4. **Notebook 08**: open `notebooks/08-tune_textbook_policy.ipynb` and run all cells against `fashion_retail_250` (336 products); every cell completes cleanly with no errors.

## Acceptance criteria

- [ ] CONTEXT.md "CRN-paired eval", "Policy tuning study", "Simulation", "Episode sampler", "World loader", "World", "WorldBuilder", "Runner", "Scenario" entries are all coherent and cross-referenced.
- [ ] `grep -r "src.rl" src/tuning/` returns at most the `test_no_rl_import` guard string.
- [ ] `grep -r "src.tuning" src/rl/` returns nothing.
- [ ] `grep -r "from src.llm.world_builder import World" src/ tests/` returns nothing.
- [ ] Tuning smoke runs cleanly: `uv run python -m src.tuning.study --world fashion_retail_250 --n-trials 5`.
- [ ] RL eval smoke matches pre-refactor to within `1e-9`.
- [ ] RL training smoke: `uv run python -m src.rl.train --total-env-steps 5000 --n-envs 2` matches pre-refactor reward curve to within `1e-9` at the first eval point.
- [ ] All cells in `notebooks/08-tune_textbook_policy.ipynb` execute cleanly against `fashion_retail_250`.
- [ ] `uv run pytest tests/sim/ tests/tuning/ tests/rl/ tests/llm/` is green.

## Blocked by

- `01-lift-metrics-to-sim.md`
- `02-lift-world-dataclass-to-sim.md`
- `03-simulation-build-world-tick-result.md`
- `04-lift-episode-sampler.md`
- `05-lift-world-loader.md`
- `06-migrate-tuning-rollout.md`
- `07-migrate-rl-eval.md`
- `08-migrate-rl-env.md`
