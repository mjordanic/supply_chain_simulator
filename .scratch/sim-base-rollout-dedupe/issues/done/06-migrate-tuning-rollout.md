# 06 — Migrate `src/tuning/rollout.py::run_policy_episode` to sim primitives; amend ADR 0009 section (a)

Status: ready-for-agent

## Parent

PRD: `.scratch/sim-base-rollout-dedupe/PRD.md`
ADR: `docs/adr/0009-policy-hyperparameter-tuning-tool.md` (amended in this slice)

## What to build

Rewrite `src/tuning/rollout.py::run_policy_episode` so it composes the sim primitives introduced in issues 01, 03, and 04 instead of carrying its own per-tick state machine. Delete the four private helpers (`_build_world`, `_make_delivery_callback`, `_dispatch_orders`, `_process_demand`) — they are now redundant with `build_world` + `Simulation.tick`. The public function shrinks from ~150 lines to ~50.

The new body uses the `policy_overrides` attach pattern from issue 03 so the spec stays immutable from the evaluator's perspective; the historical `spec.scenario.stores[0].policy = policy` in-place mutation is gone.

Target shape (illustrative, not a literal spec):

```python
def run_policy_episode(policy: Policy, spec: EpisodeSpec) -> dict[str, float]:
    sim = build_world(spec.scenario, policy_overrides=[policy])
    run_slice = RunSlice(active_pids=list(spec.active_subset))
    catalog_lookup = {w.product_id: w for w in spec.scenario.catalog}
    order_fee = spec.scenario.stores[0].template.order_fee
    for _ in range(spec.scenario.n_steps):
        result = sim.tick()
        _record_active_subset(sim.stores[0], result.actions[0],
                              spec.active_subset, run_slice,
                              catalog_lookup, order_fee)
    return aggregate_episode(run_slice)
```

`_record_active_subset` is a small (~15-line) tuning-side helper that projects the per-tick state onto the active-subset `RunSlice`. It reads cost off `Ware.unit_cost` via `catalog_lookup`; the PRD notes this is equivalent to today's `store.costs[pid]` because `store.costs` is populated from `Ware.unit_cost` at construction — verify with a single-pid assertion in `tests/sim/test_metrics.py` or `tests/tuning/test_evaluator.py`.

### CRN bit-identity gate

The CRN tests from issue 03 must stay green. Additionally, this slice runs the end-to-end tuning study smoke as a manual verification: `uv run python -m src.tuning.study --world fashion_retail_250 --n-trials 5 --output runs/tuning/baseline_pre` (on the pre-refactor commit) and `--output runs/tuning/baseline_post` (on this slice). Assert numeric equality on `per_seed.parquet` and `trials.parquet` columns via `pd.read_parquet(...).equals(...)` — use `equals(...)`, not byte-diff, because parquet binary layout can vary.

### ADR 0009 amendment

`docs/adr/0009-policy-hyperparameter-tuning-tool.md` section (a) "Module layout" is amended in place. Replace the sentence stating tuning consumes from `src/rl/episode_sampler` and `src/rl/eval._build_world` with a paragraph noting those helpers were lifted to `src/sim/` (`src/sim/episode_sampler.py`, `src/sim/world_loader.py`, `src/sim/metrics.py`, and `Simulation` / `build_world` in `src/sim/runner.py`). Tuning consumes sim directly; RL is a sibling layer, not a parent. Mark the amendment clearly as an in-place update to ADR 0009 (per user direction, not a new ADR 0010).

### Grep guard

After this slice, `grep -r "src.rl" src/tuning/` must return at most the `test_no_rl_import` guard string. The full grep audit lives in issue 09 but tuning's portion of it is locked in here.

## Acceptance criteria

- [ ] `src/tuning/rollout.py::run_policy_episode` body uses `build_world(scenario, policy_overrides=[policy])` + `Simulation.tick()` + `aggregate_episode`; no longer mutates `spec.scenario.stores[0].policy`.
- [ ] The four private helpers `_build_world`, `_make_delivery_callback`, `_dispatch_orders`, `_process_demand` are deleted from `src/tuning/rollout.py`.
- [ ] `src/tuning/rollout.py` is ≤80 lines.
- [ ] Public signature `run_policy_episode(policy, spec) -> dict[str, float]` is unchanged.
- [ ] `src.tuning.run_study(policy_space, ...)` public API behaviour is unchanged (a smoke `--n-trials 5` study still completes).
- [ ] `tests/tuning/test_evaluator.py::test_no_rl_import` passes; `grep -r "src.rl" src/tuning/` returns at most that guard string.
- [ ] CRN determinism tests from issue 03 (`tests/sim/test_runner_refactor.py`) remain green.
- [ ] Tuning smoke: `uv run python -m src.tuning.study --world fashion_retail_250 --n-trials 5` produces `trials.parquet` / `per_seed.parquet` numerically equal (via `pd.DataFrame.equals` on each column) to the pre-refactor output on the same args.
- [ ] ADR 0009 section (a) is amended in place to reflect the sim-as-base architecture; amendment is clearly marked as such.
- [ ] `uv run pytest tests/sim/ tests/tuning/ tests/rl/` is green.

## Blocked by

- `01-lift-metrics-to-sim.md` — imports `RunSlice` and `aggregate_episode` from `src.sim.metrics`.
- `03-simulation-build-world-tick-result.md` — uses `build_world` and `Simulation.tick`.
- `04-lift-episode-sampler.md` — imports `EpisodeSpec` from `src.sim.episode_sampler`.
