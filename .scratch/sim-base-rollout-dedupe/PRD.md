# PRD: Make `src/sim/` the single source of truth for rollout, episode sampling, world loading, and metrics

Status: ready-for-agent

Related ADRs:
- [ADR 0003](../../docs/adr/0003-crn-demand-for-all-products.md) — CRN demand for all products (the bit-identity contract this PRD must preserve)
- [ADR 0004](../../docs/adr/0004-rl-training-env.md) — RL training env (the slot-permutation concern stays RL-specific; this PRD respects that boundary)
- [ADR 0009](../../docs/adr/0009-policy-hyperparameter-tuning-tool.md) — Policy hyperparameter tuning tool (amended in place by this PRD: tuning's dependency on `src/rl/` is inverted in favour of `src/sim/`)

## Problem Statement

The repo currently has four copies of the same per-tick rollout state machine and three copies of the same primitives that surround it:

1. **`src/sim/runner.py::Runner`** — canonical rollout that drives `market.tick → event_engine.tick → item_registry.tick → store.observe → store.decide → dispatch → demand` and returns a verbose `run_log`. The intended source of truth.
2. **`src/tuning/rollout.py::run_policy_episode`** — duplicate of the above, projected to a `RunSlice` over the episode's active subset.
3. **`src/rl/eval.py::_run_rl` + `_run_baseline`** — third copy; CRN-paired evaluation path.
4. **`src/rl/env.py::RLEnv.step`** — fourth copy; gymnasium-compatible RL training env.

Adjacent duplicates orbit the rollout:

- **Metrics.** `src/tuning/metrics.py` ↔ `src/rl/metrics.py` are byte-equivalent (`RunSlice` + `aggregate_episode` + the four KPI helpers).
- **Episode sampler.** `src/tuning/episode.py::sample_episode` ↔ `src/rl/episode_sampler.py::sample_episode` differ only in RL's 5th seed stream (`slot`) and `slot_permutation` field; the 4-stream split, the `Scenario` build, and the default `MarketParams` / `DisruptionParams` / `ItemLifecycleParams` factories are identical.
- **World loader.** `src/tuning/world_loader.py::load_world` ↔ `src/rl/train.py::_load_world_catalog_and_template` follow the same `cache_path → archetype → synthetic fallback` resolution order and return the same 4-tuple.
- **`World` artifact.** `src/llm/world_builder.py::World` (the persisted catalog + market + store_templates dataclass) is imported by both `src/tuning/world_loader.py` and `src/rl/train.py`. The sim package depends on the LLM package for a domain artifact — an inverted layer dependency.

The duplication is load-bearing on the most subtle property in the codebase: **CRN bit-identity**. ADR 0003 fixes per-tick demand sampling for every catalog product; ADR 0004 fixes the slot-shuffle invariant; CONTEXT.md's "CRN-paired eval" entry depends on `OrderUpToPolicy` and the RL agent producing bit-identical trajectories on the same `(world_seed, init_seed, capacity, balance, active subset, slot permutation)` tuple. Four divergent copies of the per-tick state machine are four places that contract can quietly drift. Today they happen to agree because they were copied at similar times; nothing structurally enforces that they continue to.

A second problem: ADR 0009 (the policy-tuning ADR) explicitly puts `src/tuning/` *on top of* `src/rl/`. The original tuning module imported `_build_world` and `sample_episode` from RL. A recent decoupling pass (commit `460170f` and ancestors on this branch) copied those helpers into tuning so that the import was deleted — but the duplication was preserved, not resolved. The cleanest fix is to flip the architecture: `src/sim/` becomes the home for the shared primitives, and `src/tuning/` and `src/rl/` are sibling layers that both consume sim. Tuning never imports from RL; RL never imports from tuning.

## Solution

Make `src/sim/` the single source of truth for:

1. **Rollout building blocks** — `build_world(scenario, *, policy_overrides=None) -> Simulation`, `Simulation.tick_world()`, `Simulation.tick_decide_and_settle()`, `Simulation.tick()`, and a `TickResult` dataclass. The two-phase tick API exposes the seam where RL action-injection happens (between world ticks and the store's `decide()` call).
2. **Metrics** — `RunSlice` and `aggregate_episode` move out of tuning/rl into `src/sim/metrics.py`. Pure data + math; consumed by both layers identically.
3. **Episode sampling** — `EpisodeSpec(scenario, active_subset)` and `sample_episode(...)` move into `src/sim/episode_sampler.py`. The 4-stream seed split (assortment / capacity / balance / world) lives in sim; RL's 5th `slot` stream stays in `src/rl/episode_sampler.py` where it composes sim's spec into an `RLEpisodeSpec(spec, slot_permutation)`. The shared `default_market_params()` / `default_disruption_params()` / `default_lifecycle_params()` factories live in `src/sim/episode_sampler.py` as public functions.
4. **World loading** — `load_world(*, archetype, cache_path, delivery_lag, holding_rate, order_fee, K_active, synthetic_fallback, synthetic_catalog_size)` in `src/sim/world_loader.py`. Both `src/tuning/world_loader.py` and `src/rl/train.py::_load_world_catalog_and_template` collapse to ~10-line `Config` adapters around it.
5. **The `World` artifact** — `World` dataclass lifts from `src/llm/world_builder.py` to `src/sim/world.py`. `src/llm/world_builder.py` keeps `WorldBuilder` and `load_or_build_world` and imports `World` from sim. The layering becomes: sim defines the artifact type; llm provides one way (LLM-driven) to construct it.

With these primitives in place, all four rollout consumers migrate:

- **`Runner.run()`** — body becomes `sim = build_world(scenario); for _ in range(n_steps): sim.tick()` plus the existing log-collection helpers. Public API (`Runner(scenario).run() -> dict`) unchanged.
- **`src/tuning/rollout.py::run_policy_episode`** — ~50 lines: `build_world` with `policy_overrides=[policy]`, loop `sim.tick()`, record active-subset traces into a `RunSlice`, return `aggregate_episode(run_slice)`.
- **`src/rl/eval.py::_run_baseline`** — same shape as the tuning rollout.
- **`src/rl/eval.py::_run_rl`** — uses the two-phase API: `sim.tick_world()` → encode obs → run actor → decode → `RLPolicy.set_pending_action(...)` → `sim.tick_decide_and_settle()`. RL-specific state (`sales_history`, `effective_rate`, `base_demand_prior`, observation encoder, action decoder, slot permutation) stays in `src/rl/eval.py`.
- **`src/rl/env.py::RLEnv.step`** — same two-phase shape as `_run_rl`. Gymnasium concerns (`observation_space`, `action_space`, `terminated`, `truncated`, `info`) stay in `RLEnv`; the per-tick state machine delegates to `Simulation`.

The CRN bit-identity contract is preserved by construction (mechanical refactor of the existing tick order and RNG-construction order) and enforced by a new determinism test suite. The four rollout copies become one; the three duplicated primitives become one each.

## User Stories

1. As a sim user running `Runner(scenario).run()` from a notebook or a CLI script, I want the verbose `run_log` produced after this refactor to be bit-identical to the pre-refactor output on the same scenario seed, so that DataExporter, downstream analysis notebooks, and committed snapshot fixtures continue to work without re-baselining.
2. As an RL researcher running paired CRN evaluation via `src/rl/eval.py::evaluate`, I want `_run_rl` and `_run_baseline` to produce the same per-episode KPI dict (to within `1e-9`) before and after the refactor on the same seed, so that the paired uplift number in `notebooks/06-compare_rl_vs_baseline.ipynb` does not silently shift.
3. As an RL researcher running `python -m src.rl.train`, I want the training reward curve at the first evaluation point to match pre-refactor numerics to within `1e-9` on the same `--seed`, so that the migration is invisible to the RL training contract.
4. As a tuning-study runner invoking `python -m src.tuning.study --world fashion_retail_250 --n-trials 5`, I want `trials.parquet` and `per_seed.parquet` to be numerically equal (column-wise via `pd.DataFrame.equals`) to the pre-refactor output on the same args, so that any committed study fixtures remain valid.
5. As a sim-side test author, I want a new `tests/sim/test_runner_refactor.py` that captures `Runner(canonical_small_scenario).run()` on the pre-refactor commit as a golden parquet and asserts equality after the refactor, so that future regressions in the sim base layer are caught at the test boundary instead of in downstream eval.
6. As the same test author, I want a cross-path equivalence test that drives the same scenario via `Runner.run()` and via `sim = build_world(...); for _ in range(n_steps): sim.tick()` and asserts end-of-run state equality on `(world_rng.getstate(), stores[0].balance, stores[0].inventory, stores[0].demand, stores[0].sales, market.market_state, item_registry lifecycle stages)`, so that the two public sim entry points provably share the same trajectory.
7. As a tuning-module user, I want `src.tuning.run_study(policy_space, ...)` to keep its public API unchanged after the refactor, so that any existing tuning script keeps working with no edits.
8. As a custom-policy author writing my own `Policy` subclass, I want to be able to import `EpisodeSpec` and `sample_episode` from `src.sim.episode_sampler` (instead of going through tuning or RL), so that authoring a new ML layer doesn't drag in tuning or RL dependencies.
9. As a maintainer auditing the cross-layer dependency graph, I want `grep -r "src.rl" src/tuning/` and `grep -r "src.tuning" src/rl/` to return nothing (modulo `test_no_rl_import` guard strings), so that the tuning ↔ RL decoupling is structurally enforced.
10. As an LLM-world consumer, I want `data/worlds/<archetype>/world.json` files written by the pre-refactor codebase to continue loading via `World.from_json` after the relocation from `src/llm/world_builder.py` to `src/sim/world.py`, so that committed world caches remain valid.
11. As an LLM-world author calling `src.llm.world_builder.load_or_build_world(...)`, I want the function signature and behaviour to remain unchanged after `World` moves to sim, so that scenario-authoring scripts keep working.
12. As a maintainer who reads CONTEXT.md to onboard, I want a new "Simulation" architecture entry naming the bundle returned by `build_world`, the two-phase tick API, and how it relates to `Runner`, so that the new sim surface is discoverable from the glossary.
13. As the same reader, I want a new "Episode sampler" architecture entry pointing at `src.sim.episode_sampler` (with a note about `default_*_params` as synthetic-fallback defaults) and a new "World loader" entry pointing at `src.sim.world_loader`, so that the lifted primitives are findable.
14. As a CONTEXT.md reader hitting the "WorldBuilder" entry, I want it split into "World" (the persisted artifact, now in `src/sim/world.py`) and "WorldBuilder" (the LLM-driven generator, still in `src/llm/world_builder.py`), so that the type/builder distinction is clear in the glossary.
15. As an RL agent author who needs the actor to see post-world obs before the decide phase fires, I want `Simulation.tick_world()` and `Simulation.tick_decide_and_settle()` exposed as separate methods, so that I can insert encoding + inference + `RLPolicy.set_pending_action(...)` between them.
16. As a textbook-policy author whose policy uses `Store.policy.decide(obs)` normally, I want `Simulation.tick()` to compose both phases into a single call, so that simple callers don't have to think about the seam.
17. As a tuning evaluator caller, I want to pass a fresh policy via `build_world(scenario, policy_overrides=[policy])` rather than mutating `spec.scenario.stores[0].policy` in place, so that the spec stays immutable from my perspective and concurrent calls on the same spec list are safe.
18. As a scenario author whose `StoreInstance.policy` is non-`None`, I want the documented rule that `policy_overrides` wins when both are set, so that the override pattern doesn't surprise me by silently being ignored.
19. As an RL `episode_sampler` consumer, I want `RLEpisodeSpec` (composing sim's `EpisodeSpec` and adding `slot_permutation`) to be importable from `src.rl.episode_sampler`, so that RL training and RL eval keep working with the slot-shuffle invariant intact.
20. As a tuning-spec consumer, I want the historical `TuningEpisodeSpec` name retired and `EpisodeSpec` (from sim) used directly, so that there isn't a parallel type hierarchy with identical fields.
21. As a config-author for tuning, I want `TuningConfig` to remain the entry point for tuning's domain randomisation knobs (`K_active`, `episode_length`, `capacity_dist`, `balance_dist`, etc.) and to feed into `src.sim.episode_sampler.sample_episode` via a thin adapter, so that `src/sim/` stays Config-agnostic.
22. As an RL-config-author, I want the same Config-adapter pattern on the RL side: `RLConfig` feeds `src.sim.episode_sampler.sample_episode` and `src.sim.world_loader.load_world` via thin RL-side wrappers, so that the dependency direction stays sim ← rl.
23. As a sim-side test author, I want a dedicated `tests/sim/test_metrics.py` that exercises `aggregate_episode` on a hand-built `RunSlice` covering all five KPI functions (service level, stockout rate, mean price pct of MSRP, inventory turnover, profit decomposition), so that the deepest module gets the tightest test net.
24. As a sim-side test author, I want `tests/sim/test_world.py` to verify (a) JSON round-trip of a small synthetic `World` and (b) every `data/worlds/<archetype>/world.json` checked into the repo loads without regression, so that the relocation from `src/llm/` to `src/sim/` cannot silently break the cache format.
25. As a sim-side test author, I want `tests/sim/test_episode_sampler.py` to verify `sample_episode(seed)` is deterministic, that different seeds produce different active subsets / capacities / balances, and that the four sub-seed streams are independent (varying one purpose's prime/offset does not perturb the others), so that the CRN guarantee at sample time is locked in.
26. As a sim-side test author, I want `tests/sim/test_world_loader.py` to verify the three-tier resolution order (explicit `cache_path` hit → archetype lookup → synthetic fallback), confirm `synthetic_fallback=False` raises, and confirm the disruption-params regions adjustment matches `world.market.regions`, so that the loader's contract is pinned.
27. As a tuning-test maintainer running `tests/tuning/test_evaluator.py`, I want the `test_no_rl_import` guard to keep passing after the refactor, so that the tuning ↔ RL decoupling regression-protects itself.
28. As a notebook 08 user, I want all cells to run cleanly against `fashion_retail_250` (336-product catalog) after the refactor, so that the tuning demo notebook stays the canonical entry-point for the tool.
29. As a maintainer reading `docs/adr/0009-policy-hyperparameter-tuning-tool.md` after this PRD lands, I want section (a) "Module layout" amended in place to reflect that the tuning module consumes from `src/sim/` (not `src/rl/`), so that the ADR record matches the codebase.
30. As an ADR-history reader, I want the amendment to be a clearly-marked update to ADR 0009 (rather than a new ADR 0010) per user direction, so that the architectural pivot is colocated with the original tuning-tool decision.
31. As a CRN bit-identity researcher, I want the per-tick demand-sampling order (ADR 0003: every catalog product, every tick, in `store.inventory` iteration order) to be preserved exactly in `Simulation.tick_decide_and_settle()`, so that the bit-identity property survives the refactor.
32. As an RL-env caller hitting `RLEnv.reset()` and `RLEnv.step(action)`, I want gymnasium concerns (`observation_space`, `action_space`, `terminated`, `truncated`, `info` dict) to stay in `RLEnv` while the per-tick state machine delegates to `Simulation`, so that the gymnasium contract is independent of the sim refactor.
33. As a future ML-layer author (a new `src/imitation/` or similar), I want sim's primitives (`build_world`, `Simulation.tick*`, `sample_episode`, `load_world`, `RunSlice`, `aggregate_episode`) to be the obvious dependency target, so that the third ML layer does not become the fourth rollout copy.

## Implementation Decisions

### Architecture: sim is the base; tuning and RL are siblings

`src/sim/` becomes the canonical home for rollout, metrics, episode sampling, world loading, and the `World` artifact. `src/tuning/` and `src/rl/` are sibling layers that both depend on sim; neither imports from the other. ADR 0009 is amended in place to reflect this — its original "tuning consumes from `src/rl/episode_sampler` and `src/rl/eval._build_world`" framing was the consequence of a copy-then-decouple pattern; the canonical fix is to put the shared primitives in sim.

### New sim modules (the five deep modules)

Five modules are added to `src/sim/`. Each has a simple, testable interface that rarely changes:

1. **`src/sim/metrics.py`** — `RunSlice` (dataclass) + `aggregate_episode(run_slice) -> dict[str, float]` + the five KPI helpers (`service_level`, `stockout_rate`, `mean_price_pct_of_msrp`, `inventory_turnover`, `profit_decomposition`). Pure data + math; no `src.sim` runtime deps. Verbatim move of `src/tuning/metrics.py`; the byte-equivalent `src/rl/metrics.py` is deleted.

2. **`src/sim/world.py`** — `World` dataclass + `from_json` / `to_json` / `to_dict` / `from_dict`. Pure data + serialisation. Verbatim move from `src/llm/world_builder.py`; `src/llm/world_builder.py` keeps `WorldBuilder`, `load_or_build_world`, `LLMBuildAbortedError`, the Pydantic schemas, and adds a transitional `from src.sim.world import World as World` so existing imports keep working during the migration.

3. **`src/sim/episode_sampler.py`** — `EpisodeSpec(scenario, active_subset)` frozen dataclass + `sample_episode(catalog, base_template, *, K_active, episode_length, capacity_dist, balance_dist, episode_seed, market_params=None, disruption_params=None, lifecycle_params=None, start_date=None) -> EpisodeSpec` + three public factories `default_market_params()` / `default_disruption_params()` / `default_lifecycle_params()`. 4-stream seed split (assortment / capacity / balance / world) with the existing `_SUB_SEED_PARAMS` constants. Sim is Config-agnostic — kwargs are broken-out primitives, not a `Config` object. RL's 5th `slot` stream stays in `src/rl/episode_sampler.py`.

4. **`src/sim/world_loader.py`** — `load_world(*, archetype, cache_path=None, delivery_lag, holding_rate, order_fee, K_active, synthetic_fallback=True, synthetic_catalog_size=100) -> tuple[list[Ware], StoreTemplate, MarketParams | None, DisruptionParams | None]`. Resolution: explicit `cache_path` → `data/worlds/<archetype>/world.json` → synthetic fallback (gated by the kwarg). Calls `src.sim.world.World.from_json` and applies `replace(default_disruption_params(), regions=...)` for the regions adjustment both current copies do.

5. **`src/sim/runner.py`** (existing module; refactored) — gains `Simulation` (mutable bundle of `scenario`, `world_rng`, `item_registry`, `market`, `event_engine`, `stores`), `build_world(scenario, *, policy_overrides=None) -> Simulation` (module-level free function), and `TickResult` (frozen dataclass: `actions`, `demand_traces`, `active_events`). `Simulation` exposes three methods: `tick_world()`, `tick_decide_and_settle()`, and `tick()` (the composition). The existing `Runner` class stays, with its public API unchanged; its body shrinks to a composition of `build_world` + `Simulation.tick()` + the existing `_init_run_log` / `_log_state` log-collection helpers.

### The two-phase tick API

`Simulation.tick()` is split into two methods to accommodate RL action injection:

- `tick_world() -> list[WorldEvent]` advances `market.tick → event_engine.tick → item_registry.tick`.
- `tick_decide_and_settle() -> TickResult` runs per-store `observe → decide → _dispatch_orders → _process_demand`.
- `tick() -> TickResult` composes both.

RL `_run_rl` and `RLEnv.step` use the two halves with their own action-injection step in between (encode obs from post-world state, run actor, decode, `RLPolicy.set_pending_action(...)`, then `tick_decide_and_settle`). Tuning, baseline RL, and the legacy `Runner.run()` use the composed `tick()`.

### Policy attach pattern: `policy_overrides` wins

`build_world(scenario, *, policy_overrides=None)` accepts a list-of-Policy override. When set, each `Store` is constructed with the override policy in place of `StoreInstance.policy`. The override wins when both are set. Spec-based callers (tuning, RL eval, RL env — whose specs have `StoreInstance.policy=None`) supply `policy_overrides`; scenario-authoring callers (main.py, hand-authored scenarios) leave it `None` and let `StoreInstance.policy` flow through. This kills the in-place `spec.scenario.stores[0].policy = policy` mutation pattern and makes concurrent evaluator calls safe.

### Type composition for RL: `RLEpisodeSpec` wraps `EpisodeSpec`

`src/rl/episode_sampler.py` defines `RLEpisodeSpec(spec: EpisodeSpec, slot_permutation: tuple[int, ...])` and renames the historical `EpisodeSpec` symbol in RL accordingly. `sample_episode` in RL calls `src.sim.episode_sampler.sample_episode(...)` for the `(scenario, active_subset)` pair, then derives `slot_seed` separately (RL keeps the 5th `_SUB_SEED_PARAMS["slot"]` constant) and computes the permutation. Tuning's `TuningEpisodeSpec` name is retired; tuning imports `EpisodeSpec` from sim directly.

### Migration of all four rollout consumers

Every rollout consumer is migrated in this PR. Once migrated, the only place the per-tick state machine is implemented is `Simulation.tick_*`:

- `Runner.run()` — body becomes a composition of `build_world` + `Simulation.tick()` + the existing `_init_run_log` / `_log_state` helpers.
- `src/tuning/rollout.py::run_policy_episode` — ~50 lines; the four private helpers (`_build_world`, `_make_delivery_callback`, `_dispatch_orders`, `_process_demand`) are deleted.
- `src/rl/eval.py::_run_baseline` — same shape as the tuning rollout.
- `src/rl/eval.py::_run_rl` — uses the two-phase API; helpers deleted.
- `src/rl/env.py::RLEnv.step` — uses the two-phase API; helpers deleted.

### Layer shims

- `src/tuning/episode.py` shrinks from 240 lines to ~15: a Config-adapter that unpacks `TuningConfig` and calls `src.sim.episode_sampler.sample_episode`.
- `src/tuning/world_loader.py` shrinks from 134 lines to ~10: a Config-adapter that unpacks `TuningConfig` and calls `src.sim.world_loader.load_world`.
- `src/rl/train.py::_load_world_catalog_and_template` shrinks to ~10 lines around `src.sim.world_loader.load_world`. The private `_load_world_from_file` and `_build_synthetic_catalog` helpers are deleted.
- `src/rl/episode_sampler.py::sample_episode` body becomes a wrapper around sim's sampler plus the slot-permutation step.

### Doc updates (inline during execution)

- `docs/adr/0009-policy-hyperparameter-tuning-tool.md` section (a) "Module layout" is amended in place. The sentence stating tuning consumes from `src/rl/episode_sampler` and `src/rl/eval._build_world` is replaced with a paragraph noting those helpers were lifted to `src/sim/` (`src/sim/episode_sampler.py`, `src/sim/world_loader.py`, `src/sim/metrics.py`, `Simulation` / `build_world` in `src/sim/runner.py`). RL is a sibling layer, not a parent.
- `CONTEXT.md` gains three new architecture entries: "Simulation" (the bundle and two-phase tick API), "Episode sampler" (`src.sim.episode_sampler`, including the `default_*_params` synthetic-fallback factories), and "World loader" (`src.sim.world_loader`, including the cache → archetype → synthetic resolution order). Existing entries are updated:
  - "Runner": notes the new composition over `build_world` + `Simulation.tick()` and that the public API is unchanged.
  - "Scenario": documents the `build_world(scenario, *, policy_overrides=...)` attach pattern and the override-wins rule.
  - "CRN-paired eval": points at `src/sim/metrics.py::RunSlice` and `aggregate_episode` instead of leaving the location implicit.
  - "Policy tuning study": removes the implicit "from `src.rl`" sourcing; points at sim.
  - "WorldBuilder": split into "World" (`src/sim/world.py`, the artifact) and "WorldBuilder" (`src/llm/world_builder.py`, the LLM-driven generator).

## Testing Decisions

A good test for this PRD exercises external behaviour — public function signatures, return shapes, the CRN bit-identity property — and avoids asserting on private helper structure. The four rollout consumers should be observed via their public KPI outputs, not via internal trace dicts. The five new sim modules each get a dedicated test file under `tests/sim/`.

### New tests

1. **`tests/sim/test_metrics.py`** — KPI math correctness on a hand-built `RunSlice` covering all five aggregate functions plus the bundled `aggregate_episode`. Pure-function tests; fastest to run, highest signal. Prior art: `tests/rl/test_metrics.py` (which is being deleted with `src/rl/metrics.py`; its content largely transfers).
2. **`tests/sim/test_world.py`** — (a) round-trip a synthetic `World` through `to_json` / `from_json` and assert equality. (b) For every `data/worlds/<archetype>/world.json` checked into the repo, assert `World.from_json(path)` succeeds and produces the same `catalog`, `market`, `store_templates`, `_meta` keys as the pre-refactor codebase. Catches schema drift introduced by the relocation. Prior art: any existing `World` JSON-roundtrip tests under `tests/llm/` (will be moved).
3. **`tests/sim/test_episode_sampler.py`** — (a) determinism: `sample_episode(seed=N)` called twice returns equal `EpisodeSpec`. (b) cross-seed: different seeds → different active subsets / capacities / balances. (c) 4-stream independence: perturbing the `(prime, offset)` pair for one sub-seed purpose does not change the other three outputs. (d) `default_*_params()` invoked when None passed; explicit overrides honoured. Prior art: `tests/rl/test_episode_sampler.py` and `tests/tuning/test_episode.py` (the relevant assertions transfer).
4. **`tests/sim/test_world_loader.py`** — (a) explicit `cache_path` resolves first when present. (b) archetype lookup resolves second. (c) `synthetic_fallback=True` produces a viable 4-tuple when neither path resolves; `synthetic_fallback=False` raises. (d) disruption-params regions adjustment matches `world.market.regions`. Prior art: `tests/tuning/test_world_loader.py` (will be deleted; relevant assertions transfer).
5. **`tests/sim/test_runner_refactor.py`** — the CRN bit-identity gate.
   - **Snapshot test**: on the pre-refactor SHA, run `Runner(canonical_small_scenario).run()` and serialise the run_log to a golden parquet committed to the repo. After the refactor, regenerate and assert `pd.DataFrame.equals` on numeric columns and exact equality on integer / categorical columns. Catches Runner-output regressions in the legacy public API.
   - **Cross-path equivalence test**: after the refactor, drive the same scenario via `Runner(scenario).run()` and via `sim = build_world(scenario); for _ in range(n_steps): sim.tick()` and assert end-of-run state equality on `(world_rng.getstate(), stores[0].balance, stores[0].inventory, stores[0].demand, stores[0].sales, market.market_state, item_registry lifecycle stage map)`. Confirms the two public sim entry points share the same trajectory bit-for-bit.

### Existing tests

- `tests/tuning/test_evaluator.py::test_no_rl_import` — keep as-is; continues to enforce the tuning ↔ RL decoupling.
- `tests/tuning/` — all 77 tests stay green; some import lines update mechanically (`TuningEpisodeSpec` → `EpisodeSpec`).
- `tests/rl/` — all RL tests stay green; mechanical import updates only (`src.rl.metrics` → `src.sim.metrics`; `EpisodeSpec` → `RLEpisodeSpec` for the rl-side spec).

### End-to-end determinism smoke

Three integration checks run before merging:

- **Tuning study**: `uv run python -m src.tuning.study --world fashion_retail_250 --n-trials 5` produces `trials.parquet` and `per_seed.parquet` numerically equal (column-wise via `pd.read_parquet(...).equals(...)`) to the pre-refactor output on the same args.
- **RL eval**: a short RL eval run before and after; per-eval mean reward identical to within `1e-9`.
- **RL training**: `uv run python -m src.rl.train --total-env-steps 5000 --n-envs 2`; reward curve at the first eval point identical to within `1e-9`.
- **Notebook 08**: all cells run cleanly against `fashion_retail_250` (336 products) post-refactor.

### Grep audit (final gate)

- `grep -r "src.rl" src/tuning/` returns at most the `test_no_rl_import` guard string.
- `grep -r "src.tuning" src/rl/` returns nothing.

## Out of Scope

- **Promotion of `RunSlice` / `aggregate_episode` from "active-subset projection" to a first-class sim tracer hook.** The plan keeps the simple "each consumer projects to its own `RunSlice`" pattern. A tracer-hook design would invert the recording responsibility into `Simulation`; deferred until the simple pattern proves limiting.
- **Making `Scenario` and `StoreInstance` truly `@dataclass(frozen=True)`.** Once `policy_overrides` is everywhere, the in-place mutation footgun on `StoreInstance.policy` no longer needs to exist. The freeze is a follow-up; this PRD keeps both classes mutable to limit blast radius.
- **Concurrency in tuning trials.** ADR 0009(e) explicitly defers `n_jobs > 1`; this PRD preserves that. The new `policy_overrides` pattern makes it possible (the rollout no longer mutates spec state) but does not enable it.
- **Migrating `optuna` storage from in-memory to SQLite.** Same status as ADR 0009(f).
- **Slot permutation in tuning.** Slots are an RL-observation-encoder concern (ADR 0004). Tuning's policies (the four textbook variants) do not use slot-based observations; no slot stream is added on the tuning side.
- **Renaming `Runner`.** The class name is preserved to keep `from src.sim.runner import Runner` working in notebooks and downstream scripts.

## Further Notes

- **Reviewability**: this PRD's blast radius is large by design — five new sim modules, four rollout consumers migrated, three deleted files, ADR amendment, seven CONTEXT.md edits. The implementing PR should land the six moves as separately reviewable commits in this order: (1) lift metrics, (2) factor Runner / introduce `Simulation` + `build_world` + `TickResult`, (3) lift episode sampler, (4) lift world loader, (5) migrate all four rollout consumers, (6) lift the `World` artifact. The CRN determinism tests should land in commit (2) and remain green through commits (3)–(6).
- **Pre-refactor snapshot fixture**: before any code change, the implementing agent regenerates `tests/sim/fixtures/runner_snapshot_pre.parquet` (or equivalent) from the canonical small scenario on the current `main` SHA. That artifact is committed and never updated; it is the golden for the snapshot test. If a future refactor needs to update it, that update is a deliberate review item.
- **Transitional re-exports**: `src/llm/world_builder.py` keeps a `from src.sim.world import World as World` re-export so external scripts that historically imported `World` from llm continue to work. The implementing agent updates internal call sites; the transitional re-export is a safety net, not a permanent feature.
- **`Ware.unit_cost` vs `store.costs[pid]`**: the new tuning `_record_active_subset` helper reads cost off `Ware.unit_cost` via a catalog lookup. Today's rollout reads `store.costs[pid]`. These are equivalent because `store.costs` is populated from `Ware.unit_cost` at construction; the implementing agent verifies with a single-pid assertion in the new sim-side tests.
