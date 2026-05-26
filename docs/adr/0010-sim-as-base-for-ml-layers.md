# sim as base for ML layers

## Context

The repo had four copies of the same per-tick rollout state machine across `src/sim/`, `src/tuning/`, `src/rl/eval.py`, and `src/rl/env.py`. Three adjacent primitives orbited the rollout with identical or near-identical implementations:

- **Metrics** — `src/tuning/metrics.py` and `src/rl/metrics.py` were byte-equivalent (`RunSlice` + `aggregate_episode` + five KPI helpers).
- **Episode sampler** — `src/tuning/episode.py::sample_episode` and `src/rl/episode_sampler.py::sample_episode` differed only in RL's 5th seed stream (`slot`) and `slot_permutation` field; the 4-stream split, `Scenario` build, and default parameter factories were identical.
- **World loader** — `src/tuning/world_loader.py::load_world` and `src/rl/train.py::_load_world_catalog_and_template` followed the same cache → archetype → synthetic fallback resolution order.

Additionally, the `World` artifact (persisted catalog + market + store_templates dataclass) was defined in `src/llm/world_builder.py` — an inverted layer dependency where `src/sim/` depended on `src/llm/` for a domain artifact.

This duplication was load-bearing on the most subtle property in the codebase: **CRN bit-identity** (ADR 0003). Four divergent copies of the per-tick state machine were four places that contract could quietly drift. ADR 0009's "module layout" section noted that tuning consumed from `src/rl/episode_sampler` and `src/rl/eval._build_world` — a copy-then-decouple pattern that preserved the duplication rather than resolving it.

## Decision

**`src/sim/` is the canonical home for rollout primitives, metrics, episode sampling, world loading, and the `World` artifact. `src/tuning/` and `src/rl/` are sibling consumers; neither imports from the other.**

Five modules established or refactored in `src/sim/`:

1. **`src/sim/metrics.py`** — `RunSlice` dataclass + `aggregate_episode(run_slice) -> dict[str, float]` + five KPI helpers (`service_level`, `stockout_rate`, `mean_price_pct_of_msrp`, `inventory_turnover`, `profit_decomposition`). Pure data + math; no runtime deps on `src.tuning`, `src.rl`, or `src.llm`. Verbatim move from `src/tuning/metrics.py`; byte-equivalent `src/rl/metrics.py` deleted.

2. **`src/sim/world.py`** — `World` dataclass + `to_json` / `from_json` / `to_dict` / `from_dict`. Pure data + serialisation. Moved from `src/llm/world_builder.py`; `src/llm/world_builder.py` keeps `WorldBuilder`, `load_or_build_world`, `LLMBuildAbortedError`, and adds a transitional `from src.sim.world import World as World` re-export.

3. **`src/sim/episode_sampler.py`** — `EpisodeSpec(scenario, active_subset)` frozen dataclass + `sample_episode(...)` + three public factories `default_market_params()` / `default_disruption_params()` / `default_lifecycle_params()`. 4-stream seed split (assortment / capacity / balance / world) lives in sim. RL's 5th `slot` stream stays in `src/rl/episode_sampler.py` where it composes sim's `EpisodeSpec` into `RLEpisodeSpec(spec, slot_permutation)`.

4. **`src/sim/world_loader.py`** — `load_world(*, archetype, cache_path, ...)` → 4-tuple. Resolution: explicit `cache_path` → `data/worlds/<archetype>/world.json` → synthetic fallback. Both `src/tuning/world_loader.py` and `src/rl/train.py::_load_world_catalog_and_template` collapse to ~10-line Config adapters around it.

5. **`src/sim/runner.py`** (refactored) — gains `Simulation` (mutable bundle: `scenario`, `world_rng`, `item_registry`, `market`, `event_engine`, `stores`), `build_world(scenario, *, policy_overrides=None) -> Simulation` (module-level free function), and `TickResult` (frozen dataclass: `actions`, `demand_traces`, `active_events`). `Simulation` exposes three methods:

   - `tick_world() -> list[WorldEvent]` — phase 1: world advances (`market.tick → event_engine.tick → item_registry.tick`).
   - `tick_decide_and_settle() -> TickResult` — phase 2: per-store `observe → decide → _dispatch_orders → _process_demand`.
   - `tick() -> TickResult` — convenience composing both phases.

   The existing `Runner` class keeps its public API unchanged; its body shrinks to a composition of `build_world` + `Simulation.tick()` + existing `_init_run_log` / `_log_state` helpers.

**Policy attach pattern** — `build_world(scenario, *, policy_overrides=None)` accepts a list-of-Policy override. When set, each `Store` is constructed with the override policy in place of `StoreInstance.policy`. The override wins when both are set. Spec-based callers (tuning, RL eval, RL env — whose specs have `StoreInstance.policy=None`) supply `policy_overrides`; scenario-authoring callers (main.py, hand-authored scenarios) leave it `None`.

**Layer shims** — `src/tuning/episode.py` shrinks to a Config-adapter that unpacks `TuningConfig` and calls `src.sim.episode_sampler.sample_episode`. `src/tuning/world_loader.py` shrinks to a Config-adapter that unpacks `TuningConfig` and calls `src.sim.world_loader.load_world`. The historical `TuningEpisodeSpec` name is retired; `EpisodeSpec` from sim is used directly.

## Consequences

- Future ML layers (imitation learning, model-based RL, etc.) depend on `src/sim/`, never on `src/tuning/` or `src/rl/`. A new rollout consumer never becomes a fifth copy of the state machine.
- The CRN bit-identity contract (ADR 0003) has exactly one implementation to audit — `Simulation.tick_decide_and_settle()`. The per-tick demand-sampling order (every catalog product, every tick, in `store.inventory` iteration order) is preserved exactly.
- `grep -r "src.rl" src/tuning/` returns at most the `test_no_rl_import` guard string. `grep -r "src.tuning" src/rl/` returns nothing. The sibling-layer constraint is structurally enforced.
- `src/llm/world_builder.py` keeps a transitional `from src.sim.world import World as World` re-export so any external `from src.llm.world_builder import World` keeps working. Internal call sites import from `src.sim.world` directly.
- The two-phase tick API (`tick_world` / `tick_decide_and_settle`) exposes the seam where RL action-injection happens without forcing simple callers to think about it — they use `tick()`.

## Cross-references

- ADR 0003 — CRN demand for all products (bit-identity contract preserved by this refactor)
- ADR 0004 — RL training env (slot-permutation invariant stays RL-specific; `RLEpisodeSpec` in `src/rl/episode_sampler.py`)
- ADR 0009 — Policy hyperparameter tuning tool (section (a) amended in place by issue 06 to reflect tuning consumes from `src/sim/`, not `src/rl/`)

## Implementation status

Issues in the `sim-base-rollout-dedupe` PRD implement this ADR in waves:

- **Issue 01** (this slice) — `src/sim/metrics.py` established; `src/tuning/metrics.py` and `src/rl/metrics.py` deleted; ADR 0010 written.
- **Issue 02** — `src/sim/world.py` established; `World` relocated from `src/llm/world_builder.py`.
- **Issue 03** — `Simulation`, `build_world`, `TickResult` in `src/sim/runner.py`; CRN determinism gate tests.
- **Issues 04–05** — `src/sim/episode_sampler.py`, `src/sim/world_loader.py`.
- **Issues 06–08** — all four rollout consumers migrated to sim primitives.
- **Issue 09** — CONTEXT.md consolidation + grep audit + end-to-end smoke.
