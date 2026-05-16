# 01 — Lift `RunSlice` + `aggregate_episode` to `src/sim/metrics.py`

Status: ready-for-agent

## Parent

PRD: `.scratch/sim-base-rollout-dedupe/PRD.md`
ADRs: `docs/adr/0003-crn-demand-for-all-products.md`, `docs/adr/0009-policy-hyperparameter-tuning-tool.md`

## What to build

Create `src/sim/metrics.py` as the single home for the active-subset KPI machinery. The module is pure data + math: `RunSlice` (dataclass), `aggregate_episode(run_slice) -> dict[str, float]`, and the five KPI helpers (`service_level`, `stockout_rate`, `mean_price_pct_of_msrp`, `inventory_turnover`, `profit_decomposition`) plus their private utilities (`_flat`, `_safe_sum`, `_safe_mean`).

`src/tuning/metrics.py` and `src/rl/metrics.py` are byte-equivalent today. Verbatim move from one of them into `src/sim/metrics.py`, then delete both originals and re-point importers in `src/tuning/rollout.py` and `src/rl/eval.py` to `src.sim.metrics`. The RL-side test file `tests/rl/test_metrics.py` content largely transfers into a new `tests/sim/test_metrics.py`.

This slice establishes the `src/sim/` ownership pattern that the rest of the PRD builds on. No behaviour change; no consumer of the public KPI dict sees any difference.

### ADR 0010 — sim as base for ML layers

This slice also lands the architectural-decision record that makes the layering durable beyond the PRD's archival. Create `docs/adr/0010-sim-as-base-for-ml-layers.md` capturing:

- **Context**: four copies of the per-tick rollout state machine across `src/sim/`, `src/tuning/`, `src/rl/`; three copies of adjacent primitives (metrics, episode sampler, world loader); load-bearing on the CRN bit-identity contract (ADR 0003) and the slot-permutation invariant (ADR 0004).
- **Decision**:
  1. `src/sim/` owns rollout primitives, metrics, episode sampler, world loader, and the `World` artifact.
  2. `src/tuning/` and `src/rl/` are sibling consumers; neither imports from the other (enforced by `tests/tuning/test_evaluator.py::test_no_rl_import`).
  3. Two-phase tick API: `Simulation.tick_world()` advances the world; `Simulation.tick_decide_and_settle()` runs per-store decide+settle; `Simulation.tick()` composes both. RL injects actions in the seam.
  4. `policy_overrides` is the canonical attach pattern for spec-based callers (override wins when `StoreInstance.policy` is also set).
  5. `src/llm/world_builder.py` produces `World`s but no longer defines the type — `World` lives in `src/sim/world.py`.
- **Consequences**: future ML layers (imitation, model-based RL, etc.) depend on sim, never on tuning or RL; CRN bit-identity contract has exactly one implementation to audit; a new rollout consumer never becomes a fifth copy of the state machine.
- **Cross-references**: ADR 0003 (CRN), ADR 0004 (slot invariant stays RL-specific), ADR 0009 (amended in place by issue 06 to reflect tuning consumes sim).

The ADR documents the decision at the moment `src/sim/` first becomes canonical (this slice). Issues 02–09 implement the rest of it. The implementing agent may reference the in-flight implementation surface in the "Decision" section — file:line references to the canonical primitives are acceptable once issues 02–08 land, but the ADR itself ships in this slice as the upfront record.

## Acceptance criteria

- [ ] `src/sim/metrics.py` exists and exports `RunSlice`, `aggregate_episode`, `service_level`, `stockout_rate`, `mean_price_pct_of_msrp`, `inventory_turnover`, `profit_decomposition`.
- [ ] `src/sim/metrics.py` has no runtime imports from `src.tuning`, `src.rl`, or `src.llm`.
- [ ] `src/tuning/metrics.py` and `src/rl/metrics.py` are deleted.
- [ ] `src/tuning/rollout.py` and `src/rl/eval.py` import metrics from `src.sim.metrics`.
- [ ] `tests/sim/test_metrics.py` exercises `aggregate_episode` on a hand-built `RunSlice` covering all five KPI helpers; passes.
- [ ] `tests/tuning/test_evaluator.py::test_no_rl_import` continues to pass (tuning ↔ RL decoupling preserved).
- [ ] `docs/adr/0010-sim-as-base-for-ml-layers.md` exists and captures the layering decision, the two-phase tick API, the `policy_overrides` attach pattern, and cross-references to ADRs 0003 / 0004 / 0009.
- [ ] `uv run pytest tests/sim/ tests/tuning/ tests/rl/` is green.

## Blocked by

None — can start immediately.
