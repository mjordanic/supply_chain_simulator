# 02 — Rename `BaselinePolicy` → `HeuristicPolicy` + atomic code migration

Status: ready-for-agent

## Parent

PRD: `.scratch/textbook-reorder-policies/PRD.md`
ADR: `docs/adr/0006-textbook-reorder-policy-family.md`

## What to build

The atomic-rename-and-migration slice. After this slice no class named `BaselinePolicy` exists in the source tree; every existing `from src.sim.policy import BaselinePolicy` raises `ImportError`. The PRD forbids a compatibility alias — this is a hard break — so the rename and every call-site update must land in a single PR to keep main green.

The work has three parallel concerns that all ship together:

**(a) The rename itself.** `BaselinePolicy` in `src/sim/policy.py` is renamed `HeuristicPolicy`, body byte-identical. `__all__` updated. `tests/sim/test_baseline_policy.py` is renamed `tests/sim/test_heuristic_policy.py` with imports flipped to `HeuristicPolicy`; no assertion changes (this is the proof the rename is behaviour-preserving).

**(b) Bucket A — files that adopt `HeuristicPolicy` (the rename target):** files that exist to demonstrate the kitchen-sink policy framework. The old kwargs are preserved verbatim — `HeuristicPolicy` has the same body and the same constructor.

- `scenarios/example_homogeneous.py`
- `scenarios/example_paired_comparison.py`
- `scenarios/example_llm_world.py`
- `scenarios/example_llm_world_offline.py`
- `tests/test_examples_and_cli.py`

**(c) Bucket B — files that adopt `OrderUpToPolicy()` (the new CRN comparison anchor):** production scenarios, full-run tests, and the RL training/eval modules. The old heuristic kwargs are **dropped**, not translated — they refer to behaviour (dynamic pricing, promotions, deactivation, periodic catalog review) the textbook policy does not have.

- `scenarios/llm_world_20.py`
- `scenarios/llm_world_100.py`
- `scenarios/llm_world_1000.py`
- `tests/sim/test_full_run.py`
- `tests/sim/test_scenario.py`
- `tests/rl/test_episode_sampler.py`
- `tests/rl/test_eval_crn.py`
- `src/rl/eval.py` (type-hint import only — `evaluate(baseline_policy_factory, …)` signature unchanged)
- `src/rl/train.py` (factory body switches to `OrderUpToPolicy()`)

**Regression snapshot.** `tests/sim/test_regression_snapshot.py` is regenerated against `OrderUpToPolicy` with default kwargs. The old `BaselinePolicy` trajectory is no longer pinned (deliberate — `HeuristicPolicy` exists for demonstration, not as a regression anchor). The regenerated snapshot becomes the new pinned trajectory; future tuning of `cover_horizon_ticks`, `safety_lead_ticks`, or `opening_budget_pct` will surface as a snapshot-diff.

**Import-failure smoke test.** Add a test (e.g. `tests/sim/test_no_baseline_alias.py` or append to `tests/sim/test_textbook_policy.py`) asserting that `from src.sim.policy import BaselinePolicy` raises `ImportError`. This is the auditable proof the hard-break property holds.

**README + docstrings.** Search for any prose references to `BaselinePolicy` in `README.md` and module/class docstrings; update to refer to `HeuristicPolicy` (when the kitchen-sink demonstrator is meant) or `OrderUpToPolicy` (when the comparison anchor is meant). The PRD's user story 19 specifies "refer to `OrderUpToPolicy` explicitly (not the generic 'baseline')" — apply the same rule to README prose where appropriate.

**Out of scope here:** notebook updates (`notebooks/06-*.ipynb`, `notebooks/07-*.ipynb`) — those land in issue 03.

End-to-end demoability: after this slice, `uv run pytest`, `uv run python scenarios/llm_world_20.py`, and `uv run python scenarios/llm_world_1000.py` all succeed; `from src.sim.policy import BaselinePolicy` raises `ImportError`.

## Acceptance criteria

- [ ] `src/sim/policy.py` no longer defines `BaselinePolicy`; defines `HeuristicPolicy` with body byte-identical to the previous `BaselinePolicy` (same kwargs, same logic).
- [ ] `__all__` in `src/sim/policy.py` lists `HeuristicPolicy` and not `BaselinePolicy`.
- [ ] `from src.sim.policy import BaselinePolicy` raises `ImportError`; an explicit test asserts this.
- [ ] All five Bucket A files import `HeuristicPolicy` (no `BaselinePolicy` references remain).
- [ ] All nine Bucket B files import `OrderUpToPolicy`; old heuristic kwargs are dropped, not translated.
- [ ] `tests/sim/test_baseline_policy.py` is renamed to `tests/sim/test_heuristic_policy.py`; imports flipped; assertions unchanged.
- [ ] `tests/sim/test_regression_snapshot.py` is regenerated against `OrderUpToPolicy` defaults; the new snapshot is committed.
- [ ] `src/rl/eval.py::evaluate` signature is unchanged (still `baseline_policy_factory: Callable[[], Policy]`); only the type-hint imports and any prose docstrings update.
- [ ] `src/rl/train.py`'s baseline factory body returns `OrderUpToPolicy()`.
- [ ] `README.md` prose no longer references `BaselinePolicy`; references to the comparison anchor name it as `OrderUpToPolicy`.
- [ ] No `grep -rn "BaselinePolicy" src tests scenarios README.md` hit anywhere in the tracked source tree (notebooks excluded — they're issue 03's surface).
- [ ] `uv run pytest` passes (full suite).
- [ ] `uv run python scenarios/llm_world_20.py` completes without raising.
- [ ] `uv run python scenarios/llm_world_1000.py` completes without raising.

## Blocked by

- `01-textbook-base-and-order-up-to.md` (consumes `OrderUpToPolicy`)
