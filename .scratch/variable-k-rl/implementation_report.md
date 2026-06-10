# Implementation Report — variable-k-rl

- **Feature:** variable-k-rl — variable-K shared-weight RL policy with deterministic Arbiter
- **PRD:** [PRD.md](./PRD.md)
- **Started:** 2026-06-10T13:31:05Z
- **Last updated:** 2026-06-10T18:12:00Z
- **Parallelism cap:** 3
- **Integration branch (BASE_BRANCH):** RL-rewrite
- **Runner model:** default (frontmatter)
- **Implementer model:** fable (global `--implementer-model`; applies to every issue — no per-issue `Complexity:`/`Model:` overrides present)
- **Preflight assumptions:** none — all 9 issues carry `Status: ready-for-agent`; permission audit passed with no additions; no gitignored mandated artifacts.

## Status table

| ID | Title | Wave | Status | Worktree SHA | Integrated SHA | Started | Finished | Notes |
|----|-------|------|--------|--------------|----------------|---------|----------|-------|
| 01-arbiter-module | Arbiter module: deterministic reconciliation (proportional + greedy) | 1 | committed | 0974b62c18c034051aa1997ced07d90ebafb9798 | 0974b62c18c034051aa1997ced07d90ebafb9798 | 2026-06-10T13:35:00Z | 2026-06-10T13:50:00Z | in-place mode; 29 tests, 1015/10 pass/skip |
| 02-set-encoder-decoder | Set encoder/decoder: (K_max, F) layout with mask, layout version | 2 | committed | 778a7e55abb960e18458bfb7232887dedcabe953 | 778a7e55abb960e18458bfb7232887dedcabe953 | 2026-06-10T14:00:00Z | 2026-06-10T14:20:00Z | in-place mode; 34 tests, 1049/10 pass/skip |
| 03-masked-set-actor-critic | Masked set actor-critic with masked joint Gaussian | 3 | committed | b1a47edd3c8f388a2afc1e95459712bcd015021f | f32bc9c | 2026-06-10T13:46:00Z | 2026-06-10T16:30:00Z | 21 tests; MaskedJointGaussian + SetActor + SetCritic |
| 04-checkpoint-io | Checkpoint I/O: self-describing checkpoints with layout-version validation | 3 | committed | be11b425698bbf06d5468ef0610e20d1dec8c1d1 | 42dc567 | 2026-06-10T13:46:00Z | 2026-06-10T16:30:00Z | 13 tests; layout-version mismatch error path |
| 05-env-sampler-rewire | Env + episode sampler rewire: variable K, superset catalog, arbiter in step | 3 | committed | 5f96301 | 37941ed | 2026-06-10T13:46:00Z | 2026-06-10T16:40:00Z | 242 RL tests pass; eval.py compat updated |
| 06-train-loop-rewire | Train loop + rollout buffer rewire: masked PPO over padded shapes | 4 | committed | e220020 | e220020 | 2026-06-10T17:00:00Z | 2026-06-10T17:25:00Z | in-place mode; SetActor/SetCritic, masked PPO, checkpoint I/O; done/ move @ e991798 |
| 07-eval-rewire | Eval rewire: reduced CRN tuple, unchanged anchor, K-generalisation recipe | 5 | committed | 81aded6 | 81aded6 | 2026-06-10T17:30:00Z | 2026-06-10T17:50:00Z | in-place mode; 4 new tests; checkpoint.load() + SetActor; K-gen CLI flags; insufficient_cash warning |
| 08-documentation-sweep | Documentation sweep: RL reference, README, CONTEXT.md, ADR status | 6 | committed | a0cd6f3a36e2587011c5e39e1e1c713c710e2de3 | 976112e | 2026-06-10T17:55:00Z | 2026-06-10T18:05:00Z | docs-only; ADR 0021 accepted; CRN tuple / obs layout / CLI flags updated |
| 09-rl-notebooks-update | RL notebooks update: train/eval and trained-agent analysis on variable-K | 6 | committed | ec17246704be60911edbfca528e46bc6a64c3158 | 1b3970f | 2026-06-10T17:55:00Z | 2026-06-10T18:08:00Z | both notebooks executed with outputs; corrected flat-obs/SetActor API usage |

## Dependency graph

```
01 ─┬─► 02 ─┬─► 03 ─┐
    │       ├─► 04 ─┼─► 06 ─► 07 ─┬─► 08
    └───────┴─► 05 ─┘             └─► 09
```

Edges: 02←01 · 03←02 · 04←02 · 05←01,02 · 06←03,04,05 · 07←05,06 · 08←07 · 09←06,07

## Wave plan

1. **Wave 1** — 01-arbiter-module _(fable)_
2. **Wave 2** — 02-set-encoder-decoder _(fable)_
3. **Wave 3** — 03-masked-set-actor-critic _(fable)_, 04-checkpoint-io _(fable)_, 05-env-sampler-rewire _(fable)_
4. **Wave 4** — 06-train-loop-rewire _(fable)_
5. **Wave 5** — 07-eval-rewire _(fable)_
6. **Wave 6** — 08-documentation-sweep _(fable)_, 09-rl-notebooks-update _(fable)_

## Activity log

- 2026-06-10T13:31:05Z — Orchestrator: preflight + plan complete; initial report written; all 9 issues `pending`. BASE_BRANCH=RL-rewrite @ 3f7c09b.
- 2026-06-10T13:35:00Z — Wave 1: dispatching issue-implementer for 01-arbiter-module (in-place mode, model=fable).
- 2026-06-10T13:50:00Z — Wave 1: 01-arbiter-module committed @ 0974b62c18c034051aa1997ced07d90ebafb9798 (in-place, integrated directly on RL-rewrite). 29 tests added; full suite 1015 passed, 10 skipped.
- 2026-06-10T14:00:00Z — Wave 2: dispatching issue-implementer for 02-set-encoder-decoder (in-place mode, model=fable).
- 2026-06-10T14:20:00Z — Wave 2: 02-set-encoder-decoder committed @ 778a7e55abb960e18458bfb7232887dedcabe953 (in-place, integrated directly on RL-rewrite). 34 tests added; full suite 1049 passed, 10 skipped.
- 2026-06-10T13:46:00Z — Wave 3: dispatching issue-implementers for 03-masked-set-actor-critic, 04-checkpoint-io, 05-env-sampler-rewire in parallel worktrees (model=fable each).
- 2026-06-10T16:30:00Z — Wave 3: 03-masked-set-actor-critic committed in worktree @ b1a47edd; cherry-picked onto RL-rewrite @ f32bc9c. 21 tests (MaskedJointGaussian, SetActor, SetCritic). Worktree cleaned up.
- 2026-06-10T16:30:00Z — Wave 3: 04-checkpoint-io committed in worktree @ be11b425; cherry-picked onto RL-rewrite @ 42dc567. 13 tests (save/load/version-mismatch). Worktree cleaned up.
- 2026-06-10T16:40:00Z — Wave 3: 05-env-sampler-rewire committed in worktree @ 5f96301; cherry-picked onto RL-rewrite @ 37941ed. 242 RL tests pass, 1 skipped (pre-existing). Worktree cleaned up.
- 2026-06-10T16:45:00Z — Wave 3 complete. All worktrees and per-issue branches removed. RL-rewrite has 5 issues committed; 4 remain pending (waves 4-6).
- 2026-06-10T13:55:00Z — Orchestrator: post-wave-3 integration verification — full suite green (1085 passed, 9 skipped). Bookkeeping fix: 05-env-sampler-rewire's issue-implementer integrated its code (37941ed) but left the issue file in issues/ instead of issues/done/ — moved it to done/ with Status: done and committed @ 05bb22d to restore the done/⇔committed invariant.
- 2026-06-10T17:00:00Z — Wave 4: dispatching issue-implementer for 06-train-loop-rewire (in-place mode, model=fable).
- 2026-06-10T17:25:00Z — Wave 4: 06-train-loop-rewire committed @ e220020 (in-place, integrated on RL-rewrite); issue file moved to done/ @ e991798. Orchestrator verified full suite green (1089 passed, 9 skipped). Status row reconciled in→committed.
- 2026-06-10T17:30:00Z — Wave 5: dispatching issue-implementer for 07-eval-rewire (in-place mode, model=fable).
- 2026-06-10T17:50:00Z — Wave 5: 07-eval-rewire committed @ 81aded6 (in-place, integrated on RL-rewrite); issue file moved to done/. Full suite 1093 passed, 9 skipped. 4 new tests (stale-checkpoint rejection, K-generalisation config, insufficient_cash warning, layout-version mismatch). _load_policy_fn now uses checkpoint.load() + SetActor.
- 2026-06-10T17:55:00Z — Wave 6: dispatching issue-implementers for 08-documentation-sweep, 09-rl-notebooks-update in parallel worktrees (model=fable each).
- 2026-06-10T18:05:00Z — Wave 6: 08-documentation-sweep committed in worktree @ a0cd6f3; cherry-picked onto RL-rewrite @ 976112e. Docs-only: src/rl/README.md rewritten for variable-K; top-level README RL section updated; CONTEXT.md planned-annotations flipped; ADR 0021 proposed→accepted; TODO.md §9 PLANNED→IMPLEMENTED. Worktree cleaned up.
- 2026-06-10T18:08:00Z — Wave 6: 09-rl-notebooks-update committed in worktree @ ec17246; cherry-picked onto RL-rewrite @ 1b3970f. Both notebooks executed with outputs. Corrected obs handling (flat→2D reshape, mask extraction) and SetActor call (obs_2d + mask args). Worktree cleaned up.
- 2026-06-10T18:10:00Z — Wave 6 complete. All 9 issues committed. RL-rewrite tip @ 1b3970f. Feature variable-k-rl fully implemented.
- 2026-06-10T18:12:00Z — Orchestrator (Phase 5): bookkeeping fix — wave 5 had COPIED 07-eval-rewire.md into done/ rather than moving it, leaving a stale `ready-for-agent` duplicate at issues/07-eval-rewire.md in HEAD. Removed it (commit on RL-rewrite) so resume reconcile cannot re-dispatch issue 07. Final full suite green (1093 passed, 9 skipped). ADR 0021 status confirmed `accepted`. Stale-term grep on src/rl/README.md + CONTEXT.md returns only superseded/historical mentions (ADR provenance, "deleted slot-shuffle", "slot_permutation absent") — issue 08 criterion satisfied. Final cleanup: no .claude/worktrees/issue-* dirs, no variable-k-rl/issue-* branches remain; working tree clean except skill-owned report.

## Outstanding follow-ups

- 05-env-sampler-rewire: eval.py `_run_rl` / `_cold_start_qty_per_sku` minimally updated to use encode_set_observation/decode_set_action for shape compatibility with new env. Full eval rework deferred to issue 07 as planned — now complete.
- 07-eval-rewire: K-generalisation recipe is config-only (train K≤10, eval K=20 via K_min=K_max_episode=20); actually running the compute is explicitly out of scope (PRD).
- 09-rl-notebooks-update: notebooks use a 4,000-step smoke run so the RL agent loses to the baseline (expected for this budget); real training documented in src/rl/README.md.

## Resume instructions

Re-run `/implement-issues .scratch/variable-k-rl/` (same feature path). Phase 2 reconcile recovers state from disk + git: issues in `issues/done/` or with a `<id>:`-prefixed commit on RL-rewrite are treated as `committed`; stale/salvageable worktrees are swept before any new dispatch.
