# 06: Docs — ADR 0022, RL README attach section, CONTEXT.md Arbiter entry

Status: done

## Parent

`.scratch/rl-eval-parity/PRD.md` (user stories 18, 19, 20)

## What to build

Documentation for the new deployment path and the decisions behind it:

- **ADR 0022** (in the repo's `docs/adr/` sequence): "RL policy as a first-class
  IntermediatePolicy; eval runs through the standard Runner; the Arbiter travels with the
  policy." Must record:
  - the trade-off taken (one shared code path over a bespoke eval loop, making train/eval
    parity structural);
  - the eval-history discontinuity — pre-fix TensorBoard `eval/*` curves in existing run
    directories were logged by the old arbiter-less path and will not match fresh evals of
    the same checkpoints; they stay as recorded, no backfill;
  - the cold-start-probe semantics change — it now measures post-arbiter allocations (the
    real order) rather than raw proposals.
- **RL README**: add an attach section documenting checkpoint deployment —
  `RLNodePolicy.from_checkpoint(...)` + `policy_overrides` + `Runner.run()`, the
  deterministic default, and the single-instance-per-run convention — so the path is
  discoverable without reading notebook source.
- **CONTEXT.md**: extend the Arbiter glossary entry with the two-homes clarification — the
  Arbiter is environment dynamics in training (per ADR 0021) and travels inside the policy
  at eval/deployment time — so the design reads as intended rather than accidental.

## Acceptance criteria

- [ ] ADR 0022 exists, follows the existing ADR format, and covers the trade-off, the eval-history discontinuity, and the cold-start-probe change
- [ ] RL README documents the attach recipe and its conventions
- [ ] CONTEXT.md Arbiter entry states the two-homes design and cross-references ADR 0021/0022
- [ ] No documentation claim contradicts the rebuilt eval harness or the notebooks

## Prior art

- ADR 0021 — the Arbiter-as-environment-dynamics decision this ADR builds on; match its format
- The CONTEXT.md Arbiter glossary entry — extend, do not rewrite

## Blocked by

- 03-eval-harness-rebuild
