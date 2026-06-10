# Eval rewire: reduced CRN tuple, unchanged anchor, K-generalisation recipe

Status: done

## Parent

`.scratch/variable-k-rl/PRD.md` (governing decision record: ADR 0021, Decisions 1, 3)

## What to build

Rework CRN-paired evaluation onto the variable-K stack while keeping the protocol comparable to
prior results.

- **CRN tuple** shrinks to `(world_seed, capacity, balance, active_subset, allocation_sub_seed)`
  — `slot_permutation` is gone (the spec field was deleted in issue 05; this issue completes the
  eval-side rework). Paired determinism holds: the RL policy and the `OrderUpToPolicy` anchor
  see bit-identical worlds for each tuple.
- **Anchor unchanged**: `OrderUpToPolicy` needs no modification (a consequence of the no-churn
  decision); it manages the same episode product set per tuple.
- **Checkpoint loading** goes through the checkpoint I/O module (issue 04), so eval fails loudly
  on a layout-version mismatch.
- **K-generalisation recipe** ships as configuration plus a short usage note: train on K ≤ 10,
  evaluate on K = 20 — the recipe is deliverable; running the compute is follow-up work, out of
  scope.
- Opportunistic invariant during eval runs: engine rejection-log entries with reason
  `insufficient_cash` for the trainable node indicate an Arbiter bug — surface them in eval
  output (a warning is enough; no dedicated test suite).

## Acceptance criteria

- [ ] CRN-paired eval runs end-to-end on a checkpoint from the new train loop, producing paired uplift numbers vs `OrderUpToPolicy`.
- [ ] Same tuple twice ⇒ bit-identical paired results (paired-determinism property preserved on the reduced tuple).
- [ ] `OrderUpToPolicy` source untouched.
- [ ] Eval refuses a stale checkpoint with the descriptive layout-version error.
- [ ] K-generalisation eval is expressible purely via existing config knobs (K-range for eval episodes), with a usage note where eval options are documented.
- [ ] Existing eval CRN and eval-refactor tests updated to the reduced tuple (in scope per the PRD); no new dedicated eval test suite.
- [ ] Full test suite green (`uv run pytest`).

Prior art: `test_eval_crn` for paired-determinism assertions; `test_eval_refactor`,
`test_two_scale_eval_smoke` for the eval surfaces being updated.

## Blocked by

- `05-env-sampler-rewire.md`
- `06-train-loop-rewire.md`
