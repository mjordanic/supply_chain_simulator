# 03: Rebuild the offline eval harness on RLNodePolicy + Runner

Status: ready-for-agent

## Parent

`.scratch/rl-eval-parity/PRD.md` (user stories 1, 3, 8, 9)

## What to build

Replace the eval harness's bespoke ~200-line RL episode loop (and its hand-rolled per-tick
log harvesting) with a standard `Runner.run()` using `policy_overrides` and an
`RLNodePolicy` wrapping the supplied policy callable. After this, both arms of every
CRN pair run through the identical engine code path, and the Arbiter is applied at eval
exactly as in training — greedy checkpoints get their priority head connected.

Fixed by the PRD:

- **`evaluate()` public signature unchanged** (RL policy callable + baseline factory +
  specs + config); it wraps the callable in a fresh `RLNodePolicy` per spec. The training
  loop's in-training eval closure keeps sampling stochastically and needs no changes.
- The eval harness computes `base_demand_prior` from the episode spec exactly as the old
  loop did, and passes it to the policy constructor.
- The cold-start order-quantity probe is rebuilt on the same path; it now measures
  post-arbiter allocations — the real order — rather than raw proposals (semantics change
  to be noted in ADR 0022, issue 06).
- The insufficient-cash warning behaviour for the RL node is preserved.
- The RL arm's run log flows through the same inspect/metrics builders as the baseline arm.

## Acceptance criteria

- [ ] The bespoke RL episode loop and its per-tick log harvesting are deleted; the RL arm runs via `Runner.run()` + `policy_overrides`
- [ ] `evaluate()` signature is unchanged and the training loop's in-training eval works without modification
- [ ] Existing eval-harness tests stay green (CRN pairing, metric keys, two-scale eval)
- [ ] New regression test: the RL arm's run log flows through the standard inspect path
- [ ] Cold-start probe rebuilt on the standard path, measuring post-arbiter allocations
- [ ] Insufficient-cash warning behaviour preserved

## Prior art

- The existing eval tests — the CRN-paired metric contract that must survive the rebuild
- `build_eval_seeds` and the CRN-pairing convention — seed handling must not change
- The baseline arm of the eval harness — already uses `Runner.run()`; the RL arm should end up symmetrical to it

## Blocked by

- 02-rlnodepolicy
