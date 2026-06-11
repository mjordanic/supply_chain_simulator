# 04: Notebook 06a — fixed-path re-run, fresh analysis, attach-and-run demo

Status: ready-for-agent

## Parent

`.scratch/rl-eval-parity/PRD.md` (user stories 13, 14, 15)

## What to build

Update notebook 06a (analyze-a-trained-agent) for the fixed eval path and add the
deployment recipe:

- Remove the stale "the offline eval does not apply the Arbiter" caveats; replace with the
  parity statement (one shared code path; the Arbiter travels inside the policy at eval).
- Re-run every section that evaluates checkpoints. The proportional-vs-greedy comparison
  (section 5c) was a conservative estimate of the greedy checkpoint (priority head
  disconnected under the old path) — rewrite its analysis text against the fresh outputs,
  do not patch the old numbers.
- Add a new final section with a two-cell attach-and-run demo:
  1. In-distribution: attach the best proportional checkpoint to the training node via
     `RLNodePolicy.from_checkpoint(...)` + `policy_overrides` + `Runner.run()`, then analyse
     the run with the standard inspect/metrics plots.
  2. Out-of-distribution: attach the same checkpoint to a node in a deeper topology
     (notebook 03 gallery style) with an explicit caveat that this demonstrates the API,
     not transfer performance.
- Fully re-execute the notebook end-to-end; committed outputs must come from the new code.

## Acceptance criteria

- [ ] No remaining prose claims that eval skips the Arbiter
- [ ] All checkpoint-evaluating sections re-executed on the rebuilt eval harness
- [ ] Section 5c analysis rewritten against fresh post-fix numbers
- [ ] Two-cell attach demo present: in-distribution Runner + inspect tooling, then deeper-topology attach with an OOD caveat
- [ ] Notebook executes end-to-end cleanly; all cell outputs are fresh

## Prior art

- Notebook 03 (scenario gallery) — source of the deeper-topology style for the OOD cell
- Notebook re-execution is the repo's established verification pattern for notebooks — no unit tests
- Existing checkpoints in the proportional and greedy fashion-run studies are re-evaluated, not retrained

## Blocked by

- 03-eval-harness-rebuild
