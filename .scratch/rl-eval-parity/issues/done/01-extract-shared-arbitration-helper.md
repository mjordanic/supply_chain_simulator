# 01: Extract shared arbitration helper

Status: done

## Parent

`.scratch/rl-eval-parity/PRD.md` (user stories 17, 21)

## What to build

Lift the arbiter block currently inlined in the training env's step function into one
shared function in the arbiter module, and make the training env call it. This is a
behaviour-preserving extraction — training semantics must not change.

The new function takes the decoded per-product order proposals, a snapshot of the node's
state (inventory, pending, capacity, cash), per-product unit prices, per-product
priorities, and the RL config; it computes per-SKU headroom, global free space, and the
cash budget, calls the existing pure `allocate()` with the configured arbiter mode, and
returns the final arbitrated order dict (each surviving product routed as a single order
line, as the env does today). The existing pure `allocate()` stays untouched underneath.

This function is the single code path for arbitration dynamics: the training env calls it
now, and `RLNodePolicy` (issue 02) will call it next.

## Acceptance criteria

- [ ] The arbiter projection logic exists exactly once, as a public function in the arbiter module
- [ ] The training env's step function calls the shared function instead of inlining the logic
- [ ] The pure `allocate()` function is unchanged
- [ ] The entire existing RL test suite passes unchanged (arbiter unit tests, env step tests) — this is the behaviour-preservation proof
- [ ] No change to training dynamics, action layout, encoder features, or checkpoint format

## Prior art

- Arbiter unit tests (proportional/greedy allocation semantics) — the contract the extraction must preserve
- The env step tests — guard the end-to-end tick pipeline around the extracted block
- ADR 0021 Decision 4 — defines the Arbiter as environment dynamics

## Blocked by

None - can start immediately
