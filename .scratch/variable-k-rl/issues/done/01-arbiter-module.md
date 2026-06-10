# Arbiter module: deterministic reconciliation with proportional and greedy variants

Status: done

## Parent

`.scratch/variable-k-rl/PRD.md` (governing decision record: ADR 0021, Decision 4)

## What to build

A new pure, torch-free Arbiter module in the RL layer that projects a joint per-product order
proposal onto the feasible set defined by per-SKU capacity headroom, global free space, and a
cash budget. It is the single place within-tick resource contention is resolved, replacing the
engine's invisible pid-iteration-order cash starvation with a deterministic, learnable rule.

Interface (per the PRD): a pure function taking per-product proposed quantities, per-SKU and
global capacity headroom, a cash budget, per-product unit prices (from central-table offers),
optional per-product priority scalars, and a mode flag — returning integer allocations per
product.

Two variants behind the mode flag:

- **Proportional fair-share** (default): scale every proposal by the binding feasibility ratio,
  extending today's two-pass fair-share allocator semantics (per-SKU headroom pass, then
  proportional global scale) with the cash budget as a third potential binding resource.
- **Priority greedy**: fill proposals in descending priority order until a resource (per-SKU
  headroom, free space, or cash) is exhausted; can starve low-priority products entirely.

The module absorbs the existing fair-share allocator: its semantics become the proportional
mode, and the old standalone function is retired in the env/sampler rewire issue (do not delete
it here — the env still calls it until issue 05).

The cash budget itself is computed by the caller (node cash × configured fraction, costed at
central-table offer prices); the Arbiter just enforces the number it is given.

## Acceptance criteria

- [ ] Pure module: no torch import, no engine/simulation import; testable with plain dicts/floats.
- [ ] Allocations never exceed per-SKU headroom, global free space, or the cash budget (property tests).
- [ ] Proportional mode preserves proposal ratios under a binding global constraint.
- [ ] Greedy mode fills in strict descending priority order and starves the lowest priority first when a resource binds.
- [ ] Integer outputs; zero-proposal and zero-budget edge cases return all-zero allocations without error.
- [ ] K-free: identical code path works for K = 1 and K = 32.
- [ ] Dedicated test module for the Arbiter (this is one of the three pure-core modules with dedicated tests per the PRD).
- [ ] Existing fair-share allocator and its tests untouched (retirement happens in issue 05).

Prior art: the two-pass fair-share semantics are exercised in `test_textbook_helpers`; encoder
test style in `test_encoders`.

## Blocked by

None - can start immediately
