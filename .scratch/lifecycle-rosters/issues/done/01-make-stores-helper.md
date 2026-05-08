# Replace `homogeneous` / `paired` with `make_stores`

Status: done

## Parent

`.scratch/lifecycle-rosters/PRD.md`

## What to build

Replace the two store-roster authoring helpers (`homogeneous`, `paired`) with one declarative helper `make_stores(triples)` where `triples: list[tuple[StoreTemplate, int, Policy]]`. CRN comparison and robustness sweeps are expressed by repeating or varying triples in the literal list — there is no regime abstraction above the list. The two old helpers are deleted (no deprecation wrappers); both example scenarios and any test callsites are migrated. Step-0 bit-identity for shared `(template, init_seed)` is preserved across attached policies.

A k-way (k ≥ 3) CRN comparison or a mixed roster like "policies A, A, B, C, D across templates T1, T2, T3, T4, T4, T4" is just a literal triple list — no helper composition required.

## Acceptance criteria

- [ ] `make_stores(triples) -> list[StoreInstance]` exists in `src/sim/scenario.py`
- [ ] `homogeneous` and `paired` removed from `src/sim/scenario.py` (no deprecation wrappers)
- [ ] `scenarios/example_homogeneous.py` migrated to `make_stores` with three triples sharing one policy and varying `init_seed` — same numerical scenario as before
- [ ] `scenarios/example_paired_comparison.py` migrated to `make_stores` with paired triples (same `(template, init_seed)`, two policies) — same numerical scenario as before
- [ ] All existing CRN/determinism tests in `tests/sim/test_determinism.py` and `tests/sim/test_regression_snapshot.py` pass unchanged
- [ ] `tests/test_examples_and_cli.py` passes unchanged (both example scenarios run end-to-end)
- [ ] A new test pins that a k-way (k ≥ 3) CRN comparison can be expressed with `make_stores` as a literal triple list — three triples with the same `(template, init_seed)` and three different policies produce three stores with bit-identical step-0 state

## Testing notes

Drive the migration via TDD vertical slices. Don't bulk-rewrite the helpers and then update tests; migrate one example/test pair at a time. Tests should pin observable behavior (resulting `StoreInstance` list shape; bit-identical step-0 state across triples sharing `(template, init_seed)`) — not the implementation of the helper.

## Blocked by

None - can start immediately
