# allocation_rng wiring + shuffle_buyers + CentralTable EMA + determinism tests

Status: ready-for-agent

## Parent

`.scratch/multi-echelon/PRD.md`

## What to build

Activate the `allocation` sub-seed end-to-end so that the per-phase buyer shuffle is reproducible and orthogonal to the existing four sub-seeds. After this slice, supply contention is deterministic across runs with the same `world_seed`.

In `build_graph_world`: construct `allocation_rng = Random(_derive_seed(world_seed, "allocation"))` using the existing `_derive_seed(seed, purpose)` formula. Pass it into `GraphSimulation`.

Per phase: collect buyers at the current level, call `shuffle_buyers(buyers, allocation_rng)` once, then iterate in shuffled order.

Wire `CentralTable.commit` to update `fill_rate_recent` as a qty-weighted rolling EMA over a 10-tick window. (The EMA hooks were defined in issue 2; this slice exercises them from the live allocation path.)

Add `tests/sim/test_allocation_determinism.py`:
- Buyer-shuffle sequence is reproducible from `world_seed`
- Policy swap on one buyer does not perturb the shuffle (allocation_rng orthogonal to policy_rng)
- The third determinism invariant from issue 6's chain test set is now extended to cover `allocation_rng` explicitly

## Acceptance criteria

- [ ] `build_graph_world` constructs `allocation_rng = Random(_derive_seed(world_seed, "allocation"))`
- [ ] `src/sim/allocation.py` (or wherever appropriate) exports `shuffle_buyers(buyers, allocation_rng)` — single deterministic shuffle per phase
- [ ] `GraphSimulation.tick()` shuffles buyers at each level before iteration
- [ ] `CentralTable.commit` updates `fill_rate_recent` qty-weighted EMA (window 10) per `(seller_id, pid)`
- [ ] `tests/sim/test_allocation_determinism.py` covers: same `world_seed` → same shuffle sequence; policy swap on one buyer does not perturb the shuffle
- [ ] `tests/sim/test_graph_determinism_chain.py` updated to assert the third invariant explicitly against `allocation_rng`
- [ ] `uv run pytest tests/sim -x` is green

## Blocked by

- `.scratch/multi-echelon/issues/07-allocation-execute-buy-full.md`
