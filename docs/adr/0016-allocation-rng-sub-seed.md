# RNG/CRN extension: the `allocation` sub-seed stream

Status: Proposed

The existing CRN seeding contract (ADR 0003) fans one `episode_seed` into four independent sub-seeds (`assortment`, `capacity`, `balance`, `world`) via a multiply-add-mask formula in `src/sim/episode_sampler._derive_seed`. The `world_seed` sub-seed drives the shared `world_rng` consumed by `Market`, `EventEngine`, `ItemRegistry`, and per-store demand sampling.

The multi-echelon tick cascade (ADR 0014) introduces a new source of stochasticity: the per-phase buyer shuffle. Within each phase of the cascade, buyers at a given echelon level are shuffled before executing their allocations sequentially. This shuffle determines which buyer gets first access to limited inventory, directly affecting fill rates when supply is scarce. The shuffle must be (a) reproducible from the world seed so paired CRN comparisons remain bit-identical, and (b) orthogonal to `world_rng` so a policy swap on one buyer does not perturb the market demand stream.

**Decision — Add a fifth sub-seed `"allocation"` with parameters `(0xA24B_AED4, 0x0000_0007)` to `_SUB_SEED_PARAMS`, derived via the existing `_derive_seed(episode_seed, "allocation")` formula. Construct `allocation_rng = Random(_derive_seed(world_seed, "allocation"))` in `build_graph_world` and consume it exclusively in `shuffle_buyers`.**

**Sub-seed parameters.** The `(PRIME, OFFSET)` pair `(0xA24B_AED4, 0x0000_0007)` was chosen to be distinct from all four existing pairs. The offset `0x0000_0007` follows the existing sequence (1–4 for the four original sub-seeds, 6 for `init_stock`, 7 for `allocation`), preserving the convention that each purpose has a unique scalar offset. The prime ensures that `_derive_seed(seed, "allocation")` produces a different value from all other sub-purposes for any non-degenerate `episode_seed`.

**Three determinism invariants that must hold after this change:**

1. **World stream identity.** Same `world_seed` yields identical market, event, lifecycle, and buyer-shuffle sequences across runs. This holds because `allocation_rng` is seeded from `world_seed` via `_derive_seed` — a deterministic function — and `shuffle_buyers` consumes `allocation_rng` in the same order (echelon level by level, phase by phase) for every run with the same `world_seed`.

2. **Step-0 node state independent of policy.** Same `(node_template, init_seed)` yields identical step-0 node state regardless of attached policy. This holds because node initialisation consumes only `init_rng = Random(init_seed)` and never `world_rng` or `allocation_rng`.

3. **Policy swap does not perturb world or allocation streams.** Replacing the policy on one node does not change how `world_rng` or `allocation_rng` are consumed. This holds because `allocation_rng` is constructed before any nodes execute and consumed only by `shuffle_buyers`, which depends on the set of buyers at each level (fixed by topology) but not on the decisions those buyers make. A policy swap changes `decide` output but not which buyers exist or in what shuffled order they execute.

**Where `allocation_rng` is not consumed.** The `allocation_rng` stream is dedicated to buyer shuffle. It is never consumed by market draws, event sampling, lifecycle transitions, or policy RNGs. Each `NodePolicy` that needs randomness uses its own `policy_rng = Random(policy_seed)`. The separation is structurally enforced: `allocation_rng` is not threaded into `build_world` or `Simulation.tick_decide_and_settle` (legacy path); it appears only in `build_graph_world` and `GraphSimulation.tick`.

**Existing four sub-seeds are unchanged.** Adding `"allocation"` does not alter the derivation or consumption of `assortment`, `capacity`, `balance`, or `world` sub-seeds. Existing episode samplers, tests, and CRN-paired evaluators that do not use the graph engine are unaffected.

## Cross-references

- ADR 0012 — Central table + FCFS allocation (`shuffle_buyers` is called at the start of each phase by the allocator; `allocation_rng` is the randomness source)
- ADR 0014 — Tick phasing (buyer shuffle happens at the start of each level-`p` phase of the cascade)
- ADR 0003 — CRN demand for all products (the original seeding contract this ADR extends; the four existing sub-seeds are unchanged)
