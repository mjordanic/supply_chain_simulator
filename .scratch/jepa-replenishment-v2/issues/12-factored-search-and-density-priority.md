# 12: Factored search and density-ordered Ration priority

**What to build:** A proposal search that survives a large assortment, and a
Ration priority that reflects the structure of the problem.

Today `decide` draws 64 proposals, each a cover per SKU plus a random priority
permutation. With `K_active` now sampled up to roughly 30, that is 64 random
draws over a cover grid of size four across thirty SKUs, times thirty factorial
permutations. That is not a search.

With a shared capacity pool the allocation is a knapsack, so the priority order
should be *computed* from marginal profit per unit of the constrained resource
rather than sampled. The random-permutation arm is kept because ADR 0025 asks
whether the priority Ration beats textbook fair-share, and that question needs
both arms to stay answerable.

**Blocked by:** 03, 07.

**Status:** ready-for-agent

**Prior art:** existing planner tests; the `Ration` and
`Latent replenishment planner` entries in `CONTEXT.md`; ADR 0025 (priority
ration) for the question the ablation arms exist to answer.

- [ ] The proposal search is factored — cross-entropy over per-SKU covers rather
      than joint random draws — and its cost scales acceptably as `K_active`
      grows.
- [ ] Ration priority is computed from marginal imagined profit per unit of the
      shared capacity pool.
- [ ] A random-permutation priority arm is retained as an ablation.
- [ ] The textbook fair-share arm is retained, so the ADR 0025 comparison stays
      three-way.
- [ ] A test asserts the search beats uniform random sampling from the same
      action set — the check that would have caught the original failure, where
      the planner lost to random covers by 8×.
- [ ] The planner conditions its imagined transition on a fill expectation rather
      than assuming every order arrives whole, now that the log carries filled
      quantities.
- [ ] Any notebook, README section or `CONTEXT.md` text invalidated by the new
      search is updated in this ticket.
