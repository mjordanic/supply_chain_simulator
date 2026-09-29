# 05: Episode distribution, demand side

**What to build:** Episodes whose demand is nonstationary *per SKU* and whose
assortment size varies, so that the textbook censored-sales estimator has
something to lag behind and the shared capacity pool has a varying number of
claimants.

Today the item lifecycle is inert: stage-change probabilities are all zero and
every SKU starts at `maturity`, so the `stage_multipliers` never fire. The only
nonstationarity is the shared Market wave, which is identical across SKUs and
cannot create cross-SKU tension. The catalog is 100 synthetic SKUs whose mean
demand spans only 1.9×, and `K_active` is pinned at 5.

**Blocked by:** 01.

**Status:** ready-for-agent

**Prior art:** existing tuning episode tests; the `Episode sampler`,
`Active subset`, `Item / Ware / SKU` and `Distribution` entries in `CONTEXT.md`;
ADR 0001 (two-layer lifecycle) and ADR 0002 (per-stage transitions and dead
stage) for the lifecycle semantics this ticket switches on.

- [ ] Item lifecycle stage-change probabilities and per-SKU initial stages are
      sampled per episode rather than left at zero and `maturity`.
- [ ] Lifecycle transitions are drawn per SKU from the episode's own sub-seed, so
      SKUs enter, grow, mature, decline and die on independent schedules.
- [ ] A test asserts lifecycle stages actually move over an episode and that two
      SKUs in the same episode follow different stage paths.
- [ ] The synthetic catalog grows to roughly 500–1000 SKUs that are genuinely
      heterogeneous in demand scale, margin, volatility and lifecycle behaviour;
      the realised spread is asserted, not assumed.
- [ ] `K_active` is sampled per episode across a wide range rather than pinned.
- [ ] Episode sampling stays pure and deterministic in its seed.
- [ ] Any notebook, README section or `CONTEXT.md` text invalidated by the larger
      catalog or varying assortment is updated in this ticket.
