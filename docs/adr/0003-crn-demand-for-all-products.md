# Demand is sampled for every catalog product each tick, even when inactive

`Runner._process_demand` calls `Market.sample_demand(pid, store, price)` once per `(store, product)` per tick — including products that are not in `store.active_items`. This is deliberate, not an oversight: the `world_rng` draw sequence must be independent of which subset of products is currently active so two CRN-paired stores (same `(template, init_seed)`, different policy) consume identical world stochasticity even when their policies activate or deactivate different SKUs over time. Without this property, the paired-comparison authoring pattern is meaningless.

Considered alternatives: (a) **skip inactive products** — simpler logs but breaks CRN cleanliness, which is the only reason paired comparison is more powerful than independent runs; (b) **pre-draw all demand into a buffer and discard inactive entries** — same CRN guarantee as today, adds a buffering layer for no real win. Cost of the current choice: "latent demand" is recorded in the run log for products no store carries. That data is currently unused but cheap to keep, and a future policy could read it ("how would this SKU sell if I introduced it?").

Do not "fix" this loop to skip inactive products without first replacing the CRN guarantee with an equivalent — e.g., dedicated per-product RNG streams seeded from `(world_seed, product_id)`.
