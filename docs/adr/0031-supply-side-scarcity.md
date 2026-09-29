# Supply-side scarcity on replenishment episodes

Status: Accepted

Tuning episodes used to give each SKU one `FactoryNode`. It produced four times the shop's capacity every tick, opened at twice that capacity, and listed at `unit_cost`. `delivery_lag` was 3 on every edge. Upstream stockouts did not happen, suppliers earned no margin, and fill rate did not move. Cheapest-first routing had nothing to choose.

**Decision — A replenishment episode from `src.tuning.episode.sample_episode` has scarce, differentiated suppliers. Routing stays a fixed setting.**

Each active SKU has three factories, `F_<pid>_0`, `F_<pid>_1`, and `F_<pid>_2`. Index 0 is the fastest. The nominal delivery lag is an integer from 2 through 6 unless `TuningConfig.delivery_lag` pins it. Edge lead times are that nominal lag, the lag plus 2, and the lag plus 4. Draws use the `delivery_lag` sub-seed.

`list_price` is `unit_cost * (1 + markup)`. The slowest supplier's markup is drawn from `[0.05, 0.25]`. Each step faster adds a premium drawn from `[0.10, 0.35]`. The buyer therefore pays more than production cost, and a shorter lead time costs more.

Production per tick and opening inventory are sized from that SKU's expected demand per tick, not from the shop's capacity. Expected demand per tick is the ware's `demand_mean`, or 5 when that field is absent (the midpoint of the historical `Uniform(2, 8)` sink). Aggregate episode demand for the SKU is that rate times the episode length. The SKU's total production rate is a factor in `[0.15, 1.25]` times the per-tick rate, so output over the episode is that factor times aggregate demand. Opening stock is `[0, 4]` days of the per-tick rate. Market and lifecycle shocks move realised demand around this expectation; they are not folded into the factory size. Both totals are split across the three suppliers by sampled positive weights. The `supply` sub-seed, mixed with the product id, drives the draws.

`Runner.run()["global"]["supplier_fill_rate"]` is factory units filled divided by factory units requested. It is `None` when the episode requested nothing from a factory. The quantity is realised by the run, so it depends on the buyer.

The learned buyer stays ordering-only (ADR 0024). It emits a quantity per SKU. The supplier split is cheapest-first with `per_supplier_min_order_floor` 0, the same fixed settings as the textbook anchor. This ADR does not make routing an action.

ADR 0013 rule 3 still describes an authored factory whose `list_price` equals `unit_cost`: production debits `unit_cost`, and the sale credits the same amount. A sampled factory may list above `unit_cost`. Production still debits `unit_cost`. The markup stays in the factory's cash. `StaticFactoryPolicy` and the no-policy produce step keep the node's current `list_price` instead of writing `unit_cost` over it.

The RL episode sampler is unchanged: one zero-margin factory per SKU, production and opening stock still multiples of node capacity. Tightness of the shop's capacity pool and opening cash is ADR 0032.

## Consequences

A fixed `episode_seed` still rebuilds the same graph. `world_seed` derivation is unchanged, and these draws do not consume `world_rng`. Episodes that pin `delivery_lag` keep that nominal lag and still differ across suppliers. The transition-log manifest stores `delivery_lag: null` when the lag was sampled. Each log row carries that episode's nominal lag as its own column. The encoder's sales window and arrival-age profile are fixed widths, not that nominal lag, so one fitted model reads a log whose lags differ (ADR 0024).

## Cross-references

- ADR 0011 — Factory nodes and edges are the existing graph types
- ADR 0012 — Partial fills are Allocation clamping the offer
- ADR 0013 — Rule 3 remains the zero-margin authoring default; this ADR amends it for sampled replenishment factories
- ADR 0024 — Routing stays outside the learned action
