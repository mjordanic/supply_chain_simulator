# Learned replenishment policy orders only; pricing is a later action space

Status: Accepted

The synthetic tuning episodes set `price_elasticity` to `-1.5`, so list price changes demand. M5 `flat_world` sets elasticity to `0`, but shop cash inflow is still price times units sold. `OrderUpToPolicy` writes `list_price` from `base_prices` and never uses that lever (ADR 0006). A learned `IntermediatePolicy` that also sets prices would beat or lose to that anchor for reasons that are not replenishment.

**Decision — The learned Policy in this study orders only.** `decide` returns `list_price[pid] = base_price[pid]`, emits no promotions, and does not learn `min_order_imposed`. On an M5 replay both the learned Policy and the textbook arm are wrapped in the same `PriceReplayPolicy`, so both charge the observed daily price. Version 1 attaches that Policy to one intermediate, the buyer facing the sinks, on the tuning graph and on an M5 shop with one synthetic factory grafted upstream. The factory stays on `StaticFactoryPolicy`.

**Future work — pricing policies.** The action space should grow to include list price once ordering results are in. That is a separate study: same ordering rule on both arms, or a pricing-only Policy against flat `base_price`, so the two levers are not confounded. ADR 0006 already left room for a pricing-only comparison. Do not fold pricing into this study's checkpoints, notebooks, or paper tables.

**Why not price now?** Elasticity on the synthetic arm and revenue on the replay arm both move profit when price moves. A gap against `OrderUpToPolicy` would then be uninterpretable. Supplier routing is likewise out of version 1: one upstream factory means there is no route to choose.
