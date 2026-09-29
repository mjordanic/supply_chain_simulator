# Learned replenishment policy orders only; pricing is a later action space

Status: Accepted

The synthetic tuning episodes set `price_elasticity` to `-1.5`, so list price changes demand. M5 `flat_world` sets elasticity to `0`, but shop cash inflow is still price times units sold. `OrderUpToPolicy` writes `list_price` from `base_prices` and never uses that lever (ADR 0006). A learned `IntermediatePolicy` that also sets prices would beat or lose to that anchor for reasons that are not replenishment.

**Decision — The learned Policy in this study orders only.** `decide` returns `list_price[pid] = base_price[pid]`, emits no promotions, and does not learn `min_order_imposed`. On an M5 replay both the learned Policy and the textbook arm are wrapped in the same `PriceReplayPolicy`, so both charge the observed daily price. Version 1 attaches that Policy to one intermediate, the buyer facing the sinks, on the tuning graph and on an M5 shop with one synthetic factory grafted upstream. The factory stays on `StaticFactoryPolicy`.

**Future work — pricing policies.** The action space should grow to include list price once ordering results are in. That is a separate study: same ordering rule on both arms, or a pricing-only Policy against flat `base_price`, so the two levers are not confounded. ADR 0006 already left room for a pricing-only comparison. Do not fold pricing into this study's checkpoints, notebooks, or paper tables.

**Why not price now?** Elasticity on the synthetic arm and revenue on the replay arm both move profit when price moves. A gap against `OrderUpToPolicy` would then be uninterpretable.

**Routing stays outside the learned action.** The buyer emits a quantity per SKU. The split across suppliers is the same fixed strategy as the textbook anchor (cheapest-first, minimum-order floor 0). A learned route would confound a replenishment result with a sourcing result. ADR 0031 gives each SKU several suppliers and still does not make routing an action.

**What the encoder may see.** On-hand inventory. The in-transit arrival-age profile — units of each age still on the way, which is the buyer's own pipeline. A sales window whose length is its own knob, set longer than the delivery lead time, so a demand level can show up in sales. Cash and capacity. The nominal delivery lead time, the supply contract `TextbookReorderPolicy` already reads from its observation. Lead time and pipeline age are contract terms and self-knowledge, not latent world state. The feature width does not follow the sampled lead time, so one fitted model reads every episode.

**What the encoder may not see.** The Market multiplier, the disruption identity, and the calendar (simulation tick, date, or season index). Those stay hidden. The study's premise is that the buyer infers the regime from the sales window rather than being handed the latent state.
