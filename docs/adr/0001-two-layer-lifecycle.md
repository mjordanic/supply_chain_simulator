# Lifecycle is two-layer: global PLC × per-store freshness curve

Status: Superseded by ADR 0017 (lifecycle + freshness removed for the PoC; see `TODO.md`)

A product's effective demand multiplier comes from two independent sources composed multiplicatively: a **global lifecycle stage** (industry-wide PLC, owned by `ItemRegistry`, advances on `world_rng`) and a **per-(store, product) freshness curve** `m(τ) = 1 + α · exp(−τ / β)` (owned by `Store`, where `τ` is ticks since the SKU was last activated in that store).

We chose two-layer over global-only — which can't model per-store hype on introduction and forces CRN-paired stores to see identical lifecycle transitions even when their policies activate SKUs at different times — and over per-store-only, which loses the industry-wide signal of a category that's genuinely declining everywhere. Two-layer captures both effects with one extra per-store data structure (`activation_tick: dict[product_id, int]`) and one extra multiplicative factor in `Market.sample_demand`. The freshness curve resets on every `Store.activate_item` event, modelling "out-of-stock long enough that hype fades, reintroduction is fresh".

`α, β` live on `Ware` (per-product, LLM-authored per category). Realistic defaults: `α ∈ [0.1, 0.4]`, `β ∈ [15, 45]` ticks; `α = 0` for staples.
