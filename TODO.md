# TODO — features removed during the setup-files simplification

This file tracks functionality intentionally **removed** while transitioning to the
two-stage *prepare-data / run-simulation* design (see `docs/adr/0017-setup-files-as-deterministic-input.md`).
Each section is enough to put the feature back later if a PoC shows we want it.

The guiding rule for the PoC: keep the readable core (graph of typed nodes, Market,
disruptions, textbook + RL policies, tuning) and strip the demand-shaping add-on layers
that multiplied onto demand but weren't load-bearing for a first working version.

---

## 1. Per-(node, product) freshness curve  — REMOVED

**What it did.** A demand multiplier `m(τ) = 1 + α·exp(−τ/β)` where `τ` = ticks since a
product was activated in a node. Modelled the "grand-opening hype" decay for newly
stocked SKUs. Composed multiplicatively into `DemandSinkNode.demand_target`.

**Code removed.**
- `src/sim/freshness_curve.py` (whole file)
- `Ware.freshness_alpha`, `Ware.freshness_decay` fields + their CSV columns
- `StoreTemplate.init_freshness` ("baseline" vs "fresh" step-0 regime) — moot once stores go
- `DemandSinkNode.activation_tick` bookkeeping and the freshness term in `demand_target`
- LLM `WorldBuilder` freshness stage (chunked per-Ware α/β authoring) + `freshness_prompt`
  + `_sanitise_freshness` + freshness schema
- `tests/sim/test_freshness_curve.py`, `test_freshness_integration.py`, `test_demand_sink_freshness.py`

**To re-add.** Reintroduce the multiplier in `demand_target` and re-add the two per-product
columns to `catalog.csv`. ADR 0001 has the original rationale (two-layer lifecycle).

---

## 2. Global product life-cycle (PLC stages)  — REMOVED

**What it did.** A global stage per SKU in `[introduction, growth, maturity, decline, dead]`,
each mapping to a `stage_multipliers` factor on baseline demand; stochastic transitions
governed by a per-stage `stage_change_probs` table; `dead` = trickle demand. Owned by
`ItemRegistry`, advanced from `world_rng` each tick.

**Code removed.**
- `src/sim/lifecycle_clock.py` (whole file)
- Lifecycle responsibility of `src/sim/item_registry.py` — `ItemRegistry` removed entirely;
  its **only surviving job** (the CRN per-catalog-product `world_rng` draw loop, ADR 0003)
  moves into the runner iterating the catalog list directly
- `ItemLifecycleParams` dataclass + the `item_lifecycle` Scenario field + `lifecycle_df()`
- `MarketParams.stage_multipliers` field
- the `stage_multiplier` term in `DemandSinkNode.demand_target`
- `tests/sim/test_lifecycle_clock.py`, `test_item_registry.py`

**To re-add.** Restore `ItemRegistry`'s stage ownership + `lifecycle_clock`, re-add the
`lifecycle:` block to `setup.yaml`, and reinstate the `stage_multiplier` term. ADR 0002 has
the original rationale (per-stage transitions + terminal `dead` stage).

---

## 3. Per-product lifecycle/freshness overrides  — REMOVED (collapsed to global)

**What it did.** A `Ware` could override the global defaults: `init_stage`,
`stage_change_probs`. The resolution path in `item_registry` let per-`Ware` values win over
`ItemLifecycleParams` defaults.

**Decision.** Never used in any example scenario. Even if lifecycle returns (section 2),
keep it **global-only** unless a concrete need for heterogeneous per-product dynamics appears.
`init_stock_share` (per-product initial-stock weight) is retained as a `catalog.csv` column
because the generator authors it per product.

**To re-add.** Add `prob_<stage>` columns + an `init_stage` column to `catalog.csv` and
restore `_resolve_stage_change_probs` override logic in `item_registry`.

---

## 4. Notes on things deliberately KEPT (so we don't re-cut them by mistake)

- **Market stays intact** minus `stage_multipliers`: seasonal cycle, regional demand/supply
  state, trend drift, cross-product correlation (`cross_inv_*`, `cross_factor_range`,
  driven by `Ware.related_products`), price elasticity, promo multiplier, supply side.
- **Disruption events** (natural disaster / economic crisis / pandemic / political unrest /
  technological breakthrough) stay — they're the headline supply-chain feature.
- **CRN per-catalog-product demand draw** (ADR 0003) stays — load-bearing for RL/tuning
  paired evaluation. It just no longer lives inside `ItemRegistry`.
