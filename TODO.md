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

---

## 5. Tick phasing redesign — allow lateral links  — PLANNED

Replace the upward echelon cascade (ADR 0014) with a scheduling model that supports
**lateral supplier links** (same-echelon shop→shop, warehouse→warehouse, etc.) and
relaxed graph validation: drop the BFS same-level check in `validate_dag` and permit any
directed edge in the DAG **except factory→demand-sink**, which bypasses intermediates and
breaks the multi-echelon economics the simulator is built around. The current cascade assigns
one phase per longest-path echelon level and shuffles same-level buyers in parallel — that
breaks when a buyer depends on a same-phase supplier (order depends on shuffle) or when
shortcuts make BFS and longest-path disagree on what counts as a peer link. A new ADR should
pick the replacement mechanism (e.g. simultaneous clearing within a tick, topological phases
over the full edge set, or multi-pass settle) and define what each buyer observes before
deciding; until then, `build_graph` and `Simulation.tick` remain on the tiered-cascade
contract.

---

## 6. Replace the per-catalog CRN draw loop with per-stream RNG  — PLANNED

`DemandSinkNode.demand_target` loops over the **whole catalog** every tick, drawing one
`world_rng` sample per product but returning only the draw for `self.product_id` (the sink's
single bound product). The loop exists to keep CRN alignment on a single shared `world_rng`
stream: a single stream is position-sensitive, so the number of draws each sink makes must be
constant — a pure function of `(catalog size, tick)` — regardless of binding or active set, or
two paired runs (same scenario, different policy) desync and the comparison is contaminated
(ADR 0003). This was load-bearing in the old `Store` model where a store's *policy* could
activate/deactivate SKUs, making draw count vary by policy. In the multi-echelon model a sink
is bound to one fixed `product_id` set in the scenario (never policy-dependent), so that
original justification has largely evaporated — the loop now survives mostly as inherited
ceremony plus shared-stream coupling, and it depends on a fragile quirk (`Constant` skips its
draw, silently breaking alignment — see the docstring caveat).

**Better alternative.** Independent per-stream RNG seeded from `(world_seed, key)` (key =
`product_id` or `sink.id`), as ADR 0003 itself names. Each sink draws **one** sample from its
own stream: no loop, draw count/binding of one sink can't affect any other, and CRN holds by
construction (a stream's position depends only on its own key + tick, never on global ordering,
active sets, or the `Constant` quirk). Robust to adding/removing/reordering sinks and to future
multi-product changes.

**Tradeoffs.** Behavior-changing (RNG seeding moves ⇒ recorded runs and golden tests shift,
needs a re-baseline); loses the "latent demand for all products" side-data the loop emits
(ADR 0003 notes it is currently unused). Worth doing when next touching the CRN layer; not
worth standalone churn while current paired-eval tests are green. A new ADR should supersede
ADR 0003 and define the seeding scheme.

---


## 7. Run a trained RL policy inside a multi-echelon graph  — PLANNED

Both tuning and RL optimise a single `IntermediateNode` ("S") in a degenerate 3-node graph
(`factory → S → sink`; see ADR 0004 and ADR 0009). A *tuned* textbook policy is a plain
`IntermediatePolicy`, so deploying it into a real multi-echelon graph is trivial — attach it to
any one node via `Runner(scenario, policy_overrides={"shop-1": tuned}).run()` (demonstrated in
`notebooks/05-tune-a-policy.ipynb`). A *trained RL actor* is not portable that way yet.

**Why it doesn't work today.** `RLIntermediatePolicy` is a shim: its `decide()` just returns an
action pre-decoded and injected via `set_pending_action()`, which only the env's two-phase loop
(`tick_world` → `encode_observation` → actor → `set_pending_action` → `tick_decide_and_settle`)
ever calls. A normal node's `decide(obs_intermediate, central_table)` is *not* handed `market` or
`item_registry`, but `encode_observation` needs them (seasonality + lifecycle features, obs slots
6–13). So a checkpoint cannot be dropped onto a node and run through `Runner.run()` the way a
textbook policy can. The RL eval path (`src/rl/eval._run_rl`) sidesteps this by driving the
*whole* graph with the two-phase API itself — but it is hardwired to the 3-node `sample_episode`
graph.

**To re-add.** Two options. (a) A self-contained inference policy (`src/rl/inference.py`) that
holds references to `market` / `registry` and the per-node bookkeeping (`slot_perm`, rolling
`sales_history`, `active_subset`, `supplier_ids_for`, `initial_cash`) and runs
encode→actor→decode inside `decide()` — attachable like any `IntermediatePolicy`, but it must
reach world state the current `decide()` contract doesn't pass (so likely also a small engine
change to forward `market`/`registry`). (b) A generic two-phase driver that steps an arbitrary
multi-echelon graph and calls `set_pending_action()` on one chosen RL node while the other nodes
run their own policies — generalising `_run_rl` to any graph, no `decide()`-contract change.
Either way the RL node must carry exactly `K_active` SKUs (the actor's fixed input width). Add an
example to `notebooks/06-rl-train-and-eval.ipynb` once shipped, and update `src/rl/README.md`
(which currently points here).


---

## 8. Price-elastic demand and store competition  — PLANNED

Make price actually affect demand. Today it does nothing on the demand side: textbook policies
emit `list_price == base_price` flat, and `MarketParams.price_elasticity` / `promo_multiplier`
are parsed but **never read** by the graph engine — `demand_target` is only `demand_dist ×
seasonal × regional` (`node.py:298`, `market.py:305`). All of the below is synthetic and does
**not** depend on real data.

Two price roles: per-product **reference** (`base_price`, the yardstick) and per-(store,product)
**selling** (`list_prices`, the lever). They must be able to diverge — demand responds to the
ratio. Four things to build:

- **Pricing lever.** A policy that sets `selling` off `base` (markup / promo discount). Today no
  graph node prices off base, so there's nothing to react to.
- **Category elasticity (sink-only).** Add a factor to `demand_target`:
  `demand × (p_market / base) ** elasticity`. Sink-only — mid-graph nodes have no price-elastic
  demand (their order qty is inventory-policy-driven). `p_market = min(list_price)` over the
  **sink's direct suppliers only** (the sink already filters to direct suppliers, `runner.py:597`).
  All stores cheap → pool grows. `promo_multiplier` rides the same ratio.
- **Soft-share competition (sink-only).** Substitution already exists as hard cheapest-first at
  every buyer (`_split_across_suppliers`, `_default_sink_action` `runner.py:603`) — winner-take-all
  up to capacity, so a tiny price gap swings 100% of share. Replace the **sink** allocator with a
  soft-share (logit/attraction) split so a cheaper store *gradually* attracts more orders. Leave
  intermediate sourcing cheapest-first (a warehouse isn't a shopper).
- The two effects reinforce: a store cutting price both grows the pool and wins more share.

Wiring is small — one factor in `demand_target`, thread the faced `p_market` in (offers are
published before sinks act, so it's available), a pricing policy, and a sink routing-strategy swap.

**Open questions for grilling.** Soft-share functional form (logit temperature? attraction
weights?) and how it degrades to the current cheapest-first. Where the pricing lever lives
(new policy vs extend existing). Whether `base_price` should ever drift (a slow "regular price"
schedule) or stay static. Interaction with the existing affordability cap (`sink.cash /
list_price`) and CRN draw count (§6).

*(Real-data note: this machinery also enables an M5 replay mode later — declare `base = observed
price`, set `selling = base` so elasticity goes inert (M5 demand already embeds the real price
response — re-applying it would double-count), one sink per (item, store) so soft-share can't
re-split data that's already per-store. Separate effort; see the M5 feasibility study.)*


