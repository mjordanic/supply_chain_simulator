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


## 9. Variable product count in RL: shared-weight per-product policy + deterministic reconciliation  — PLANNED

Remove the fixed-`K_active` assumption from the RL stack so a policy can manage an indefinite,
changing set of products — including products entering or ending mid-episode. Today fixed K is
baked in at four points: the flat obs/action `Box` shapes (`observation_dim(K) = K·18 + 4`,
`action_dim(K) = 2K`, `src/rl/encoders.py:106`); the MLP actor/critic first/last layer dims
(`src/rl/agents/ppo.py:71`); the slot-shuffle machinery (which exists *only* to stop the fixed-slot
MLP binding identity to position, ADR 0004 Decision 2); and the frozen-assortment rule
(ADR 0004 Decision 4).

**Key structural fact.** In the current sim, products couple only through two shared resources —
node capacity and the cash pool (plus the per-(node, supplier) order fee). Demand sinks are
per-product with independent distributions; no cross-elasticity yet (that's §8). And the
reconciliation step already exists: `fair_share_allocate` (`src/rl/encoders.py:607`) projects the
joint per-product request onto the feasible set and is K-free, as is the per-pid dict format of
`decode_action`. So "decide per product independently, reconcile deterministically" is the
natural completion of the current design, not a rework.

**Slim the obs row first — 8 of the 18 per-SKU features are dead in the graph engine.**
`Simulation.item_registry` is hardcoded `None` (`src/sim/runner.py:104`) and both `env.py` and
`eval.py` pass it straight to the encoder, so during all current training and eval: the lifecycle
one-hot (slots 6–10) is a constant `[1,0,0,0,0]` fallback, the in-season flag (slot 12) is a
constant `1.0`, ticks-since-activation (slot 11) degenerates to a log-scaled step counter
(`activation_tick` only ever existed on `DemandSinkNode`, and was removed with freshness — §1),
and mean-lead-time (slot 16) is hardcoded `0.0` (`src/rl/encoders.py:319–342, 366`). Drop them
when building the per-product row rather than porting dead weight: the live row is ~10 per-SKU
features. Side effect that collapses §7's hard part: `market` was only needed for the in-season
flag and `registry` is gone, so the slimmed encoder needs nothing that
`decide(obs_intermediate, central_table)` can't supply — rolling sales history is self-trackable
from the `observed_sales` the runner already injects (`src/sim/runner.py:439`), exactly as the
textbook policies do (`src/sim/policy.py:2074`). §7's option (a) then needs **no engine change**:
a self-contained inference policy becomes a thin wrapper.

**Design sketch.**
- **Per-product obs row** (~16-dim): the ~10 live per-SKU features + the 4 global features
  broadcast onto each row, plus 1–2 aggregate "contention" features (e.g. Σ naive requests /
  free space) so products sense competition without seeing each other.
- **Shared-weight actor** applied per product: maps `(B, K, F) → (B, K, 2–3)` (price mult,
  order-up-to target — head semantics unchanged from ADR 0007 — plus an optional learned
  *priority* scalar under the greedy arbiter, below). Joint log-prob = masked sum of
  per-product Gaussian log-probs; PPO math otherwise unchanged.
- **Deterministic reconciliation — implement both arbiters behind a config switch.** When the
  joint proposal exceeds free capacity or the cash budget: (a) **proportional fair-share** —
  scale all proposals by the feasibility ratio (this is `fair_share_allocate` today); (b)
  **priority greedy** — the actor's per-product priority scalar ranks products, and the arbiter
  fills orders in priority order until capacity/cash is exhausted. From the actor's perspective
  the arbiter is just environment dynamics: reward reflects post-arbitration outcomes, so
  gradients push proposals toward what survives — no second learner needed. Priority-greedy lets
  the network *learn* "starve A to stock B", which proportional structurally cannot express;
  having both makes the comparison a one-flag CRN-paired experiment.
- **Critic**: DeepSets-style — mean/sum-pool per-product embeddings + global block → scalar V.
- **Coordination upgrades — note, don't build.** If contention features + arbiter prove
  insufficient: *sequential decisions* (decide products one at a time, each row seeing the
  *remaining* free capacity/cash after earlier commitments this tick — true coordination, single
  network, but K sequential forward passes and order randomisation during training) or a
  *shadow-price penalty* (Lagrangian decomposition: penalise each product's reward by
  λ × resource usage — principled, but tuning λ is its own loop). §10's attention encoder is the
  heavier endgame for learned coordination.
- **Variable K**: pad to `K_max` with a mask channel in obs, action, and rollout buffer; masked
  slots drop out of loss. K sampled per episode; rows appear/disappear mid-episode as products
  enter/end. Cold start already solved — `effective_rate` floors at `base_demand_prior`.
- **Delete the slot-shuffle**: permutation invariance becomes structural. `slot_permutation`
  drops out of `RLEpisodeSpec` and the CRN tuple (`src/rl/eval.py`).
- **Per-product reward decomposition** (same change): node-level cash-delta reward shared across
  K per-tick decisions gets noisy as K grows; the per-pid flows (revenue, order cost, holding —
  ADR 0019) give a decomposed reward.
- Checkpoint break: obs/action layout changes → retrain (same as the ADR 0007 migration).
  Unblocks §7's "exactly `K_active` SKUs" constraint as a side effect. While at it, stop saving
  a bare `actor.state_dict()` (`src/rl/train.py:94–109`): bundle the config and an obs-layout
  version into the checkpoint dict so a stale checkpoint fails loudly instead of with a shape
  error (or silently, once dims happen to match again).

**Open questions for grilling.** What drives mid-episode assortment churn now that lifecycle is
removed (§2) — restore `ItemRegistry`/PLC stages, or a simpler scripted entry/exit schedule in
the episode sampler? What does "inactive product" mean topologically — superset graph with all
potential sinks/factories from tick 0 (the demand-pull schedule is computed once at `build_world`
and cached), or true dynamic topology (engine change)? Credit assignment: is the decomposed
per-pid reward needed at K ≤ 20 or is shared reward fine — and where does the per-supplier order
fee land in the decomposition? Which contention aggregates suffice without attention? Arbiter
details: which of the two is the default; is the priority head worth the extra action dim
(measure: CRN-paired uplift, greedy vs proportional, equal budget); what exactly is the cash
budget the arbiter enforces (full cash? a configured fraction?) and does it pre-empt or duplicate
the engine's own feasibility clipping? `K_max` padding vs bucketed batching for the vec-env? Eval
protocol: does `OrderUpToPolicy` need changes to anchor CRN-paired comparison under churn, and
what replaces `slot_permutation` in the CRN tuple? Train on fixed-K episodes first and introduce
churn as curriculum, or churn from day one?

---

## 10. Set/attention policy over product tokens (upgrade of §9) + JEPA-ready latent space  — PLANNED

Swap §9's independent per-product encoder for a set encoder: products as tokens (per-product
features → embedding), optionally a global token, 2–3 self-attention blocks, per-token action
heads, attention-pooled critic. ADR 0004 Decision 2 already anticipated exactly this ("a future
switch to a set-encoder is a class-level change: only `Actor`/`Critic` need to change; the
encoder contract is unchanged"). §9 should therefore keep the per-product encoder behind a
narrow interface so this is a drop-in swap; §9's masking, variable-K buffer, and reconciliation
all carry over.

**When it pays.** (a) Once cross-product demand coupling exists — §8's price elasticity and
soft-share competition, or substitution via `related_products` — coordination must be learned
in-policy, which the independent encoder structurally cannot do beyond its aggregate features.
(b) Large catalogs: full self-attention is O(K²); switch to inducing-point attention
(Set-Transformer style) if K reaches hundreds.

**JEPA tie-in (the load-bearing reason to keep the token architecture).** A per-product token
space is the substrate a JEPA world model needs: pretrain the token encoder + a latent predictor
on cheap sim rollouts (textbook-policy runs are free data) by masking product tokens and/or
predicting next-tick token embeddings, loss in latent space, no reconstruction. Then PPO
fine-tunes only the action heads. The flat fixed-K MLP is JEPA-hostile; §9's per-product encoder
is the degenerate zero-attention case of this architecture, so the A → B path keeps the option
open the whole way.

**Open questions for grilling.** Is attention worth anything *before* §8 lands (no substitution
today — what coordination signal would it learn beyond fair-share contention)? Architecture:
full self-attention vs inducing points; depth/width; explicit global token vs pooled readout.
JEPA specifics: mask across products, across time, or both; does the predictor condition on
actions (action-conditioned world model) or state only; collapse safeguards (EMA target encoder
à la I-JEPA); pretraining data mix (textbook vs random vs mixed policies); freeze or fine-tune
the encoder during PPO. Evaluation: what's the falsifiable claim — pretrained-then-finetuned
beats from-scratch §9 PPO on sample efficiency at equal wall-clock, measured on the CRN-paired
uplift? Is this sim rich enough for representation pretraining to matter at all, or does JEPA
wait for §8/M5-replay worlds?


