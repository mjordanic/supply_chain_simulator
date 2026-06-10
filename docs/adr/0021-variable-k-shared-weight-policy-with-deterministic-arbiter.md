# Variable-K RL: shared-weight per-product policy with deterministic arbiter

Status: accepted. Supersedes ADR 0004 Decision 2 (slot-shuffle) and resolves ADR 0004 Decision 4's
deferred assortment head. Action/observation layout changes break existing checkpoints (retrain,
as with the ADR 0007 migration). Implemented in the variable-k-rl feature (issues 01–09).

The RL stack drops the fixed-`K_active` assumption so one policy manages any number of products,
with K sampled per episode. Products couple only through shared node capacity and the cash pool
(plus the per-`(node, supplier)` order fee), so the architecture is "decide per product
independently, reconcile deterministically" — completing the design already latent in
`fair_share_allocate` and the per-pid `decode_action` format.

**Decision 1 — Shared-weight per-product actor, padded to `K_max`, slot-shuffle deleted.** The
actor is one MLP applied row-wise: `(B, K_max, F) → (B, K_max, 3)` (price multiplier, order-up-to
target — head semantics unchanged from ADR 0007 — plus a priority scalar, Decision 4). Joint
log-prob is the masked sum of per-product Gaussian log-probs; the critic is DeepSets-style (masked
mean-pool of per-product embeddings + global block → scalar V). Episodes pad to `K_max = 32` with
a mask channel; K is sampled from [1, 20] per episode. Permutation invariance becomes structural,
so the slot-shuffle machinery is deleted: `slot_permutation` drops out of `RLEpisodeSpec` and the
CRN tuple, which shrinks to `(world_seed, capacity, balance, active_subset, allocation_sub_seed)`.
Chosen over (a) **fixed-slot MLP + slot-shuffle** (status quo) — cannot represent variable K at
all; (b) **attention/set-transformer encoder** — true learned coordination, but heavier; deferred
as the endgame upgrade, and the encoder contract makes it a class-level swap; (c) **bucketed
batching by K** — avoids padding waste but complicates the vec-env for negligible savings at
`K_max = 32`. The per-SKU observation row is slimmed first: 8 of the 18 features are dead in the
graph engine (`item_registry` is hardcoded `None`, so the lifecycle one-hot, in-season flag,
ticks-since-activation, and mean-lead-time are constants); the live row is ~10 per-SKU features
+ 4 broadcast globals + contention aggregates (Decision 4) + mask.

**Decision 2 — Implicit assortment: ordering zero is stopping.** The trainable node carries the
full episode catalog superset in `carried_products`; "stop a product" is the order-up-to head at
target 0 (sell down inventory), "start" is raising it. No discrete carry/drop head, no engine
listing/delisting mechanism. Chosen over (a) **explicit Bernoulli carry head + per-SKU listing
fee** — a real learned decision, but requires new engine economics and a mixed
continuous/discrete action distribution; deferred until there is evidence the implicit form is
insufficient; (b) **explicit head without new economics** — rejected outright: with holding cost
charged only on on-hand units, "dropped" and "carried at zero inventory" are economically
identical, so the head has no gradient signal.

**Decision 3 — No mid-episode churn.** K is frozen within an episode (the mask never changes
between `reset()`s); products enter and exit only across episodes. Chosen over (a) **scripted
per-product entry/exit windows in the episode sampler** and (b) **restoring `ItemRegistry`/PLC
lifecycle** — both add mid-episode masking, cold-start handling, and curriculum design before the
base variable-K problem is solved, and (b) additionally reverses ADR 0002. Consequence:
`OrderUpToPolicy` needs no changes as the CRN comparison anchor, and no curriculum is needed —
K is sampled per episode from day one.

**Decision 4 — Deterministic arbiter owning capacity *and* cash; two variants behind a config
switch.** The arbiter projects the joint per-product order proposal onto the feasible set defined
by node capacity and a cash budget (node cash × configured fraction, default 1.0, costed at
central-table offer prices). Variants: **proportional fair-share** (scale every proposal by the
binding feasibility ratio; extends today's `fair_share_allocate`, which enforces capacity only)
and **priority greedy** (fill in order of the actor's learned priority scalar until a resource is
exhausted — can express "starve A to stock B", which proportional structurally cannot). Default is
proportional. The priority head is always present in the action layout and ignored under
proportional, so the two arms share one checkpoint format and the comparison is a one-flag
CRN-paired experiment. Cash moves into the arbiter because engine-side enforcement alone fills
order lines in pid-iteration order until cash runs out — arbitrary starvation the policy can
neither observe nor learn; engine clipping in `execute_buy` remains as a backstop and should be a
no-op. From the actor's perspective the arbiter is environment dynamics: reward reflects
post-arbitration outcomes, so no second learner is needed. The observation row carries 1–2
aggregate contention features (Σ proposed quantity / free space, Σ estimated order cost / cash) so
products sense competition without seeing each other. Chosen over (a) **sequential per-product
decisions** (each row sees remaining resources — true coordination, but K forward passes per tick
plus order randomisation) and (b) **shadow-price/Lagrangian penalties** (principled, but tuning λ
is its own loop) — both noted as upgrades if contention features + arbiter prove insufficient.

**Decision 5 — Shared scalar reward.** Reward stays the node's per-tick cash delta (which, per
ADR 0019, includes holding cost and order fees); PPO is vanilla with a single advantage. Chosen
over **per-pid decomposed reward** from the ADR 0019 flow logs — cleaner credit assignment as K
grows, but requires per-product advantages, a factored rollout buffer, and an unresolved
attribution choice for the per-`(node, supplier)` order fee. The flows are already logged, so
decomposition remains a measured follow-up if learning stalls at higher K, not a speculative
upfront cost.

**Decision 6 — Self-describing checkpoints.** Training saves a checkpoint dict bundling the actor
`state_dict`, the training config, and an observation-layout version, replacing the bare
`actor.state_dict()` save — a stale checkpoint fails loudly instead of with a shape error (or
silently, once dims happen to match again).

Validation targets: permutation-invariance test (identical decisions under row reordering),
K-generalisation (train on K ≤ 10, evaluate on K = 20), and CRN-paired uplift vs `OrderUpToPolicy`
unchanged as the protocol.
