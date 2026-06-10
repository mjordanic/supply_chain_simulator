# PRD: Variable-K RL — shared-weight per-product policy with deterministic arbiter

Status: ready-for-agent

Governing decision record: ADR 0021 (variable-K shared-weight policy with deterministic
arbiter), which supersedes ADR 0004 Decision 2 (slot-shuffle) and resolves ADR 0004 Decision 4's
deferred assortment head. Glossary terms used throughout are defined in CONTEXT.md: Active
subset, Arbiter, Implicit assortment, CRN-paired eval.

## Problem Statement

The RL stack can only train and run a policy for a fixed, episode-frozen number of products
(`K_active`), baked into the flat observation/action Box shapes, the MLP layer dimensions, the
slot-shuffle machinery, and the frozen-assortment rule. A practitioner cannot train one policy
that manages 3 products in one scenario and 20 in another, cannot express "stop stocking this
product / start stocking that one" as a policy decision, and cannot trust a saved checkpoint to
fail loudly when loaded against a mismatched layout. Separately, 8 of the 18 per-SKU observation
features are dead constants in the graph engine, wasting capacity and obscuring what the policy
actually sees, and within-tick cash exhaustion is resolved by arbitrary pid-iteration order in
the engine — a starvation pattern the policy can neither observe nor learn.

## Solution

One shared-weight policy applied per product row, with K sampled per episode (variable between
episodes, frozen within an episode — no mid-episode churn) and padded to `K_max` with a mask.
The policy expresses the full decision surface the engine supports at an intermediate node:
per-product price, per-product order quantity (via the order-up-to head), and implicit
start/stop (order-up-to target 0 = stop). A deterministic Arbiter — a pure module with two
variants behind a config switch (proportional fair-share, default; priority greedy) — projects
the joint per-product proposal onto the feasible set defined by node capacity *and* a cash
budget, making within-tick resource contention visible and learnable. The slot-shuffle is
deleted (permutation invariance becomes structural), the observation row is slimmed to live
features only, the critic becomes DeepSets-style, reward stays the shared node cash delta, and
checkpoints become self-describing.

## User Stories

1. As an RL researcher, I want the policy network to accept any number of products up to `K_max`, so that one trained policy generalises across scenarios with different catalog sizes.
2. As an RL researcher, I want K sampled per episode from a configured range, so that training data covers the K distribution without hand-building per-K environments.
3. As an RL researcher, I want K frozen within an episode, so that the learning problem stays stationary per episode and the baseline anchor needs no changes.
4. As an RL researcher, I want the policy to express "stop carrying a product" by driving its order-up-to target to zero, so that assortment decisions are learnable without a discrete action head or engine changes.
5. As an RL researcher, I want the trainable node to carry the full episode catalog superset, so that the policy can start ordering any episode product at any tick.
6. As an RL researcher, I want per-product decisions made by one shared-weight network applied row-wise, so that permutation invariance is structural and parameters don't scale with K.
7. As an RL researcher, I want the joint action log-prob computed as the masked sum of per-product Gaussian log-probs, so that PPO math is unchanged from the current implementation.
8. As an RL researcher, I want padded product slots masked out of the loss, advantages, and entropy, so that padding never contributes gradient signal.
9. As an RL researcher, I want a DeepSets-style critic (masked pool of per-product embeddings plus the global block), so that the value function is permutation-invariant and K-agnostic.
10. As an RL researcher, I want the observation row to contain only live features (the ~10 per-SKU features that vary in the graph engine, the 4 globals broadcast per row, contention aggregates, and the mask), so that dead constants (lifecycle one-hot, in-season flag, ticks-since-activation, mean-lead-time) stop wasting network capacity.
11. As an RL researcher, I want 1–2 aggregate contention features (total proposed quantity over free space; total estimated order cost over cash) in every product row, so that products sense competition for shared resources without seeing each other's identities.
12. As an RL researcher, I want a deterministic Arbiter that enforces both node capacity and a cash budget at decision time, so that within-tick resource exhaustion is decided visibly instead of by pid-iteration order in the engine.
13. As an RL researcher, I want the Arbiter's cash budget computed as node cash times a configurable fraction (default 1.0), costed at central-table offer prices, so that the engine's own cash clipping becomes a no-op backstop.
14. As an RL researcher, I want both Arbiter variants (proportional fair-share and priority greedy) implemented behind a single config switch, so that comparing them is a one-flag CRN-paired experiment.
15. As an RL researcher, I want the action layout to always carry three heads per product (price multiplier, order-up-to target, priority scalar) under both arbiter variants, so that both experiment arms share one checkpoint format and equal parameter budget.
16. As an RL researcher, I want proportional fair-share as the default arbiter, so that the baseline behaviour extends today's allocator semantics.
17. As an RL researcher, I want the priority-greedy arbiter to fill orders in the actor's learned priority order until capacity or cash is exhausted, so that the network can learn to starve one product to stock another — which proportional scaling structurally cannot express.
18. As an RL researcher, I want reward to remain the node's per-tick cash delta (inclusive of holding cost and order fees), so that PPO stays vanilla with a single advantage.
19. As an RL researcher, I want the slot-shuffle machinery deleted and `slot_permutation` removed from the episode spec and the CRN tuple, so that the eval protocol carries no vestigial state.
20. As an RL researcher, I want CRN-paired evaluation against the unchanged `OrderUpToPolicy` anchor on the reduced tuple, so that uplift numbers remain bit-identical-paired and comparable in protocol to prior results.
21. As an RL researcher, I want a permutation-invariance guarantee verified by test (reordering product rows yields correspondingly reordered decisions), so that the structural claim replacing the slot-shuffle is proven, not assumed.
22. As an RL researcher, I want a K-generalisation evaluation recipe (train on K ≤ 10, evaluate on K = 20), so that the scalability claim is measured rather than asserted.
23. As an RL practitioner, I want checkpoints saved as a dict bundling weights, training config, and an observation-layout version, so that loading a stale checkpoint fails with a clear error instead of a shape mismatch or silent success.
24. As an RL practitioner, I want the encoder to own a single obs-layout version constant, so that any layout change has exactly one place to bump and checkpoints validate against it.
25. As an RL practitioner, I want the vectorised training loop to work with padded uniform shapes across parallel envs, so that the existing vector-env machinery needs no replacement.
26. As a simulator maintainer, I want the engine untouched except where the episode sampler authors scenarios (superset `carried_products`), so that the redesign stays inside the RL sibling layer per the sim-as-base architecture.
27. As a simulator maintainer, I want the Arbiter to be a pure, torch-free module, so that it can be reasoned about and tested in isolation from the learner.

## Implementation Decisions

All per ADR 0021; restated here with module boundaries.

- **Eight modules.** (1) Arbiter — new pure module; (2) set encoder/decoder — rewrite of the
  observation/action codec; (3) masked set actor-critic, containing a masked joint Gaussian
  distribution sub-module; (4) episode sampler; (5) RL env; (6) checkpoint I/O — new small
  module; (7) eval; (8) train loop / rollout buffer.
- **Arbiter interface.** Pure function: per-product proposed quantities, per-SKU and global
  capacity headroom, cash budget, per-product unit prices (from central-table offers), optional
  per-product priority scalars, and a mode flag → integer allocations per product. Enforces
  per-SKU headroom, global free space, and the cash budget. Proportional mode scales by the
  binding feasibility ratio (extending today's two-pass fair-share semantics); greedy mode fills
  in descending priority until a resource binds. Absorbs the existing fair-share allocator.
- **Cash budget.** Node cash × configurable fraction, default 1.0. Engine clipping remains as a
  backstop and is expected to be a no-op; the rejection log would reveal violations.
- **Observation layout.** Per-product row: ~10 live per-SKU features (the slimmed survivors of
  the current 18 — dropping lifecycle one-hot, in-season flag, ticks-since-activation,
  mean-lead-time, all dead while `item_registry` is None) + 4 global features broadcast per row
  + 2 contention aggregates + 1 mask channel. Obs tensor `(K_max, F)` flattened for the Box
  space; mask recoverable from the layout. `K_max = 32`; K sampled per episode from [1, 20].
- **Action layout.** `(K_max, 3)`: price multiplier and order-up-to target with semantics
  unchanged from the scale-invariance package (ADR 0007), plus priority scalar. Priority is
  ignored in proportional mode but always present, so both arbiter arms share one layout.
- **Actor/critic.** Actor: one MLP applied row-wise, `(B, K_max, F) → (B, K_max, 3)`;
  state-independent log-std per head as today; tanh-squashed means. Joint distribution: masked
  sum of per-product diagonal Gaussian log-probs; masked entropy. Critic: per-product embedding
  MLP, masked mean-pool, concatenated with the global block, → scalar V.
- **Reward.** Unchanged: per-tick cash delta of the trainable node (includes ADR 0019 holding
  cost and order fees). No per-pid decomposition — the flow logs keep it available as a measured
  follow-up.
- **Episode sampler.** Draws K per episode, builds the degenerate graph (K factories → one
  trainable intermediate → K sinks) with the intermediate's `carried_products` set to the full
  episode product set. `slot_permutation` field deleted from the RL episode spec; the RL `slot`
  sub-seed stream is retired.
- **CRN tuple.** `(world_seed, capacity, balance, active_subset, allocation_sub_seed)`.
  `OrderUpToPolicy` anchor unchanged.
- **Checkpoint format.** Dict: actor state-dict, training config snapshot, obs-layout version.
  Loading validates the version and raises a descriptive error on mismatch. The encoder module
  owns the layout version constant.
- **No engine changes.** No dynamic `carried_products`, no listing economics, no mid-episode
  topology or demand churn. The implicit-assortment and no-churn decisions (ADR 0021 Decisions
  2–3) make these unnecessary.
- **Checkpoint break is accepted.** Existing trained checkpoints are invalidated; retraining is
  the migration, as with the ADR 0007 layout change.

## Testing Decisions

Good tests here assert external behaviour through the module's public interface — layout
contracts, feasibility invariants, distribution math — not internal structure. Property-style
assertions (invariance under permutation, constraints never violated) are preferred over
golden values wherever the property is the actual requirement.

Dedicated new tests for the three pure-core modules only (user decision):

- **Arbiter**: allocations never exceed per-SKU headroom, global free space, or cash budget;
  proportional mode preserves proposal ratios under a binding constraint; greedy mode fills in
  strict priority order and starves lowest priority first; integer outputs; K-free (works for
  K = 1 and K = 32 alike); zero-proposal and zero-budget edge cases.
- **Set encoder/decoder**: row layout matches the documented feature list; mask correct for
  K < K_max; padded rows are zero; decode honours the mask (no orders for padded slots);
  round-trip through arbiter produces per-pid dicts only for active products; layout version
  constant present and bumped (single source).
- **Masked joint Gaussian distribution**: log-prob equals the sum over active rows only; entropy
  excludes masked rows; sampling at fixed seed is deterministic; **permutation invariance** —
  permuting active rows of the input permutes per-row actions correspondingly and leaves the
  joint log-prob unchanged (the test that replaces the slot-shuffle's guarantee).

Existing tests in the RL suite that encode the old layout (encoder dims, slot-permutation
behaviour, episode-spec fields, CRN tuple, env smoke, PPO smoke, eval CRN) must be updated to
the new contracts as part of the respective issues — updating them is in scope; adding new
dedicated env/train/eval test suites is not.

Prior art: the existing RL encoder tests (`test_encoders`) for layout/decoding test style, the
eval CRN tests (`test_eval_crn`) for paired-determinism assertions, and the PPO/env smoke tests
(`test_ppo_smoke`, `test_env_smoke`) for the cheap-smoke pattern. The arbiter's feasibility
properties have prior art in the textbook-policy helper tests (`test_textbook_helpers`), which
already exercise the two-pass fair-share allocator semantics.

## Out of Scope

- Mid-episode product entry/exit (churn), scripted windows, or any `ItemRegistry`/PLC
  lifecycle restoration.
- Explicit carry/drop action head and per-SKU listing/shelf-space economics.
- Per-pid decomposed reward, factored advantages, or any non-vanilla PPO variant.
- Sequential per-product decisions, shadow-price/Lagrangian coordination, attention/set-
  transformer encoders (noted upgrades in ADR 0021, not built).
- Multi-node control or non-degenerate training graphs; supplier-selection action heads.
- Exposing `min_order_imposed` as an action head.
- The priority-vs-proportional uplift experiment itself (the *capability* ships; running and
  analysing the comparison is follow-up work).
- Engine changes of any kind, including dynamic `carried_products`.
- The K-generalisation training run (the eval recipe ships as configuration/documentation; the
  compute run is follow-up work).

## Further Notes

- Decision provenance: ADR 0021 records the chosen-over alternatives for every decision above;
  TODO.md §9–10 hold the original design notes, including deferred coordination upgrades.
- The arbiter making engine cash-clipping a no-op is an invariant worth asserting opportunistically
  during eval runs (rejection-log entries with reason `insufficient_cash` for the trainable node
  would indicate an arbiter bug), without building a dedicated test suite for it.
- `K_max` is a buffer/space constant, not a network parameter — raising it later does not
  invalidate weights, only checkpoint layout metadata (the layout version covers this).
- Issue breakdown should keep the three pure-core modules (arbiter, encoder/decoder, masked
  distribution) as early, independently mergeable issues; env/sampler/train/eval rewires depend
  on them.
- Scope addition at breakdown time (user request): a documentation sweep (the RL reference doc,
  top-level README RL section, CONTEXT.md planned-annotation flips, ADR 0021 status) and a
  rework of the two RL notebooks (train/eval and trained-agent analysis) so they illustrate the
  new stack, executed end-to-end with a tiny smoke config. Owned by issues 08 and 09.
- Issues live in `.scratch/variable-k-rl/issues/01`–`09`; dependency order is 01 → 02 → {03, 04}
  → 05 → 06 → 07 → {08, 09}.
