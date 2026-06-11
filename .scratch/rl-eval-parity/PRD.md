# PRD: RL eval parity & attach-anywhere RL policy

Status: ready-for-agent

## Problem Statement

Two gaps surfaced while analysing the `fashion_run` PPO studies in notebook 06a:

1. **The offline eval harness simulates a different environment than training.** ADR 0021
   defines the Arbiter as environment dynamics — the deterministic reconciler that projects
   the agent's joint order proposal onto the feasible set (per-SKU headroom, global free
   space, cash budget). The training env applies it every tick; the offline CRN-paired eval
   harness never calls it, and additionally feeds zeroed contention features into the
   observation (training feeds last-tick proposals/prices). Consequences: every offline eval
   number is measured under slightly wrong dynamics, and greedy-arbiter checkpoints are
   evaluated with their priority head disconnected — the notebook's arbiter comparison
   (section 5c) understates the greedy agent.

2. **There is no first-class way to attach a trained RL policy to a graph node.** Running a
   trained checkpoint requires hand-driving the engine's two-phase tick loop (the bespoke
   ~200-line loop inside the eval harness). Every other policy in the repo attaches with one
   line via `policy_overrides` and runs through the standard Runner; the RL policy should
   too — on any intermediate node, in any scenario.

## Solution

Make the trained RL policy a first-class `IntermediatePolicy`:

- **`RLNodePolicy`** — a self-contained `IntermediatePolicy` subclass. Inside `decide()` it
  builds the `(K_MAX, F)` set observation from the engine-provided observation dict and
  central-table snapshot, runs the `SetActor`, decodes the action, **applies the Arbiter**,
  and returns the standard action dict. Attaching a trained agent becomes
  `policy_overrides={"<node-id>": RLNodePolicy.from_checkpoint(...)}` + `Runner.run()`.
- **Shared arbitration helper** — the arbiter block currently inlined in the training env's
  step function (proposal totals → headroom/free-space/cash-budget → `allocate(mode=...)` →
  rebuilt order dict) is extracted into one shared function in the arbiter module, called by
  both the training env and `RLNodePolicy`. Train/eval parity becomes structural: one code
  path computes the dynamics.
- **Eval rebuilt on the policy** — the offline eval harness's bespoke RL episode loop is
  replaced by `Runner.run()` with an `RLNodePolicy`, making the RL arm of every CRN pair run
  through the identical engine code as the baseline arm. The `evaluate()` public signature is
  unchanged (still accepts an RL policy callable; wraps it in a fresh `RLNodePolicy` per
  spec), so the training loop's in-training eval needs no changes.
- **Notebooks 06 and 06a** updated and fully re-executed on the fixed path; 06a gains a new
  final section demonstrating attach-and-run.
- **Docs**: ADR 0022 recording the decision and the eval-history discontinuity; RL README
  attach section; CONTEXT.md Arbiter entry clarified.

## User Stories

1. As an RL researcher, I want the offline eval to apply the Arbiter exactly as training does, so that eval numbers measure the environment the agent actually learned.
2. As an RL researcher, I want eval observations to carry the same contention features as training observations, so that the agent is not evaluated on silently-zeroed input channels.
3. As an RL researcher, I want greedy-arbiter checkpoints evaluated with their priority head connected, so that the proportional-vs-greedy comparison in notebook 06a is a fair fight.
4. As an RL researcher, I want to attach a trained checkpoint to a node with one line (`policy_overrides` + `Runner.run()`), so that running an RL agent is as easy as running any textbook policy.
5. As an RL researcher, I want to attach the policy to **any** intermediate node in **any** scenario topology, so that I can probe how the agent behaves outside its training world.
6. As an RL researcher, I want deterministic actions by default when loading a checkpoint, so that the same world state always produces the same decision without hidden global-seed dependencies.
7. As an RL researcher, I want an opt-in stochastic mode with a policy-local random generator, so that I can reproduce training-style rollouts without touching global torch state.
8. As an RL researcher, I want the RL arm and the baseline arm of a CRN pair to run through the identical engine code path, so that paired uplift is attributable to the policy alone.
9. As an RL researcher, I want a run produced by an attached RL policy to flow through the standard run log, `inspect`, and `metrics` tooling, so that I can analyse RL runs with the same plots as any other run.
10. As an RL researcher, I want the policy to manage the node's own assortment by default and let me override the managed product set, so that attachment works without ceremony but stays controllable.
11. As an RL researcher, I want a clear, early error when a node's assortment exceeds the encoder's K_MAX, so that an oversized attachment fails loudly instead of silently truncating.
12. As an RL researcher, I want orders routed to the cheapest currently-offering direct supplier per product, so that routing on multi-supplier topologies is deterministic and matches how the agent's cash constraint is priced.
13. As a notebook reader, I want notebook 06a's sections re-run on the fixed eval path with the stale "eval ignores the Arbiter" caveats removed, so that the analysis text matches what the code actually does.
14. As a notebook reader, I want a worked attach-and-run example in notebook 06a (in-distribution node, standard Runner + inspect tooling), so that I can copy the deployment recipe directly.
15. As a notebook reader, I want a second example attaching the same checkpoint to a node in a deeper topology with an explicit out-of-distribution caveat, so that I understand the API generalises even where the agent does not.
16. As a notebook reader, I want notebook 06 re-executed so its training-and-eval walkthrough reflects the new eval semantics, so that the two RL notebooks do not contradict each other.
17. As a future contributor, I want one shared arbitration function used by training and eval/deployment, so that train/eval parity bugs of this class are structurally impossible.
18. As a future contributor, I want ADR 0022 to record why pre-fix TensorBoard eval curves do not match fresh evals of the same checkpoints, so that the historical discontinuity is explained rather than rediscovered.
19. As a future contributor, I want the CONTEXT.md Arbiter entry to state that the Arbiter is environment dynamics in training and travels inside the policy at eval/deployment, so that the two-homes design reads as intended rather than accidental.
20. As a future contributor, I want the RL README to document checkpoint attachment, so that the deployment path is discoverable without reading notebook source.
21. As an agent implementer, I want the arbiter extraction to be behaviour-preserving for training (existing RL test suite green, unchanged training semantics), so that retrained baselines remain comparable.
22. As an agent implementer, I want an observation-parity test asserting the training env and `RLNodePolicy` produce the same observation tensor on identical world state, so that the unification claim is verified, not assumed.

## Implementation Decisions

Decisions below were settled in a design interview; treat them as fixed.

- **Arbiter always applied in eval — no flag.** No `apply_arbiter` opt-out; a no-arbiter eval
  is a category error under ADR 0021, not a useful ablation. (If an ablation is ever wanted,
  `arbiter_mode` can grow a `"none"` value later — out of scope now.)
- **Eval numbers change, accepted.** All offline eval outputs shift (slightly for
  proportional, more for greedy). Historical TensorBoard `eval/*` curves in existing run
  directories were logged by the old path and stay as recorded; ADR 0022 documents the
  discontinuity.
- **Shared arbitration helper** lives in the arbiter module: takes the decoded order
  proposals, a snapshot of node state (inventory, pending, capacity, cash), per-product unit
  prices, per-product priorities, and the RL config; returns the final arbitrated order dict.
  Both the training env's step function and `RLNodePolicy.decide()` call it. The existing
  pure `allocate()` stays untouched underneath.
- **`RLNodePolicy`** lives in a new module under the RL package (torch must not leak into the
  sim package). Constructors: `from_checkpoint(path, ...)` (validates layout version via the
  existing checkpoint loader) and an equivalent that wraps a raw policy callable (used by the
  eval harness so `evaluate()` keeps its signature).
- **Self-contained observation building.** Everything comes from the `decide()` arguments:
  the intermediate observation dict (inventory, pending, capacity, cash, tick,
  observed_sales, direct_supplier_ids, list_prices) plus the central-table snapshot. Captured
  on first `decide()`: opening cash (cash normalisation) and MSRP anchors (tick-0 list prices
  equal catalog MSRPs). Rolling sales history is accumulated from `observed_sales` across
  ticks. The encoder's market argument needs only the current step — satisfied from the obs
  dict's tick. `base_demand_prior` (cold-start demand guess) is an optional constructor
  argument; the eval harness computes it from the episode spec exactly as the old loop did.
- **Contention features at parity.** The policy retains its last-tick proposals and unit
  prices and feeds them into the next observation, matching the training env (fixes the
  silent zeroed-features gap).
- **Single-run statefulness.** One policy instance per run (same convention as the baseline
  factory in the eval harness). Document it; no reset method.
- **Deterministic by default in `from_checkpoint()`**: actions are the squashed-Gaussian
  distribution means. Opt-in `deterministic=False` with a seed held in a policy-local torch
  generator, never the global torch state.
- **Managed product set**: defaults to the node's assortment — sorted list-price keys
  captured at first `decide()`; hard error with a clear message if it exceeds K_MAX (32);
  optional constructor override to manage a subset. Sorting is for determinism only (the
  set actor is permutation-equivariant).
- **Supplier routing**: per tick per product, candidates are the node's direct suppliers that
  currently offer the product on the central table; the order line goes to the cheapest
  current offer, ties broken by supplier id; no offer → no order line for that product this
  tick. The full candidate list still feeds the encoder's supplier-count/min-price features.
  No splitting of one product's order across suppliers (the action space has no split head).
- **Eval harness rebuild**: the bespoke RL episode loop and its hand-rolled per-tick log
  harvesting are deleted; the RL arm runs `Runner.run()` with `policy_overrides`, the run log
  flows through the same inspect/metrics builders as the baseline arm. The cold-start
  order-quantity probe is rebuilt the same way (it now measures post-arbiter allocations —
  the real order — rather than raw proposals; noted in ADR 0022). The insufficient-cash
  warning behaviour for the RL node is preserved.
- **`evaluate()` public signature unchanged** (RL policy callable + baseline factory +
  specs + config). The training loop's in-training eval closure keeps sampling
  stochastically, so the meaning of in-training eval curves is unchanged going forward.
- **Notebook 06a**: stale caveats replaced by the parity statement; sections that evaluate
  checkpoints re-run; new final section with two cells — (1) in-distribution attach of the
  best proportional checkpoint via Runner + standard inspect/metrics plots, (2) same
  checkpoint attached to a node in a deeper topology (notebook 03 gallery style) with an
  explicit out-of-distribution caveat. Fully re-executed.
- **Notebook 06**: prose updated (including "where to next" mentioning attachment), fully
  re-executed (retrains its tiny demo agent — minutes).
- **ADR 0022**: "RL policy as a first-class IntermediatePolicy; eval runs through the
  standard Runner; the Arbiter travels with the policy." Records the trade-off (bespoke-loop
  parity vs. one-code-path), the eval-history discontinuity, and the cold-start-probe
  semantics change.
- **CONTEXT.md**: Arbiter entry gains the two-homes clarification; RL README gains an attach
  section.

## Testing Decisions

- Good tests here assert **external behaviour**: the action dict a policy returns, the
  metrics an eval produces, the tensor an encoder emits — never private attributes or call
  sequences.
- **Arbiter extraction** is guarded by the existing RL test suite (arbiter unit tests for
  proportional/greedy semantics; env step tests) — it must pass unchanged, proving
  behaviour preservation for training.
- **`RLNodePolicy`** is the deep module that earns the new tests:
  - observation parity: training env and policy produce the same observation tensor on
    identical world state (the unification claim, verified);
  - arbiter application: with contention engineered (small capacity, many products), the
    returned order dict respects headroom/free-space/cash and differs from raw proposals;
  - deterministic default: two `decide()` calls on identical state return identical actions
    with no global-seed manipulation;
  - K_MAX overflow raises with a clear message;
  - supplier routing: cheapest-offer selection and the no-offer → no-order-line rule on a
    multi-supplier topology;
  - MSRP/opening-cash capture at first tick.
- **Eval rebuild**: existing eval-harness tests (CRN pairing, metric keys, two-scale eval)
  must stay green with the new internals; add one regression asserting the RL arm's run log
  flows through the standard inspect path.
- Prior art: the decode routing test (`test_decode_routes_each_pid_to_its_own_supplier`) for
  action-decoding behaviour tests; the arbiter unit tests for allocation semantics; the
  existing eval tests for CRN-paired metric contracts. Notebook re-execution is verified by
  executing the notebooks end-to-end (the repo's established pattern), not by unit tests.

## Out of Scope

- An `arbiter_mode="none"` ablation value.
- Splitting a single product's order across multiple suppliers (no action-space support).
- Retraining the `fashion_run_fixed` / `fashion_run_greedy` studies; existing checkpoints are
  re-evaluated, not retrained.
- Any change to training dynamics, the action layout, the encoder feature set, or the
  checkpoint format.
- Making the agent perform well on out-of-distribution topologies — the any-node demo proves
  the API, not transfer.
- Backfilling or rewriting historical TensorBoard eval scalars.

## Further Notes

- Expected observable effect of the fix on the existing analysis: notebook 06a section 5c is
  currently a conservative estimate of the greedy checkpoint (priority head disconnected at
  eval). Post-fix numbers supersede the recorded comparison table; the text must be rewritten
  against fresh outputs, not patched.
- The interview that fixed these decisions happened on 2026-06-10/11; the design tree was
  resolved question-by-question (arbiter-always-on → Runner-compatible policy class → eval
  unification → deterministic default → assortment/routing defaults → notebook scope →
  ADR). If an implementer hits an ambiguity not covered above, prefer the reading that
  maximises train/eval parity.
