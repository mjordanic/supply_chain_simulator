# RL policy as a first-class IntermediatePolicy; eval runs through the standard Runner; the Arbiter travels with the policy

Status: Accepted (implemented 2026-06 — `src/rl/node_policy.py`, `src/rl/eval.py`;
`rl-eval-parity` feature, issues 01–06).

The RL stack gains a first-class `IntermediatePolicy` subclass (`RLNodePolicy`) that wraps
a trained `SetActor` checkpoint so it can be attached to any intermediate node via the
standard `Runner(scenario, policy_overrides={"<node-id>": policy}).run()` API — the same
one-liner used for tuned textbook policies. As a consequence, both the offline eval harness
and any multi-echelon deployment run the RL policy through identical engine code paths,
making train/eval parity structural rather than asserted.

**Decision 1 — `RLNodePolicy` as a first-class `IntermediatePolicy`.**
`RLNodePolicy` (from `src/rl/node_policy.py`) implements `decide(obs, central_table)` so
the policy is self-contained: it builds the `(K_MAX, F)` observation tensor from the
engine-provided observation dict and central-table snapshot, runs the `SetActor`,
decodes the action, applies the Arbiter via the shared `arbitrate_orders` helper, and
returns the standard intermediate action dict. `from_checkpoint(path)` is the primary
constructor; it validates layout version via `checkpoint.load()`, failing loudly on
mismatch rather than silently.

Design constraints baked into `RLNodePolicy`:
- Deterministic by default (squashed-Gaussian distribution means); opt-in stochastic mode
  via `deterministic=False` using a policy-local `torch.Generator`, never global torch state.
- First-tick capture: opening cash and MSRP anchors are latched on the first `decide()` call.
- Rolling sales history accumulated from `observed_sales`.
- Contention features for the next observation derived from last-tick proposals and prices.
- Managed product set: sorted `list_prices` keys at first `decide()`.
- Hard error if the assortment exceeds `K_MAX`.
- Supplier routing: cheapest current offer among direct suppliers; no offer → no order line.
- Single-run statefulness; no `reset()` method. Create a fresh instance per `Runner.run()`.

Chosen over (a) **ad-hoc wrapper inside `RLEnv`** — was the pre-fix status quo; kept
the Arbiter out of the eval path because eval called a bespoke RL-specific loop, not
`Runner.run()`. Train/eval asymmetry meant every eval-harness fix drifted from training
dynamics. (b) **modify `Runner` or `build_world` to accept a raw policy callable** —
special-cases the engine; `IntermediatePolicy` is already the extension point and carries
no cost.

**Decision 2 — Offline eval and cold-start probe rebuilt on `Runner.run()`.**
`evaluate()` (`src/rl/eval.py`) wraps the RL policy callable in `RLNodePolicy` via
`RLNodePolicy.from_policy_fn()` and attaches it via `policy_overrides`. Both the RL arm
and the baseline arm (`OrderUpToPolicy` + `SingleSupplierAdapter`) therefore run through
`Runner.run()` identically — the same engine code, the same demand path, the same
settlement logic. The only difference between the two arms is the policy; all world
variance cancels through CRN pairing.

The cold-start order-quantity probe is rebuilt on the same path. It now measures
**post-Arbiter allocations** (the real order) rather than raw proposals. This is a
semantics change vs the old bespoke path: pre-fix probes measured what the actor
proposed before the Arbiter reconciled, which could exceed capacity or cash; post-fix
probes measure what actually entered the engine.

**Decision 3 — Eval-history discontinuity: pre-fix TensorBoard `eval/*` curves are not
backfilled.**
TensorBoard `eval/*` scalars recorded before this fix (in existing `runs/` directories)
were logged by the old arbiter-less eval path and will not match a fresh eval of the
same checkpoints under the rebuilt harness. They are left as recorded; no backfill is
performed. The discontinuity is noted in run-directory names by convention
(`_fixed` suffix in the reference fashion runs), not by the tooling. Any new eval run
is comparable only with other runs that use the rebuilt harness.

**Decision 4 — Single-instance-per-run convention and deterministic default.**
`RLNodePolicy` is stateful (rolling sales history, first-tick anchors, contention carry)
and has no `reset()` method. Sharing one instance across multiple `Runner.run()` calls
will corrupt state silently. The convention: one fresh `RLNodePolicy` instance per
`Runner.run()` call. The deterministic default (`deterministic=True`) ensures
reproducibility for analysis notebooks and CRN-paired comparisons; stochastic mode
requires an explicit `deterministic=False` plus an optional `policy_seed`.

Considered alternatives:

(a) **Add a `reset()` method to `RLNodePolicy`.** Rejected: the opening-cash and
MSRP anchors need the node's `list_prices` and `cash` at the very first decide tick,
which is not available at construction time. A `reset()` that needs to re-capture
node state is equivalent to constructing a new instance, and hiding that via a method
invites subtle bugs when the caller forgets to call it.

(b) **Run the eval harness through `RLEnv.step()` directly (keep the bespoke loop).**
Rejected: the bespoke loop cannot reuse `inspect.flow_frame` / `metrics.business_metrics`
without duplicating the extraction logic; it cannot emit a `run_log` compatible with
the standard analysis notebooks; and it keeps two separate code paths for what is
conceptually one operation (running a policy on a scenario).

(c) **Separate eval `Arbiter` instance independent of the policy.** Rejected: the Arbiter
configuration (mode, cash-budget fraction) must match the one the policy trained under.
Packaging the Arbiter inside `RLNodePolicy` means `from_checkpoint` reads the saved
config and wires the correct Arbiter automatically; a separate instance requires the
caller to know and replicate the training config.

## Cross-references

- ADR 0021 — Variable-K shared-weight policy and deterministic Arbiter owning capacity +
  cash; `RLNodePolicy` is the deployment wrapper for the architecture described there.
- ADR 0004 — Original RL env design; this ADR resolves the "deploy a checkpoint onto an
  arbitrary graph node" follow-up item left open in ADR 0004.
- ADR 0019 — Flow-logged metrics; `evaluate()` uses `inspect` + `metrics` for KPI
  extraction on both arms, so both arms benefit from the ADR 0019 decomposition.
