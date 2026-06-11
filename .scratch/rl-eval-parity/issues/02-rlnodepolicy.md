# 02: RLNodePolicy — Runner-compatible RL policy with the Arbiter inside

Status: ready-for-agent

## Parent

`.scratch/rl-eval-parity/PRD.md` (user stories 2, 4, 5, 6, 7, 10, 11, 12, 22)

## What to build

A new `RLNodePolicy` class — an `IntermediatePolicy` subclass in a new module under the RL
package (torch must not leak into the sim package). Attaching a trained agent to any
intermediate node becomes `policy_overrides={"<node-id>": RLNodePolicy.from_checkpoint(...)}`
plus a standard `Runner.run()`.

Inside `decide()`, the policy is fully self-contained: it builds the `(K_MAX, F)` set
observation from the engine-provided intermediate observation dict and central-table
snapshot, runs the `SetActor`, decodes the action, applies the Arbiter via the shared
arbitration helper (issue 01), and returns the standard intermediate action dict.

Behaviour fixed by the PRD (read its Implementation Decisions section in full; highlights):

- **Constructors**: `from_checkpoint(path, ...)` validating layout version via the existing
  checkpoint loader, plus an equivalent constructor wrapping a raw policy callable (needed
  by the eval harness in issue 03).
- **Deterministic by default** (squashed-Gaussian distribution means); opt-in stochastic
  mode seeds a policy-local torch generator, never global torch state.
- **First-tick capture**: opening cash (cash normalisation) and MSRP anchors (tick-0 list
  prices equal catalog MSRPs). Rolling sales history accumulated from `observed_sales`.
  The encoder's market argument needs only the current step, taken from the obs dict's tick.
  `base_demand_prior` is an optional constructor argument.
- **Contention features at parity**: the policy retains last-tick proposals and unit prices
  and feeds them into the next observation, matching the training env.
- **Managed product set**: defaults to the node's assortment (sorted list-price keys at
  first `decide()`); hard error with a clear message if it exceeds K_MAX; optional
  constructor override for a subset.
- **Supplier routing**: per tick per product, the order line goes to the cheapest current
  offer among the node's direct suppliers (ties by supplier id); no offer → no order line;
  no splitting across suppliers. The full candidate list still feeds the encoder's
  supplier-count/min-price features.
- **Single-run statefulness**: one instance per run, documented; no reset method.

## Acceptance criteria

- [ ] `RLNodePolicy.from_checkpoint(...)` + `policy_overrides` + `Runner.run()` completes a simulation on an intermediate node with no hand-driven tick loop
- [ ] Observation-parity test: training env and `RLNodePolicy` produce the same observation tensor on identical world state
- [ ] Arbiter-application test: under engineered contention (small capacity, many products) the returned order dict respects headroom/free-space/cash and differs from raw proposals
- [ ] Deterministic default: two `decide()` calls on identical state return identical actions with no global-seed manipulation; stochastic mode uses a policy-local generator
- [ ] K_MAX overflow raises early with a clear message
- [ ] Supplier-routing tests: cheapest-offer selection, supplier-id tie-break, and no-offer → no-order-line on a multi-supplier topology
- [ ] MSRP and opening-cash capture at first tick verified
- [ ] Tests assert external behaviour (returned action dicts, emitted tensors), not private attributes or call sequences

## Prior art

- `test_decode_routes_each_pid_to_its_own_supplier` — style reference for action-decoding behaviour tests
- Arbiter unit tests — allocation semantics the policy must surface
- `RLIntermediatePolicy` — the existing thin shim showing the `IntermediatePolicy` contract the new class must honour
- `build_intermediate_obs` plus the runner's injection of `observed_sales` / `direct_supplier_ids` — the exact obs dict `decide()` receives

## Blocked by

- 01-extract-shared-arbitration-helper
