# Phase-1 concrete policies + chain smoke scenario + chain tests

Status: ready-for-agent

## Parent

`.scratch/multi-echelon/PRD.md`

## What to build

Make the graph engine demoable on a 3-node chain (`F1 → S1 → Sink1`). After this slice, `uv run python scenarios/example_chain_three_node.py` produces a run log with non-trivial dynamics: factory produces, shop orders via the textbook rate-estimate / safety-horizon math, sink generates demand and buys.

Ship three concrete policies:
- `StaticFactoryPolicy(capacity_per_tick)` — produce `capacity_per_tick`, list at `unit_cost`
- `DefaultDemandSinkPolicy` — greedy cheapest-feasible buyer; respects `cash`
- `IntermediatePolicy.SingleSupplierAdapter` — wraps the existing `TextbookReorderPolicy` math, emits one-supplier order lines. Validates that the rate-estimate / safety-horizon math survives the new tick phasing before the multi-supplier extension lands in issue 9.

Ship the smoke scenario `scenarios/example_chain_three_node.py`: single product, `world_seed=42`, `n_steps=180`.

Ship two test files:
- `tests/sim/test_graph_runner_chain.py` — runs the chain for 50 ticks; asserts cash conservation (sum sinks-created == sum inventory-value-flowing + holding+fee-to-void + final balances) and no negative inventory anywhere
- `tests/sim/test_graph_determinism_chain.py` — asserts the three determinism invariants on the chain scenario:
  1. Same `world_seed` → bit-identical market, event, lifecycle sequences
  2. Same `(node_template, init_seed)` → bit-identical step-0 node state regardless of attached policy
  3. Policy swap on one node does not perturb `world_rng` (allocation_rng invariant deferred to issue 8 since allocation_rng is not consumed yet)

## Acceptance criteria

- [ ] `src/sim/policy.py` adds `StaticFactoryPolicy`, `DefaultDemandSinkPolicy`, `IntermediatePolicy.SingleSupplierAdapter`
- [ ] `SingleSupplierAdapter` preserves the textbook rate-estimate / safety-horizon math from `TextbookReorderPolicy`
- [ ] `scenarios/example_chain_three_node.py` exists with the spec above; `uv run python scenarios/example_chain_three_node.py` writes to `runs/example_chain_three_node/`
- [ ] `tests/sim/test_graph_runner_chain.py` asserts cash conservation and no negative inventory over 50 ticks
- [ ] `tests/sim/test_graph_determinism_chain.py` asserts the three invariants (allocation_rng invariant deferred to issue 8)
- [ ] `uv run pytest tests/sim -x` is green; legacy CLI smoke unaffected

## Blocked by

- `.scratch/multi-echelon/issues/05-graph-simulation-runner-chain-engine.md`
