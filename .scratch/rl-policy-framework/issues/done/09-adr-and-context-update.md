# 09 — ADR 0004 + `CONTEXT.md` glossary update

Status: ready-for-agent

## Parent

PRD: `.scratch/rl-policy-framework/PRD.md`

## What to build

Documentation artifacts that capture the architectural decisions baked into the RL framework so future readers (and future heavier-model swaps) inherit the rationale rather than the conclusions alone.

### `docs/adr/0004-rl-training-env.md`

A new ADR covering the four core RL-env design decisions:

1. **Randomised assortment per episode** — drawing K active SKUs uniformly per episode from a 100-product catalog universe, rather than holding the assortment fixed across runs.
2. **Slot-shuffled observation encoding** — per-episode permutation of the K slots over the active SKUs to force permutation invariance, rather than fixed identity slots or set-based attention.
3. **Hidden regional market state** — `market_demand` / `market_supply` deliberately excluded from the observation, preserving the partial-observability assumption the baseline policy works under.
4. **Frozen-assortment-within-episode** — `activate` / `deactivate` decisions disabled for the duration of an episode; catalog rotation deferred to a follow-up.

For each decision, cite the alternatives considered (set-based attention, per-product factored policy, exposing market state, per-tick rotation head) and the rationale that selected the chosen path. Use the project's ADR format (see `.agents/skills/grill-with-docs/ADR-FORMAT.md` if present).

### `CONTEXT.md` glossary update

Add the five new domain terms inline (alphabetically or topically integrated, not appended in a separate "RL" section):

- **RL Env** — Gymnasium-compatible env wrapping the simulator's `Market` / `EventEngine` / `ItemRegistry` / `Store` subsystems in step-by-step semantics. Each `reset()` produces a fresh `Scenario` via `episode_sampler`. Distinct from `Runner`, which remains the batch-run entry point.
- **RL Episode** — one `reset()`-to-terminated pass through the env. Fixed at 180 ticks. Distinct from "Scenario" (which describes a full experiment).
- **Active subset** — the K (default 5) product ids drawn uniformly per episode from the catalog universe; the only SKUs the RL agent decides for during that episode.
- **Slot-shuffled observation** — observation tensor where the K active SKUs occupy K slots, with the slot-to-SKU mapping permuted per episode (drawn from the per-episode RNG; inverse applied on action decode).
- **CRN-paired eval** — evaluation protocol where the RL policy and `BaselinePolicy` are run on bit-identical `(world_seed, init_seed, capacity, balance, active subset, slot permutation)` tuples; uplift is computed paired per seed.

## Acceptance criteria

- [ ] `docs/adr/0004-rl-training-env.md` exists and follows the project's ADR format
- [ ] All four decisions documented with alternatives and rationale
- [ ] `CONTEXT.md` has the five new terms integrated into the existing glossary
- [ ] No new domain term is coupled to implementation details (file paths, function names) — terms read at the domain-expert level

## Blocked by

- `.scratch/rl-policy-framework/issues/05-rl-policy-shim-and-env.md`
