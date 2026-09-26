# 02: Fit the latent model and plan with it

**What to build:** From a transition log, fit a per-SKU latent predictor and a direct predictor. Attach either as an `IntermediatePolicy`. `decide` draws a fixed budget of proposals from `policy_rng`, rations by priority, and keeps the proposal with the best imagined engine `net_profit` over the lead-time horizon. Probes decode on-hand, in-transit, and sales, and their loss does not change the encoder. A frozen Policy faced with a till that empties only at `holding_rate = 0.05` places a different order than it does at `0.01`. The PPO stack is not imported.

**Blocked by:** 01 — Exploratory Policy and synthetic transition log.

**Status:** ready-for-agent

## Parent

`.scratch/jepa-replenishment/PRD.md`. ADR 0025, ADR 0027, ADR 0029.

## Stories

PRD stories 12–29, 39, 40, 64, 69.

## Prior art

The observation an `IntermediatePolicy` already receives (on-hand, pending, `observed_sales`, cash, capacity). Holding cost and order fee charged by the engine (ADR 0019). `policy_rng` kept disjoint from `world_rng`. Ticket 01's log schema and exploratory ration.

## Artifact homes

- Fit and planner live in `src/jepa/` (tracked).
- Smoke checkpoints may stay in memory. The published checkpoint path is `runs/jepa-replenishment/checkpoints/` (gitignored), written by ticket 06.

## Acceptance criteria

- [ ] A tiny log produced by `Runner` can be fit, and the resulting Policy returns a valid order dict through `policy_overrides`: `list_price` from `base_price`, quantities inside capacity and cash.
- [ ] The encoder sees on-hand, in-transit, a sales window of length `delivery_lag`, and broadcast cash and capacity. The Market multiplier and the disruption type are not features.
- [ ] The predictor unrolls open-loop for `H = delivery_lag`. The latent loss matches a stopped target embedding at each step. An anti-collapse penalty is on.
- [ ] Probes regress next on-hand, next in-transit, and sales over the step from a stopped embedding. A test that fits probes does not change encoder parameters. The profit arithmetic is not differentiated into the encoder.
- [ ] Imagined profit is revenue minus purchase cost minus holding cost minus order fee. There is no stockout penalty and no learned reward head. Imagined cash uses the rate under test and feeds the next ration.
- [ ] `decide` draws its budget from `policy_rng` only. The published default budget is 64. Smoke tests use 4. The same `policy_seed` repeats the same orders.
- [ ] The direct predictor, trained on the same log to emit those three quantities with no latent target, is a second Policy with the same shooter and the same priority ration.
- [ ] On a fixture where every proposal still fits at `holding_rate = 0.01` and the till empties at `0.05`, a frozen Policy places a different order at the higher rate, without a refit.
- [ ] The RL stack is not imported.
