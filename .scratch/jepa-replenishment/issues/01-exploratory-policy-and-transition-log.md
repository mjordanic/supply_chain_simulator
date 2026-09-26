# 01: Exploratory Policy and synthetic transition log

**What to build:** A replenishment Policy that can be attached to the tuner graph's single intermediate and that places base-stock orders from a cover in lead times of recent demand, then rations scarce capacity and cash by priority. Prices stay at `base_price`. The same policy seed repeats the same orders. A collector uses that Policy for half of its episodes and published-default `OrderUpToPolicy` for the other half, and writes a per-SKU transition log on the tuner episode shape. Factories stay on `StaticFactoryPolicy`. The PPO stack is not imported.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

## Parent

`.scratch/jepa-replenishment/PRD.md`

## Stories

PRD stories 1–11, 30–33, 64, 67, 69, 70. Story 10's M5 graph is ticket 05. Story 11's factory on the M5 graph is ticket 05. This ticket owns the synthetic graph.

## Prior art

`OrderUpToPolicy` and the textbook censored-sales rate. `_allocate_two_pass_fair_share` stays the textbook ration and is not the learned ration. `evaluate_policy_normalised` and the tuner episode shape (`K_active` 5, 365 ticks, `delivery_lag` 3, `holding_rate` 0.01). Seed offsets on `TuningConfig`: search `12_000_000`, holdout `13_000_000`. This log uses `14_000_000`. Business metrics `net_profit`. Determinism tests that pin `policy_rng` apart from `world_rng`.

## Artifact homes

- Package: `src/jepa/` (tracked). This ticket adds the behavior Policy and the log collector.
- Published log: `runs/jepa-replenishment/log/` (gitignored with `runs/`).

## Acceptance criteria

- [ ] Attached with `policy_overrides` on the tuner graph, the exploratory Policy orders every tick from a cover in `{0, 1, 2, 4}` lead times of the textbook censored-sales rate, with position equal to on-hand plus in-transit.
- [ ] Before the ration, the quantity is `max(0, round(cover × rate × delivery_lag) − position)`.
- [ ] `list_price` equals `base_price` for every product. Promotions are empty. The minimum-order floor is zero.
- [ ] Granted quantities fit in free capacity and cash. A two-SKU fixture with room for one cover can give the whole pool to the higher priority.
- [ ] The same `policy_seed` and world seed place the same orders. Changing only the world seed does not change the proposal sequence.
- [ ] A collected log is half published-default `OrderUpToPolicy` (fair-share unchanged) and half the exploratory Policy.
- [ ] The published collector uses 365 ticks, five active SKUs, lead time 3, `holding_rate` 0.01, the tuner default order fee, 64 episodes of each behavior, and seed offset `14_000_000`. Smoke tests may pass a shorter episode and a smaller count, and they check the mix and the schema rather than the published count.
- [ ] Each row carries the per-SKU window (on-hand, in-transit, a sales window of length `delivery_lag`, cash, capacity), the granted quantity, and the next on-hand, in-transit, and sales.
- [ ] Factories on these episodes stay on `StaticFactoryPolicy`. The RL stack is not imported.
