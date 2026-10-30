# Binding tightness on replenishment episodes

Status: Accepted

Tuning episodes used to draw `IntermediateNode` capacity from `LogUniform(100, 10_000)` and opening cash from `LogUniform(10_000, 1_000_000)`, independently of demand. Inventory scale varied about 97× and cash about 110× while mean demand per SKU varied about 2.4×. Almost every episode sat in a regime where stock far exceeded demand: service level was 1, neither constraint bound, and a tuned policy had nothing to beat.

**Decision — Tightness is the sampled quantity. Capacity and opening cash are multiples of that episode's demand.**

`sample_episode` draws two factors on the `tightness` sub-seed, after the active subset is known. Capacity tightness is `LogUniform(0.5, 6)`. Cash tightness is `LogUniform(0.02, 0.8)`. Aggregate episode demand is the sum over active SKUs of expected demand per tick times the episode length. Expected demand per tick is the ware's `demand_mean`, or 5 when that field is absent.

The capacity pool is `tightness × aggregate episode demand × (14 / episode_length)`, rounded to an integer of at least 1. At tightness 1 the shelf holds 14 ticks of expected demand, the textbook cover horizon. Opening cash is `cash_tightness × aggregate episode demand valued at unit cost`, with no extra horizon. The factors are separate draws. One shared factor cannot make cash the operative limit on a year-long episode: cash sized to the whole episode buys about twenty times what a 14-tick shelf holds, so the shelf would always set the fill. A buyer that asks for two cover horizons of demand, and can actually be filled by the supplier, is stopped by the shelf, by cash, or by neither, each in a controlled share of episodes.

`TuningEpisodeSpec.tightness` and `cash_tightness` record the factors. `evaluate_policy_normalised` returns them as `per_seed_tightness` and `per_seed_cash_tightness`. `per_seed.parquet` and `holdout.parquet` carry both columns.

`capacity_dist` and `balance_dist` default to `None`. An explicit distribution still overrides that pool. When both are set, both tightness fields are `None` and the old absolute draws are used. RL is unchanged: it still samples log-uniform capacity and balance (ADR 0007).

This replaces ADR 0009 decision (d) for the tuning episode only.

## Consequences

A fixed `episode_seed` still rebuilds the same pools. The draws do not consume `world_rng`. `net_profit / initial_cash` stays the objective; the per-seed tightness columns are how a result is read against how binding the episode was. A small cash tightness makes the budget short of a full episode of purchases at `list_price`. A small capacity tightness makes the shelf short of a cover horizon.

## Cross-references

- ADR 0007 — RL keeps log-uniform capacity and balance
- ADR 0009 — Decision (d) no longer describes the tuning episode
- ADR 0031 — Supplier markup is why unit-cost cash is tighter than a full episode of purchases
