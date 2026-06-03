# 03 — `episode_sampler` pure-function module

Status: ready-for-agent

## Parent

PRD: `.scratch/rl-policy-framework/PRD.md`

## What to build

A pure-function module `src/rl/episode_sampler.py` that turns a catalog plus an `episode_seed` into a fully deterministic `EpisodeSpec` ready to drop into `RLEnv`.

Public surface:

- `EpisodeSpec` — small dataclass holding:
  - `scenario: Scenario` — full Scenario object ready to feed into the env (`n_steps = config.episode_length`, `world_seed` derived from `episode_seed`)
  - `slot_permutation: tuple[int, ...]` — slot index → position in the active subset
  - `active_subset: tuple[str, ...]` — the K product ids selected this episode

- `sample_episode(catalog, base_template, config, episode_seed) → EpisodeSpec`
  - Splits `episode_seed` into independent sub-seeds for (assortment, capacity, balance, world, slot permutation) so any subset can be frozen for ablations.
  - Capacity sampled from `config.capacity_dist` (default `Uniform(150, 400)`).
  - Balance sampled from `config.balance_dist` (default `Uniform(15000, 40000)`).
  - Active subset sampled uniformly without replacement from the catalog.
  - Synthesised `StoreTemplate` has `init_active_products = active_subset`, `init_stock_pct = 0.0`, `init_freshness = "fresh"`.

## Acceptance criteria

- [ ] `src/rl/episode_sampler.py` exports `EpisodeSpec` and `sample_episode`
- [ ] Function is pure and deterministic — no I/O, no module-level state
- [ ] `tests/rl/test_episode_sampler.py` covers:
  - [ ] Determinism — same `episode_seed` → identical `EpisodeSpec` across all fields (capacity, balance, assortment, slot permutation, world_seed)
  - [ ] Assortment coverage — across many seeds, every product in the catalog appears in some assortment
  - [ ] Distribution sanity — sampled capacities and balances fall inside the configured ranges; means within tolerance
  - [ ] Scenario validity — returned `Scenario` round-trips through `Runner` for at least one tick without raising
  - [ ] Seed splitting independence — freezing the assortment sub-seed and varying the rest changes capacity/balance/slot_perm but not the active subset
- [ ] Tests run under `uv run pytest tests/rl/test_episode_sampler.py` and pass

## Blocked by

- `.scratch/rl-policy-framework/issues/01-rl-deps-and-config.md` (consumes `RLConfig`)
