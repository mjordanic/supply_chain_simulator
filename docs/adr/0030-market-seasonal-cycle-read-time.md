# Seasonal cycle is a read-time multiplier

Status: Accepted

`Market.update_market` used to add `cycle_amp * sin(2π * step / cycle_len)` into the regional demand level on every tick. The level is clamped to `[min_value, max_value]`, so the sine integrated until the level sat on the ceiling, fell through, and sat on the floor. On the synthetic episode world (`cycle_amp=0.3`, `cycle_len=365`) that dead stretch was most of the episode, and two world seeds produced nearly the same path because both were pinned to the same clamps. `cycle_amp` did not mean a bounded seasonal swing.

**Decision — Keep the demand level a stochastic clamped process, and apply the seasonal cycle only when the demand factor is read.**

The level update is `clamp(level * trend + demand_shock, min_value, max_value)`. No cycle term enters `market_state`. `demand_multiplier` emits the floored level times `1 + cycle_amp * sin(2π * step / cycle_len)`, using `Market.current_step()` at read time (the step `tick()` has already incremented). `EventEngine` still writes disruption shocks onto the stored level; the cycle scales that level afterwards and is not written back.

`MarketParams.cycle_amp` must be a finite float with absolute value below 1. Otherwise `1 + cycle_amp * sin(...)` can be zero or negative. That is the bound that keeps the cycle factor strictly positive. `min_value` may still be 0: `flat_world` uses that as an identity clamp the stable level never hits.

`default_market_params` widens the episode world so the seed shows up in the path: `demand_shock` is `Normal(0, 0.02)` (was `Normal(0, 0.01)`) and `trend` is `Uniform(0.997, 1.003)` (was `Constant(1.0)`, which drew no random number). Supply shock is unchanged. The LLM world-builder math defaults are a separate authoring path and stay as they were; the degenerate episode world was the sampler default. `trend` now consumes `world_rng` on each resample, so CRN identity for a fixed `world_seed` changes for any episode that uses these defaults.

On seeds 0–7 over 365 ticks, the mean pairwise RMS of the demand-level paths is 0.334 (it was 0.161 before the widening). Dividing the calendar factor and the stored level out of `demand_multiplier` leaves `cycle_amp * sin(...)` on every one of those steps. The longest run pinned to either clamp is 3 ticks. The lowest second-half mean of the emitted demand factor is 0.376, against a floor near `min_value * off_factor * (1 - cycle_amp) = 0.098`.

## Consequences

Existing CRN baselines that depended on the integrated cycle are superseded. Reinforcement-learning policies are not retrained, and published RL results are not replaced.

Blast radius, measured by replaying the canonical regression scenario once with the old additive update and once with this correction (the old replay reproduces the previous digest `085cb490…`):

- **Moved:** `tests/sim/test_regression_snapshot.py` (`test_run_log_hash_matches_snapshot`). Digest `085cb4904b59c0dc537762a872c38ea1829062bbea709a86f0dd0f67bbba7d90` → `452af00424353958b87e6d1969baf588171047102aebe7501d1b8e187338c7d1`. On that 30-tick scenario (`cycle_amp=0.0015`) the US demand level moved by mean absolute delta 0.0072 (max 0.0208). Supply did not move. Final cash moved only at the two shops: shop-store1 −0.48 (11666.97 → 11666.49) and shop-store2 −0.24 (11571.96 → 11571.72). Factory cash, sink cash, and ending inventory units did not move.
- **Did not move:** the rest of the suite. `uv run pytest` after the correction and before the digest bump was 1 failed, 1197 passed, 9 skipped. The single failure was the digest above. `tests/rl` and `tests/tuning` stayed green; their CRN checks are paired equality on one seed, not a stored profit. No file under `src/jepa/` changed. No notebook or README sentence described the additive cycle, so those files were left as they were. `CONTEXT.md` and ADR 0015 now point here.

## Cross-references

- ADR 0003 — CRN demand draws are unchanged in count per catalog product; the world stream moves where `trend` newly consumes the RNG
- ADR 0015 — `demand_multiplier` remains the sink's market factor; the cycle is part of that read, not of the stored level
