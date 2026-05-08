# `LifecycleClock` + `dead` stage + per-stage transition probabilities

Status: done

## Parent

`.scratch/lifecycle-rosters/PRD.md`

## What to build

Replace the scalar `stage_change_prob` with a per-stage `default_stage_change_probs: dict[str, float | Distribution]` table on `ItemLifecycleParams` and add a `dead` stage to the canonical lifecycle list `[introduction, growth, maturity, decline, dead]`. `dead` carries `stage_multipliers["dead"] = 0.05` (trickle demand).

Extract `LifecycleClock.advance_stage(rng, current_stage, stage_change_probs) -> next_stage` as a pure function. It encodes: "draw once against the probability for the *current* stage; if it fires, advance to the next stage in the canonical list; from `dead`, the next stage is `introduction` (cyclic-with-terminal-default — set `dead → introduction = 0` to disable)".

`ItemRegistry` reads per-`Ware` overrides for `init_stage` and `stage_change_probs`, falling back to the defaults, and delegates per-tick advancement to `LifecycleClock`. The old `Item.advance_stage` `decline → introduction` cycle is removed; re-launches now happen only via the explicit `dead → introduction` transition.

Default `stage_change_probs["dead"] = 0.003` per tick (~1-year mean comeback).

Existing example scenarios author `stage_change_prob = 0.0`; after migration they set every per-stage entry to `0.0` and run identically (no transitions).

## Acceptance criteria

- [ ] `LifecycleClock.advance_stage` exists as a pure function (no `Store` / `Market` / `Registry` wiring) in its own module
- [ ] `[introduction, growth, maturity, decline, dead]` is the canonical stage list on `ItemLifecycleParams`
- [ ] `ItemLifecycleParams.default_stage_change_probs: dict[str, float | Distribution]` replaces `stage_change_prob: float | Distribution`
- [ ] `ItemLifecycleParams.stage_multipliers["dead"] == 0.05`
- [ ] `Ware.init_stage: str | None` and `Ware.stage_change_probs: dict[str, float | Distribution] | None` added; both optional (fall back to `ItemLifecycleParams.default_*`)
- [ ] `ItemRegistry.tick()` delegates per-item stage advancement to `LifecycleClock.advance_stage`, drawing from `world_rng`
- [ ] `Item.advance_stage` no longer cycles `decline → introduction` — re-launches happen only through `dead → introduction`
- [ ] `Scenario.to_json()` / `from_json()` round-trips the per-stage `stage_change_probs` dict (both default and per-`Ware` override) with structural and numerical equality
- [ ] `LifecycleClock.advance_stage` unit tests pin: `prob = 1.0` ⇒ advance every tick; `prob = 0.0` ⇒ never advance; from `dead` with `prob = 1` returns `introduction`; from `dead` with `prob = 0` stays in `dead`; cyclic ordering matches `[introduction, growth, maturity, decline, dead, introduction, ...]`
- [ ] Existing example scenarios migrated to set every per-stage entry to `0.0` and run with bit-identical lifecycle behavior to before (no transitions)
- [ ] Existing CRN/regression tests pass

## Testing notes

TDD `LifecycleClock` as a pure-function unit. One behavior per test (advance, no-advance, dead→intro cycle, terminal-when-zero, cyclic ordering). Keep `ItemRegistry` tests integration-style — exercise via `tick()` and observe the resulting stages, don't mock `LifecycleClock`.

## Blocked by

None - can start immediately
