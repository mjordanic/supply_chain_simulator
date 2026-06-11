# 04 · Dead eval seeds: initial inventory can exceed total episode demand

Status: needs-triage

## Problem

The episode sampler (`src/rl/episode_sampler.py:212`) sets per-product initial
inventory to `max(1, capacity // K)`. Capacity is sampled log-uniform across two
orders of magnitude; when a large capacity is drawn with small K, the node starts so
over-stocked that the entire episode's demand cannot draw inventory position below any
reachable order-up-to target. Then `max(0, target·rate − position)` is 0 for every
possible action: no policy ever orders, prices are inert engine-side (issue 02), and
both CRN-paired arms produce bit-identical runs.

Verified example (notebook 06, eval seed 0): capacity 3,340, K=1, demand 2–8/tick over
30 ticks — 0 buy rows in both arms, identical 262 units sold, paired uplift exactly
0.0. The RL policy *is* invoked (30 calls); its outputs just cannot matter. Notebook
06 seeds 0 and 4 and notebook 06a §5 seed 0 are such "dead seeds".

Consequences: dead seeds contribute exact-0 uplift and a tie to paired eval — they
dilute the mean uplift toward zero and (combined with issue 05) inflate `win_rate`,
while measuring nothing about policy quality. They also waste training episodes on
worlds where no action has any effect.

## Fix options

1. Scale initial stock to expected demand instead of capacity — e.g. a few lead-times
   of `base_demand_prior` per product, capped by `capacity // K`.
2. Filter or flag dead seeds in `build_eval_seeds` / `evaluate` (e.g. drop seeds whose
   initial inventory exceeds a bound on total episode demand, or report them as a
   separate `n_dead` count).

Option 1 changes the training distribution too (probably a good thing — fewer no-op
episodes); option 2 is eval-only and non-breaking.

## Notes

- Prior art: `build_eval_seeds` / `RLEpisodeSpec` (`src/rl/eval.py`),
  `sample_episode` (`src/rl/episode_sampler.py`).
- Cross-reference: notebook 06 "Reading the result honestly" and 06a §5 now explain
  the exact-0.0 artifact; issue 05 (win_rate ties).
