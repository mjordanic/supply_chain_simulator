# 03 · RLEnv observations never contain supplier features (central table is None)

Status: needs-triage

## Problem

`tick_decide_and_settle` clears the central table at the end of the tick
(`self._current_table = None`, `src/sim/runner.py:258`), and
`RLEnv._build_observation` runs after that (and after `reset()`), so the encoder's
supplier features — `ROW_SUPPLIER_COUNT`, `ROW_MIN_PRICE`, `ROW_FILL_RATE` — are
identically 0 in **every training observation**.

At eval/deploy time, `RLNodePolicy.decide()` receives the live `central_table`
mid-tick, so the same features are populated (measured first-decide values on a
starved spec: 0.125 / 0.343 / 1.0). This is a train/eval distribution shift: the
policy trains on a marginal of the obs space it never sees again at eval.

There is also a broader obs-timing shift: the eval-path obs is mid-tick
(post-sink-sales — sales history includes the current tick; sin/cos one tick ahead)
while the training obs is end-of-previous-tick. Measured on the 2026-06-11
investigation, the combined shift currently *favours* eval (same checkpoint scores
higher under eval dynamics than RLEnv dynamics: 87k/85k/70k vs 64k/60k/8.6k cash delta
on three starved seeds), so it does not explain any RL underperformance — but it
should still be closed for parity.

## Fix options

1. Stash the table before `tick_decide_and_settle` clears it, and let
   `RLEnv._build_observation` use the stashed copy.
2. Build the next observation at the start of `RLEnv.step()` before settle, aligning
   the training obs timing with the eval path.

Either way, existing checkpoints were trained on zeroed supplier features; retraining
is needed to benefit.

## Notes

- Prior art: `RLNodePolicy._NodeProxy` / `_MarketProxy` obs construction
  (`src/rl/node_policy.py`) is the eval-side reference for what the features should
  contain.
- Cross-reference: ADR 0022 (one shared decide path was the rebuild's goal; this is
  the residual gap on the training side).
