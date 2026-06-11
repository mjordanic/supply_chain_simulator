# 05 · `eval/win_rate` counts ties as wins

Status: needs-triage

## Problem

`_aggregate_paired` (`src/rl/eval.py:373`) computes

```python
win_rate = sum(1 for u in paired_uplift if u >= 0) / n
```

The `>=` counts exact-tie seeds (uplift 0.0) as wins. Combined with the dead seeds of
issue 04, this materially misreports results: notebook 06's logged `win_rate` of 0.33
is 2 dead-seed ties and **0 real wins** out of 6 seeds.

## Proposed fix

Use strict `u > 0`, or report ties separately (e.g. `eval/win_rate` with `>` plus an
`eval/tie_rate`). Reporting ties separately is the more informative option given dead
seeds exist; strict `>` is the one-character fix.

Notebook prose (06 "Reading the result honestly") now documents the current behaviour;
update it when the metric changes, and note that historical TensorBoard `eval/win_rate`
scalars were computed with `>=`.

## Notes

- Prior art: `_aggregate_paired` and its tests under `tests/rl/` (eval-harness tests
  rebuilt in the `rl-eval-parity` work, ADR 0022).
- Cross-reference: issue 04 (dead seeds are the source of the ties).
