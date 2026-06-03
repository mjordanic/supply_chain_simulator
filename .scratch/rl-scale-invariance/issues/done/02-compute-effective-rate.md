# 02 — `compute_effective_rate` rate primitive

Status: ready-for-agent

## Parent

PRD: `.scratch/rl-scale-invariance/PRD.md`
ADR: `docs/adr/0007-rl-scale-invariance-package.md`

## What to build

A pure function in `src/rl/encoders.py` that returns the per-product effective sales rate consumed by both the observation encoder (slot 13, slice 5) and the action decoder (qty math, slice 4). Single source of truth for "what rate does the RL stack think this SKU has", so future changes to the rate primitive (EMA smoothing, stockout-bonus inheritance) land in one place and stay consistent across the two consumers.

PRD user story 15: "As an engineer extending the RL stack, I want the demand-rate computation (`effective_rate = max(rolling_5_mean_sales, base_demand_prior)`) to live in one pure-function module consumed by both encoder and decoder, so that future changes to the rate primitive land in one place and stay consistent across the two consumers."

### Signature and contract

```python
def compute_effective_rate(
    sales_history: dict[str, deque],
    base_demand_prior: float,
) -> dict[str, float]:
    """Return effective rate per pid: max(rolling-5-mean of sales[pid], prior).

    Empty or missing history for a pid contributes a value equal to the prior.
    Partial windows (1-4 entries) use the partial-window mean, still clamped
    to the prior as a floor.
    """
```

Stateless, no I/O, no env coupling. Operates on the same per-pid `deque` objects that `RLEnv` already maintains in `self._sales_history`.

The output dict has one key per pid that appears in `sales_history` (which the env populates with every catalog pid at reset, see `RLEnv.reset` `self._sales_history = {pid: deque(maxlen=100) for pid in ...}`).

### Tests added

Append to `tests/rl/test_encoders.py`:

- `test_effective_rate_empty_history_returns_prior` — `compute_effective_rate({"P0001": deque(), "P0002": deque()}, prior=3.0) == {"P0001": 3.0, "P0002": 3.0}`.
- `test_effective_rate_partial_window_uses_partial_mean` — history of `[10, 12]` (window length 2) with prior `3.0` returns `11.0` (the partial-window mean dominates the prior).
- `test_effective_rate_full_window_uses_rolling_5_mean` — history of `[10, 12, 14, 8, 6, 100, 100, 100, 100, 100]` (length 10, deque maxlen ≥ 10) returns the mean of the last 5 entries (`100.0`), not the mean over all 10.
- `test_effective_rate_prior_floors_low_history` — history of `[0, 0, 0, 0, 0]` with prior `2.5` returns `2.5` (prior floors below-prior empirical rate).
- `test_effective_rate_output_keys_match_input` — every key in `sales_history` appears in the output; nothing else does.
- `test_effective_rate_zero_prior_allows_zero_output` — history of `[0, 0]` with prior `0.0` returns `0.0` (no implicit floor beyond the prior; behaviour is fully determined by inputs).

## Acceptance criteria

- [ ] `src/rl/encoders.py` exports `compute_effective_rate` (added to `__all__`).
- [ ] Function signature exactly matches the contract above.
- [ ] Output is a `dict[str, float]` with one entry per key in `sales_history`.
- [ ] All six new tests pass under `uv run pytest tests/rl/test_encoders.py`.
- [ ] No regression in `uv run pytest tests/rl/` or `uv run pytest tests/sim/`.

## Blocked by

None — can start immediately.
