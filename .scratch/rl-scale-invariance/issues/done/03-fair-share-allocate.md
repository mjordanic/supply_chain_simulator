# 03 — `fair_share_allocate` capacity allocator

Status: done

## Parent

PRD: `.scratch/rl-scale-invariance/PRD.md`
ADR: `docs/adr/0007-rl-scale-invariance-package.md`

## What to build

A pure function in `src/rl/encoders.py` implementing the two-pass fair-share capacity allocator the order-up-to decoder will use (slice 4). Mirrors the textbook policy family's allocator (`TextbookReorderPolicy._allocate_*` in `src/sim/policy.py`) at the clamp boundary so the RL decoder and the CRN comparison anchor allocate identically — which is where they differ most on the cold-start tick.

PRD user story 16: "As an engineer extending the RL stack, I want the two-pass fair-share allocator to live in one pure-function module, so that the same allocator can be reused if the encoder ever needs a clamp-aware feature or if the decoder grows additional constraints."

The PRD's architectural note: this is capacity-only — no cash-budget logic. The textbook policy's allocator bundles the cash pool because the textbook policy enforces cash as a hard constraint; the RL decoder exposes cash to the agent's reward signal instead, so the two implementations diverge on purpose. Duplicating ~20 lines of capacity-only logic is preferable to coupling to the textbook policy's broader allocator.

### Signature and contract

```python
def fair_share_allocate(
    requested: dict[str, float],
    per_sku_headroom: dict[str, int],
    global_free_space: int,
) -> dict[str, int]:
    """Two-pass capacity allocator.

    Pass 1: cap each request at the per-SKU physical headroom
            (capped[pid] = min(requested[pid], per_sku_headroom[pid])).
    Pass 2: if sum(capped) > global_free_space, scale every capped
            value proportionally by global_free_space / sum(capped).
    Return integer quantities (truncate, do not round up — under-allocation
    is safer than over-allocation against a hard capacity constraint).
    """
```

Stateless, no I/O. Returns one entry per key in `requested`.

### Tests added

Append to `tests/rl/test_encoders.py`:

- `test_fair_share_empty_input_returns_empty` — empty `requested` returns empty dict.
- `test_fair_share_sum_below_free_space_returns_capped_unchanged` — when `sum(min(requested, headroom)) <= global_free_space`, output equals `floor(min(requested, headroom))` per pid (no proportional scaling fires).
- `test_fair_share_sum_above_free_space_scales_proportionally` — when `sum(requested) > global_free_space` and headroom is non-binding, output sums to ≤ `global_free_space` and preserves the relative proportions of the requested amounts within `±1` (integer truncation tolerance).
- `test_fair_share_per_sku_headroom_binds_tighter_than_global` — request of `{P1: 100, P2: 100}`, headroom of `{P1: 10, P2: 1000}`, `global_free_space = 200`: P1 is capped to 10 by headroom (not by proportional scaling), P2 receives the residual.
- `test_fair_share_zero_global_free_space_returns_all_zeros` — `global_free_space = 0` returns `{pid: 0}` for every pid.
- `test_fair_share_total_never_exceeds_free_space` — across a randomised set of input shapes, `sum(output.values()) <= global_free_space` is always true.
- `test_fair_share_per_sku_never_exceeds_headroom` — across a randomised set of input shapes, `output[pid] <= per_sku_headroom[pid]` is always true.
- `test_fair_share_output_keys_match_requested_keys` — every key in `requested` appears in the output; nothing else does.

## Acceptance criteria

- [ ] `src/rl/encoders.py` exports `fair_share_allocate` (added to `__all__`).
- [ ] Function signature exactly matches the contract above.
- [ ] Output is a `dict[str, int]` with one entry per key in `requested`.
- [ ] Returned quantities respect per-SKU headroom (`output[pid] <= per_sku_headroom[pid]`).
- [ ] Total returned quantity never exceeds `global_free_space`.
- [ ] All eight new tests pass under `uv run pytest tests/rl/test_encoders.py`.
- [ ] No regression in `uv run pytest tests/rl/` or `uv run pytest tests/sim/`.

## Blocked by

None — can start immediately.
