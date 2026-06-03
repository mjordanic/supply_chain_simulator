# 01 — `TextbookReorderPolicy` base + helpers + `OrderUpToPolicy`

Status: ready-for-agent

## Parent

PRD: `.scratch/textbook-reorder-policies/PRD.md`
ADR: `docs/adr/0006-textbook-reorder-policy-family.md`

## What to build

The foundation of the textbook reorder-policy family — additive only. No existing class is renamed or removed in this slice; `BaselinePolicy` continues to work untouched. A new public abstract base `TextbookReorderPolicy(Policy)` and one concrete subclass `OrderUpToPolicy` (the (s,S) continuous-review policy that will later become the canonical CRN comparison anchor) are added to `src/sim/policy.py`. Two deep helpers are extracted as private module-level functions so they can be unit-tested in isolation.

End-to-end demoability: after this slice, a developer can write

```python
from src.sim.policy import OrderUpToPolicy
policy = OrderUpToPolicy(policy_seed=0)
```

and attach it to a Store inside a Scenario; the resulting trajectory runs end-to-end through `Runner.run()` without error and finishes a 180-tick episode at flagship scale (`capacity=10_000`, `balance=$1_000_000`) with non-negative net P&L on the LLM fashion world.

### New abstract base

`TextbookReorderPolicy(Policy)` — public, abstract by virtue of two `@abstractmethod` hooks:

- `_trigger(self, pid, step, position, rate) -> bool` — should this SKU reorder this tick?
- `_quantity(self, pid, position, rate) -> int` — if triggered, how much?

Constructor kwargs (inherited by all four subclasses, but only `OrderUpToPolicy` ships here):

- `policy_seed: int | None = None` — for `policy_rng` symmetry with `HeuristicPolicy`.
- `cover_horizon_ticks: int = 10` — `S = (delivery_lag + safety_lead_ticks + cover_horizon_ticks) × rate`.
- `safety_lead_ticks: int = 2` — `s = (delivery_lag + safety_lead_ticks) × rate`.
- `opening_budget_pct: float = 0.50` — pilot order budget as a fraction of opening cash, split evenly across the K active SKUs.
- `stockout_safety_bonus_ticks: int = 0` — opt-in. When `> 0`, the policy bumps `safety_lead_ticks` for any pid whose recent window contains stockout ticks. Default off (textbook-pure).
- `min_qty: int = 0` — order-qty floor. Default `0` (textbook-pure).

The `decide(observation)` method on the base implements the shared pipeline:

1. Read `obs["active_products"]` as the universe.
2. Pilot pass: for any active pid never observed (no row in `sales_log`), schedule a pilot qty `int(opening_budget_pct × balance / K_active / unit_cost[pid])`, clamped to free space. Pilot pids bypass `min_qty`.
3. For every active pid: compute `position = inventory[pid] + pending[pid]` and `rate = _estimate_rate(...)`.
4. For every active pid not in the pilot set: call `self._trigger(pid, step, position, rate)`; if true, request `desired[pid] = max(0, self._quantity(pid, position, rate))`.
5. Run `_allocate_two_pass_fair_share(desired, unit_costs, free_space, cash, min_qty, pilot_pids)` to produce a feasible allocation under capacity and cash pools.
6. Emit the action:
   - `order` = non-zero entries from the allocator.
   - `price[pid] = obs["base_prices"][pid]` for every pid in `obs["active_products"]` (flat, no cost floor).
   - `activate = []`, `deactivate = []`, `promotions = {}`.

### New concrete subclass

`OrderUpToPolicy(TextbookReorderPolicy)` — (s,S) continuous review.

- `_trigger(pid, step, position, rate)` returns `position < s` where `s = (delivery_lag + safety_lead_ticks) × rate`.
- `_quantity(pid, position, rate)` returns `int(round(S − position))` where `S = (delivery_lag + safety_lead_ticks + cover_horizon_ticks) × rate`.

No additional kwargs beyond those on the base.

### New private helpers

`_estimate_rate(sales_log_view, demand_window, inv_before_settle_window, stockout_safety_bonus_ticks) -> float`
- Statistic: censored sales over the last `demand_window` ticks (default `delivery_lag`).
- Aggregator: simple mean.
- Returns `0.0` when the window is empty.
- When `stockout_safety_bonus_ticks > 0` and the window contains stockout ticks (`sales[pid] == inv_before_settle[pid]`), the caller is expected to add the bonus to `safety_lead_ticks` for that pid; the estimator itself returns the unbiased mean. (Bonus application lives on the base class.)

`_allocate_two_pass_fair_share(desired, unit_costs, free_space, cash, min_qty, pilot_pids) -> dict[pid, int]`
- Pass 1 fair-share: each active SKU gets `min(desired, space_block, cash_block_in_qty)` where pools are split per-SKU. Iteration-order-independent.
- Pass 2 water-filling: redistribute leftover space/cash to SKUs whose shortfall > 0. Bounded to ≤ K rounds.
- Integer mop-up: greedy pass over residual shortfalls collects rounding tail (≤ K-1 units).
- `min_qty` floor applies only to non-pilot allocations after both passes. Pilots bypass it.

### Tests added

- `tests/sim/test_textbook_helpers.py` — unit tests for the two extracted helpers. See "Acceptance criteria" for the invariants list.
- `tests/sim/test_textbook_policy.py` — integration tests driving a real `Store`/`Market`/`ItemRegistry` stack:
  - `test_order_up_to_pilot_fires_on_tick_zero` — fresh Store with `init_stock_pct=0.0` places a pilot order on tick 0 sized from `opening_budget_pct`.
  - `test_order_up_to_oscillates_between_s_and_S` — under constant-demand market for `4 × delivery_lag` ticks, position oscillates inside `[s, S]` after warmup.
  - `test_order_up_to_no_order_when_position_above_s` — no order fires while `position > s`.
  - `test_order_up_to_flagship_scale_is_profitable` — full 180-tick episode on `capacity=10_000`, `balance=$1_000_000` LLM-fashion-world scenario finishes with `final_balance >= initial_balance` (smoke test for the comparison-anchor fitness claim from the PRD).
  - `test_order_up_to_crn_self_consistency` — two `OrderUpToPolicy()` instances on the same `(world_seed, init_seed, capacity, balance, active subset, slot permutation)` produce bit-identical trajectories (CRN/RNG-isolation check).

`BaselinePolicy`, `HeuristicPolicy`, and the existing tests are not touched in this slice.

## Acceptance criteria

- [ ] `src/sim/policy.py` exports `TextbookReorderPolicy` and `OrderUpToPolicy` (added to `__all__`).
- [ ] `TextbookReorderPolicy` is an abstract class — instantiating it directly raises `TypeError`.
- [ ] `OrderUpToPolicy()` instantiates with no arguments and has the documented kwarg defaults.
- [ ] `OrderUpToPolicy` consumes only `policy_rng` — never `world_rng` or `init_rng` (CRN-disjoint).
- [ ] `_estimate_rate` and `_allocate_two_pass_fair_share` exist as private module-level functions in `src/sim/policy.py`.
- [ ] `tests/sim/test_textbook_helpers.py` covers:
  - [ ] `_allocate_two_pass_fair_share` pass-1 fair-share allocation is iteration-order-independent.
  - [ ] Capacity-binding: when `sum(desired) > free_space`, total allocated equals `free_space` (modulo integer rounding ≤ K-1).
  - [ ] Cash-binding: same property for the cash pool.
  - [ ] Water-filling: when one SKU asks for more than its share and others ask for less, leftover redistributes to the over-share SKU up to its desired qty.
  - [ ] Integer mop-up: a synthetic case with pass-2 rounding slack is cleaned by greedy mop-up.
  - [ ] Edge cases: `K=1`, all-zero desired, `min_qty` floor on non-pilot SKUs, pilot SKUs bypass `min_qty`.
  - [ ] `_estimate_rate` returns `0.0` for an empty window.
  - [ ] `_estimate_rate` returns the constant for constant-demand history.
  - [ ] `_estimate_rate` with `stockout_safety_bonus_ticks > 0` and a stockout-marked window surfaces the bonus signal correctly (exact contract spelled out in the test).
- [ ] `tests/sim/test_textbook_policy.py` covers the five `OrderUpToPolicy` integration tests listed above.
- [ ] `uv run pytest tests/sim/test_textbook_helpers.py tests/sim/test_textbook_policy.py` passes.
- [ ] `uv run pytest tests/sim/` (full sim suite) still passes — no regressions to existing `BaselinePolicy` tests.

## Blocked by

None — can start immediately.
