# 03 — Un-lagged demand signal (`observed_sales`)

Status: done

## Parent

`.scratch/lateral-links-scheduling/PRD.md` (ADR 0018)

## What to build

Now that the tick runs demand-pull (issue 02), every downstream buyer has already transacted by the
time an intermediate decides — so the intermediate can observe **complete current-tick sales** and
post-sale inventory. Remove the one-tick-lag workaround and feed the live signal.

- Accumulate sales during the walk into `Simulation._tick_sales` (per node, per product).
- Inject `observed_sales = self._tick_sales.get(node.id, {})` into the intermediate observation;
  it is guaranteed complete by the demand-pull ordering.
- Rename the read at the two policy sites (`MultiSupplierTextbookPolicy.decide`,
  `_SingleSupplierAdapter.decide`) from `prev_tick_sales` → `observed_sales` and map it into
  `observation["sales"]`.
- Delete the inventory-delta fallback (the demand signal is now exactly current-tick sales, possibly
  zero — never a delivery-confounded proxy).
- Remove the runner's prev-tick capture/swap machinery entirely.

Why this is safe: `execute_buy` decrements the seller's on-hand inventory at sale time, so under
demand-pull a seller's post-sale inventory and complete sales are both known by the time it decides.

## Acceptance criteria

- [ ] The intermediate observation carries `observed_sales` reflecting **current-tick** demand.
- [ ] On a lateral graph, an intermediate reorders in response to the same tick's downstream purchase (end-to-end test).
- [ ] No `prev_tick_sales` capture/swap or inventory-delta fallback remains anywhere in the runner or policies.
- [ ] Both policy read-sites consume `observed_sales` and map it into `observation["sales"]`.
- [ ] Cash conservation holds (ADR 0013) and the run stays bit-deterministic from the seed.

## Blocked by

- Issue 02 (demand-pull walk — `observed_sales` is only complete once the schedule is reverse-topological)
