# CentralTable deep module

Status: ready-for-agent

## Parent

`.scratch/multi-echelon/PRD.md`

## What to build

Land the `CentralTable` deep module: a globally visible table of live supplier offers keyed by `(seller_id, pid)`. Sellers publish offers at the start of each tick; the allocator commits decrements live during a phase as buyers execute; buyers snapshot the post-prior-allocation state per product.

This module is the live-state contract of ADR 0012. It is a small interface around rich semantics — mutation happens during a phase, not between ticks.

`fill_rate_recent` is wired as a per-`(seller_id, pid)` rolling EMA that tracks qty-weighted partial fills (window = 10 ticks). The EMA update is exercised by `commit` calls; the actual EMA consumers land in issue 8 (allocation_rng wiring).

This slice is additive — no existing module modified, no existing test touched.

## Acceptance criteria

- [ ] `src/sim/central_table.py` exports `Offer`, `CentralTable`
- [ ] `Offer` carries `available_qty`, `list_price`, `min_order`, `fill_rate_recent`
- [ ] `CentralTable.publish(seller_id, pid, offer)` overwrites the row
- [ ] `CentralTable.commit(seller_id, pid, qty)` decrements `available_qty` live, never below zero, and updates `fill_rate_recent` as qty-weighted EMA (window 10)
- [ ] `CentralTable.snapshot_for_buyer(pid)` returns every `(seller_id, Offer)` for that product reflecting post-commit state
- [ ] `tests/sim/test_central_table.py` covers: publish overwrite, commit decrement, commit-below-zero rejection, EMA tracks qty-weighted partial fills, snapshot reflects post-commit state
- [ ] `uv run pytest tests/sim -x` is green; no existing test modified

## Blocked by

None — can start immediately
