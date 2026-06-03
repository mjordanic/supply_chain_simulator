# Tuning re-integration: rollout, search spaces, evaluator

Status: ready-for-agent

## Parent

`.scratch/multi-echelon/PRD.md`

## What to build

Hyperparameter tuning of `MultiSupplierTextbookPolicy` on the graph engine. Tuning contract preserved (Optuna, paired CRN seeds, parquet artifacts).

**Changes:**
- `src/tuning/rollout.py::run_policy_episode`: `spec.scenario` is a graph; policy attached to the trainable intermediate by node id
- `_record_active_subset` collects from the intermediate node (sales, demand, inventory, price, msrp, revenue, holding_cost, order_cost, order_fee — field set identical to today)
- `src/tuning/search_spaces.py`: add `per_supplier_min_order_floor: int` and `routing_strategy: Categorical(["cheapest_first", "fill_rate_weighted"])` to each factory
- `src/tuning/evaluator.py`: per-trial CRN tuple expanded to include the `allocation` sub-seed so paired comparisons across trials remain bit-identical on the world stream

**Tests rewritten:** `tests/tuning/test_evaluator.py`, `test_search_spaces.py`.

**Verification:** `uv run pytest tests/tuning`. `uv run python -m src.tuning.study --name smoke --trials 5` produces `runs/tuning/smoke/{trials.parquet, per_seed.parquet, study.json}`.

## Acceptance criteria

- [ ] `src/tuning/rollout.py::run_policy_episode` runs against a graph-shaped scenario; policy attached by node id
- [ ] `_record_active_subset` collects the same field set from the intermediate node as today's `Store`
- [ ] `src/tuning/search_spaces.py` adds `per_supplier_min_order_floor` and `routing_strategy` Categorical to each factory
- [ ] `src/tuning/evaluator.py` CRN tuple expanded with the `allocation` sub-seed
- [ ] `tests/tuning/test_evaluator.py` rewritten and green
- [ ] `tests/tuning/test_search_spaces.py` rewritten and green
- [ ] `uv run pytest tests/tuning` is green
- [ ] `uv run python -m src.tuning.study --name smoke --trials 5` produces `runs/tuning/smoke/{trials.parquet, per_seed.parquet, study.json}`

## Blocked by

- `.scratch/multi-echelon/issues/11-retire-legacy-store-engine.md`
