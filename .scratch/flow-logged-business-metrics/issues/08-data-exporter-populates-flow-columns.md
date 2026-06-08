# 08 — DataExporter populates graph-mode flow columns

Status: ready-for-agent

## Parent

`.scratch/flow-logged-business-metrics/PRD.md` (ADR 0019)

## What to build

Complete the exported table: graph-mode runs currently write the legacy store-mode flow columns
(`sales` / `demand` / `price` / `revenue`) as `None` because the data did not exist. With the flow
log (issue 02) it now does — populate them from the flow log so the exported `timeseries.parquet` is
complete.

- `DataExporter` fills the previously-null graph-mode flow columns from the per-tick flow log.
- The exported values reconcile with the flow frame the metrics consume (one source of truth).

## Acceptance criteria

- [ ] Graph-mode `timeseries.parquet` has non-null `sales` / `demand` / `price` / `revenue` columns sourced from the flow log.
- [ ] The exported flow columns reconcile with the flow frame (parquet parity test, e.g. `test_data_exporter_parquet_parity.py`).
- [ ] The existing inventory reconciliation in `test_saved_run_roundtrip.py` stays green.

## Blocked by

- Issue 02 (per-tick flow log + DataFrame builder).
