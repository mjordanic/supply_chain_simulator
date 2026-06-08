# 07 — Consumer refactors: tuning + RL drop bespoke collectors

Status: ready-for-agent

## Parent

`.scratch/flow-logged-business-metrics/PRD.md` (ADR 0019)

## What to build

Unify the two duplicated, monkey-patching metric collectors onto the shared run-log-derived path, so
there is one tested code path from run → metrics.

- `src/tuning/rollout.py` deletes its bespoke `_TrackingDemandSinkNode` monkey-patch +
  `_record_active_subset` collector and builds the flow frame from the run log, then calls the
  DataFrame-native metrics (issues 02/04/05).
- `src/rl/eval.py` deletes its two inline `RunSlice` collectors and uses the same builder + metrics.
- Tuning/RL read `holding_rate` / `order_fee` from the scenario (issue 01), not from
  `src/tuning/config.py`.
- **Verify the early dependency:** confirm whether `rl/eval.py` runs through `Runner.run()` (and thus
  already gets a standard run log to build the frame from) or uses a bespoke stepping loop that must
  be routed through the same flow logging. Record the finding in the commit body and handle
  accordingly.

The tuner/RL objective KPIs are the operational metrics, which are value-preserving (issue 04) — this
refactor must not shift them.

## Acceptance criteria

- [ ] `src/tuning/rollout.py` no longer contains a bespoke collector; it builds the flow frame and calls the shared metrics.
- [ ] `src/rl/eval.py` no longer contains inline `RunSlice` collectors; it uses the same path.
- [ ] Tuning/RL source `holding_rate`/`order_fee` from the scenario.
- [ ] The tuner/RL objective KPI values are unchanged by the refactor (value-preserving).
- [ ] The `rl/eval.py` run-path finding (Runner.run vs. bespoke loop) is documented and the flow logging is wired accordingly.

## Blocked by

- Issue 04 (DataFrame-native operational metrics), issue 05 (profit decomposition).
