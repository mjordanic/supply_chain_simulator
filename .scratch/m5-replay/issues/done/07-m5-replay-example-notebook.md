# 07: Example notebook — multi-echelon CA replay

Status: done

## Parent

`.scratch/m5-replay/PRD.md` (see also `.scratch/m5-replay/STUDY.md`)

## What to build

The end-to-end demonstration notebook: raw M5 files → adapter → quality report →
multi-echelon scenario → run → verified exact replay. The public face of the feature and
its living documentation.

Flow:

1. Pointer to download instructions (the issue 06 README); graceful early exit with a
   clear message if raw files are absent (the notebook must not assume committed data).
2. Run the adapter on a chosen slice (a few CA stores, one dept, fast movers over a
   ~1-year window); render the quality report and discuss what it shows (zero runs,
   fill counts, the "observed sales = true demand" caveat).
3. Build the multi-echelon example: author an invented upstream (factory → CA DC) in
   front of the emitted CA shops + replay sinks; invented lead times/costs are explicit,
   labeled scenario knobs.
4. Run with a textbook policy at the shops; **assert** exact replay (per-tick sink
   consumption == series) and zero `insufficient_cash` sink rejections.
5. Evaluate: service level / lost sales / costs of the policy against real demand,
   using the rejection log and run-log metrics.
6. Event-overlay plots: demand series with calendar events/SNAP markers overlaid (from
   the calendar parquet — visualization only, not wired to the engine).
7. Optional variant cell: wrap the shop policy in the price-replay wrapper (issue 04)
   and show observed prices flowing into offers/revenue.

## Acceptance criteria

- [ ] Notebook runs top-to-bottom against locally downloaded raw M5 files; exits early with clear instructions when files are absent
- [ ] Inline assertions pass: exact replay under flat world; zero sink `insufficient_cash` rejections
- [ ] Demonstrates the (ii) topology: invented factory + CA DC → emitted CA shops → replay sinks, with invented params labeled as such
- [ ] Quality report rendered and interpreted; censoring caveat stated in prose
- [ ] Event-overlay plot present; price-replay variant cell present
- [ ] Notebook is demonstration, not test suite — no CI dependency on real M5 data

## Blocked by

- `04-price-replay-policy.md`
- `06-m5-setup-dir-emission-readme.md`
