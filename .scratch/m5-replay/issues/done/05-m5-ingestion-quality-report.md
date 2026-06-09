# 05: M5 ingestion + quality report

Status: done

## Parent

`.scratch/m5-replay/PRD.md` (see also `.scratch/m5-replay/STUDY.md` §"The dataset")

## What to build

The data half of the M5 adapter, in a sibling package the sim core never imports
(ADR 0010 pattern): parse the three raw M5 Kaggle files (daily sales, weekly
`sell_prices`, `calendar`), select a slice (items × stores × date window), and produce
analysis-ready per-(item, store) daily series.

- **Weekly→daily price expansion** via the calendar's `wm_yr_wk` ↔ date mapping; consumers
  downstream never see Walmart week numbering.
- **Price gap filling**: forward-fill gaps, back-fill leading gaps (pre-launch weeks);
  fill counts are recorded, never silent.
- **Quality report**: per-item DataFrame with price coverage %, ffill/bfill counts,
  zero-sale-day share, longest zero run (suspected out-of-stock flag), and launch date —
  the tool for judging slice quality under the declared "observed sales = true demand"
  semantics.

Tests run on small hand-built fixture files shaped like the M5 schema — never real M5
data (raw files are git-ignored and not redistributable).

Setup-dir emission is issue 06; this slice is demoable as: fixture files in → slice +
quality report out.

## Acceptance criteria

- [ ] Parses sales, prices, and calendar files of the M5 schema from a directory of raw files
- [ ] Slice selection by item ids, store ids, and date window; tick 0 = window start
- [ ] Weekly prices expand to correct daily arrays across Walmart week boundaries (fixture-verified)
- [ ] Gap ffill and leading bfill behave as specified, with counts surfaced in the quality report
- [ ] Quality report contains all specified per-item fields and flags long zero runs
- [ ] All tests use hand-built fixtures; no real M5 data in the repo or test suite

## Blocked by

None - can start immediately
