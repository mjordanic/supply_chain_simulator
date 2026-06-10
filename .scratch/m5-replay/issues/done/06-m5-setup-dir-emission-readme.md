# 06: M5 setup-dir emission + README

Status: done

## Parent

`.scratch/m5-replay/PRD.md` (see also `.scratch/m5-replay/STUDY.md`)

## What to build

The emission half of the M5 adapter: turn an ingested slice (issue 05) into a complete,
loadable replay setup directory (issue 03 format) — raw Kaggle files in, runnable scenario
out, one call.

- **Catalog derivation**: `base_price` = median observed price over the slice per item;
  `unit_cost` = documented default fraction of `base_price`, author-overridable; M5 dept →
  `Ware.category`; M5 state → node `region`; `Ware.seasonality` left default.
- **Replay sinks**: one per (item, store) series, emitted as `sink_replay` nodes wired to
  their store's shop node; `start_date` from the calendar, `n_steps` = window length.
- **Sink economics**: sink `income_rate` set so the affordability cap can never bind
  (the sim must not silently censor replayed demand).
- **Pass-through artifacts**: daily prices parquet (for the issue 04 wrapper), calendar
  parquet (events, SNAP — preserved for future calibration, not wired to the engine), and
  the issue 05 quality report persisted alongside them (its launch-date column is the
  on-disk home of launch dates — PRD "preserved but not wired").
- **README** (user requirement, committed): lives next to the adapter in the datasets
  package — NOT inside the data folder, which is blanket git-ignored (`data/` in
  `.gitignore`), so a README there would be silently untracked. Explains what the adapter
  is, that raw files must be downloaded from Kaggle by the user (git-ignored,
  redistribution rights unclear), exactly where to place them, how to run the adapter, and
  points to the example notebook. States the "observed sales = true demand" declaration.
- Git-ignore coverage for raw and emitted M5 data. Note `data/` is already
  blanket-ignored; add entries only for any M5 path that lands outside it.

Note: the emitted topology covers shops + replay sinks + the edges between them; upstream
(DC, factory) remains the scenario author's job — the adapter must make it easy to attach
the emitted parts to an author-supplied upstream (the example notebook, issue 07, does
exactly this).

## Acceptance criteria

- [ ] One call takes raw-file directory + slice spec → emitted setup dir loadable by `load_setup`
- [ ] Emitted scenario runs end-to-end and replays demand exactly under a flat world (fixture-based test)
- [ ] Catalog fields derived as specified; `unit_cost` fraction documented and overridable
- [ ] No `insufficient_cash` rejections at sinks in a fixture-based run (income_rate non-binding by construction)
- [ ] Prices and calendar parquets emitted alongside; events/SNAP not wired to the event engine
- [ ] Quality report persisted in the emitted setup dir alongside the other artifacts
- [ ] README committed next to the adapter (datasets package, not the git-ignored data folder) covering purpose, Kaggle download instructions, file placement, adapter usage, notebook pointer
- [ ] Raw and emitted M5 data paths are git-ignored

## Blocked by

- `03-setup-dir-replay-serialization.md`
- `05-m5-ingestion-quality-report.md`
