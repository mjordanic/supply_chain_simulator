# 08 — `CONTEXT.md` glossary update

Status: ready-for-agent

## Parent

PRD: `.scratch/rl-scale-invariance/PRD.md`
ADR: `docs/adr/0007-rl-scale-invariance-package.md`

## What to build

Update the **RL Env** glossary entry in `CONTEXT.md` so future readers reaching this code via the glossary understand the design intent of the order-up-to decoder and the demand-units inventory feature without having to dive into the ADR.

PRD user stories: 21 (RL Env paragraph describes new decoder semantics in domain language), 22 (TextbookReorderPolicy cross-reference points at ADR 0007, ADR 0005 marked superseded in the decisions list).

### Changes

**RL Env paragraph (`CONTEXT.md`).** The entry already references ADR 0007 and broadly describes the order-up-to decoder. Confirm the paragraph still reads correctly against the code after slices 4 and 5 ship, and tighten it if any term drifted:

- The decoder's quantity formula is "lead-times of expected demand" via `target_lt × effective_rate − inventory_position`, where `target_lt = clip(target_centre_lt + order_raw × target_half_span_lt, 0, target_max_lt)` and `effective_rate = max(rolling_5_mean_sales, base_demand_prior)`.
- The encoder carries inventory in two frames: capacity-units (slot 0, useful for pricing) and demand-units (slot 13, useful for ordering, saturating at `max_inventory_lt = 30` lead-times of cover).
- Per-episode capacity and balance are sampled from log-uniform distributions spanning two orders of magnitude (`LogUniform(100, 10_000)` capacity, `LogUniform(10_000, 1_000_000)` balance).

**TextbookReorderPolicy paragraph (`CONTEXT.md`).** The cross-reference to ADR 0007 should already be in place from the textbook policy work; confirm it still reads correctly.

**Decisions list (`CONTEXT.md`).** ADR 0005 is already marked superseded; verify the line still reads `[ADR 0005] ... *Superseded by ADR 0007.*` and that ADR 0007 has its own one-line entry.

**Distribution paragraph (`CONTEXT.md`).** Add `LogUniform(lo, hi)` to the list of concrete `Distribution` implementations (`Constant`, `Uniform`, `Normal`, `Choice`, `LogUniform`).

### Format constraints

This is doc-only. No code changes, no test changes. Edits respect the existing `CONTEXT.md` conventions:
- One paragraph per glossary term.
- Domain language ("lead-times of expected demand", "inventory position") rather than implementation detail ("the `decode_action` function").
- Reference ADRs by number, not by file path.

## Acceptance criteria

- [ ] `CONTEXT.md` RL Env paragraph describes the order-up-to decoder in lead-times-of-demand terms, the demand-units inventory encoder feature, and the log-uniform capacity / balance defaults.
- [ ] `CONTEXT.md` Distribution paragraph lists `LogUniform` alongside `Constant` / `Uniform` / `Normal` / `Choice`.
- [ ] `CONTEXT.md` decisions list shows ADR 0005 marked superseded and ADR 0007 with a one-line entry.
- [ ] No code or test changes.

## Blocked by

- `05-encoder-demand-units-feature.md` — the glossary describes shipped behaviour; do not update the doc ahead of the encoder slice landing or the doc and code temporarily disagree.
