# Documentation sweep: RL reference, README, CONTEXT.md, ADR status

Status: ready-for-agent

## Parent

`.scratch/variable-k-rl/PRD.md`

## What to build

Bring all prose documentation in line with the implemented variable-K stack. The RL reference
doc is heavily coupled to the old layout (fixed `K_active`, the 18-slot feature table,
slot-shuffle, `2 * K_active` actions, the 6-element CRN tuple, bare state-dict reload snippet)
and must be rewritten, not patched.

- **RL reference (`src/rl/README.md`)**: new observation layout table (live per-SKU features,
  broadcast globals, contention aggregates, mask), `(K_max, 3)` action layout with the priority
  head, Arbiter section (both variants, cash budget, default), variable-K episode sampling,
  DeepSets critic and masked joint Gaussian, reduced CRN tuple, self-describing checkpoint
  format with a corrected reload snippet, updated CLI flags (K-range / `K_max` / arbiter mode /
  cash-budget fraction replacing the fixed-K knob), and the K-generalisation recipe. Remove the
  slot-shuffle and frozen-assortment explanations; point to ADR 0021 for provenance.
- **Top-level README**, RL section: refresh the stack description (one policy, any catalog size
  up to `K_max`; implicit assortment; deterministic Arbiter) without growing it much.
- **CONTEXT.md**: flip the "(planned, variable-K redesign)" annotations on the Arbiter,
  Implicit assortment, Active subset, and CRN-paired eval entries to present tense; keep the
  glossary meanings intact.
- **ADR 0021**: Status proposed → accepted (user-approved).
- **TODO.md §9**: mark the design-notes section as implemented with a pointer to ADR 0021 and
  the PRD (do not delete the notes; they hold the deferred coordination upgrades referenced by
  the ADR).

No code changes. Verify documented commands/flags against the actual CLI by running `--help`,
and the layout table against the encoder's named constants.

## Acceptance criteria

- [ ] No documentation anywhere in the repo still describes slot-shuffle, fixed `K_active` shapes, the 18-slot row, or the 6-element CRN tuple as current behaviour (grep-verified; ADR history may of course still describe them as superseded decisions).
- [ ] `src/rl/README.md` feature/action tables match the encoder's layout constants one-to-one.
- [ ] Reload snippet in the RL reference uses the checkpoint I/O module and works as written.
- [ ] Documented CLI flags match the actual train/eval entry points (`--help` output).
- [ ] CONTEXT.md planned-annotations flipped; ADR 0021 status accepted.
- [ ] No source code modified.

## Blocked by

- `07-eval-rewire.md` (documents the final shipped surface)
