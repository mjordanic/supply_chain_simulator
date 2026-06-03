# Doc sync — `CONTEXT.md` + `README.md` notebook references

Status: ready-for-agent

## Parent

`.scratch/notebooks-multi-echelon-rewrite/PRD.md`

## What to build

`CONTEXT.md` and `README.md` reference notebooks by their old names/numbers (e.g.
`03-inspect_scenario`, `04a-deep_dive_active_only`, the `01/02/05/06/08` walkthrough links). Once the
suite has been fully renumbered/renamed, update every such reference to the new ten-notebook lineup:

| New notebook |
| --- |
| `00-build-or-load-world` |
| `01-inspect-world` |
| `02-inspect-scenario` |
| `03-run-and-inspect-simulation` |
| `04-deep-dive-per-product` |
| `05-topology-gallery` |
| `06-policy-comparison` |
| `07-tune-textbook-policy` |
| `08-monitor-rl-training` |
| `09-rl-vs-baseline` |

This is a docs/UX change only — no ADR is created. Scope is limited to fixing references that now
point at deleted/renamed files; do not rewrite surrounding prose beyond what the rename requires.

## Acceptance criteria

- [ ] No reference in `CONTEXT.md` or `README.md` points at a deleted/old notebook name or number
- [ ] All notebook references match the new ten-notebook lineup above
- [ ] No new ADR; changes are confined to the stale references (plus minimal wording the rename forces)

## Blocked by

- `.scratch/notebooks-multi-echelon-rewrite/issues/02-finalize-rewritten-notebooks.md`
- `.scratch/notebooks-multi-echelon-rewrite/issues/03-run-and-inspect-simulation.md`
- `.scratch/notebooks-multi-echelon-rewrite/issues/04-deep-dive-per-product.md`
- `.scratch/notebooks-multi-echelon-rewrite/issues/05-topology-gallery.md`
- `.scratch/notebooks-multi-echelon-rewrite/issues/06-policy-comparison.md`
- `.scratch/notebooks-multi-echelon-rewrite/issues/07-monitor-rl-training.md`
- `.scratch/notebooks-multi-echelon-rewrite/issues/08-rl-vs-baseline.md`
