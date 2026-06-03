# Notebooks 01–03 + 08 rewritten, CONTEXT.md updated, ADRs 0011–0016 ratified

Status: ready-for-agent

## Parent

`.scratch/multi-echelon/PRD.md`

## What to build

Final documentation pass. After this slice, the simplest path through the new engine is documented before contention and RL, and the ADRs that have been Proposed since issue 4 are ratified to Accepted.

**Notebooks.**
- `notebooks/01-quickstart.ipynb` rewritten to walk a 3-node chain end-to-end (factory → shop → sink). Simplest entry point onto the new engine
- `notebooks/02-*.ipynb`, `notebooks/03-inspect_scenario.ipynb` rewritten around graph scenarios; use `Scenario.nodes_df()` / `Scenario.edges_df()` for inspection
- `notebooks/08-tune_textbook_policy.ipynb` rewritten to tune `MultiSupplierTextbookPolicy` on the graph engine
- Notebooks 04–07 explicitly untouched (per PRD §36)

**CONTEXT.md updates.**
- Rewrite entries for `Store` → `Node` family, `Market`, `Policy`, `Scenario`, `Run model`, `RL Env`, `Policy tuning study`
- Add new entries for `Graph`, `EdgeSpec`, `CentralTable`, `Allocation`, `Phase cascade`
- Per-node-type observation tensor shapes intentionally out of scope (documented inline at call sites per PRD)

**ADR ratification.** Promote `docs/adr/0011..0016.md` from `Status: Proposed` → `Status: Accepted`. All phases have shipped; design decisions held. No content drift expected — if any ADR needs amendment, flag in PR description rather than silently editing.

## Acceptance criteria

- [ ] `notebooks/01-quickstart.ipynb` walks a 3-node chain end-to-end on the graph engine
- [ ] `notebooks/02-*.ipynb`, `notebooks/03-inspect_scenario.ipynb` rewritten around graph scenarios; demonstrate `nodes_df()` / `edges_df()`
- [ ] `notebooks/08-tune_textbook_policy.ipynb` rewritten to tune `MultiSupplierTextbookPolicy`
- [ ] Notebooks 04–07 untouched
- [ ] `CONTEXT.md` entries rewritten for `Store`→`Node`, `Market`, `Policy`, `Scenario`, `Run model`, `RL Env`, `Policy tuning study`
- [ ] `CONTEXT.md` adds entries for `Graph`, `EdgeSpec`, `CentralTable`, `Allocation`, `Phase cascade`
- [ ] `docs/adr/0011..0016.md` promoted from `Status: Proposed` → `Status: Accepted`
- [ ] Any required amendments to ADR content flagged in PR description, not silently edited
- [ ] `uv run pytest` (full suite) is green
- [ ] All four end-to-end smoke commands from the plan's verification section run: `uv run python main.py scenarios/example_homogeneous.py`; `uv run python scenarios/example_two_factories_two_shops.py`; `uv run python -m src.rl.train --config src/rl/configs/default.py --steps 5000`; `uv run python -m src.tuning.study --name smoke --trials 5`

## Blocked by

- `.scratch/multi-echelon/issues/12-rl-re-integration.md`
- `.scratch/multi-echelon/issues/13-tuning-re-integration.md`
