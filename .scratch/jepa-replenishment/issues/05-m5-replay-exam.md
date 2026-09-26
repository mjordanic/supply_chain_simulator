# 05: M5 replay exam

**What to build:** A notebook that exams the frozen synthetic planner on one Walmart shop. When the raw M5 files are absent it skips with the download command and the slice rule. When they are present it chooses five FOODS items from 2015 only, pins those ids, grafts one factory at lead time 3 onto `shop-CA_1`, and reports a zero-shot row and a 2015-adaptation row on the 2016 test window. Both ordering policies sit inside the same `PriceReplayPolicy` under `flat_world`. The textbook anchor is tuned on the fixed 2015 replay scenario, not by the synthetic episode sampler.

**Blocked by:** 02 — Fit the latent model and plan with it.

**Status:** ready-for-agent

## Parent

`.scratch/jepa-replenishment/PRD.md`. ADR 0020, ADR 0026, ADR 0028.

## Stories

PRD stories 10 and 11 for the M5 graph, 43–53, 56, 57, 65, 67, 68.

## Prior art

`ReplayDemandSinkNode`, `flat_world`, `PriceReplayPolicy`, `quality_report`, and `emit_m5_setup_dir`. The M5 example notebook's skip when the raw files are absent. `OrderUpToPolicy` search via the existing tuner trial space, but the objective is profit on the fixed replay scenario: the tuner's `setup_dir` path only borrows catalog and market and still builds a synthetic graph, so it is not this exam. Business metrics from a `Runner` run log.

## Artifact homes

- `notebooks/11-jepa-m5.ipynb` (tracked).
- Raw files: `data/m5/` (gitignored). Emitted scenario: `data/m5_jepa/` (gitignored).
- Pinned item ids are written into the notebook and, by ticket 06, into the paper. They are not a separate committed data file.
- Replay eval tables, when the files exist: `runs/jepa-replenishment/m5/` (gitignored).

## Acceptance criteria

- [ ] With the raw directory missing, the notebook skips, names the Kaggle download, and states the slice rule. A test covers that skip without `data/m5`.
- [ ] Selection uses 2015-01-01 through 2015-12-31 only. Rank `FOODS_3` at `CA_1` by shortest `longest_zero_run`, then highest mean sales. Drop a zero run longer than 7 days. If fewer than five items survive, widen to all FOODS departments at `CA_1` under the same rule. The five ids are pinned in the notebook after the first quality report.
- [ ] The graph is one `StaticFactoryPolicy` factory feeding `shop-CA_1` at lead time 3, plus one replay sink per pinned item. `unit_cost` is 40% of median price. `flat_world` is on. Both the learned Policy and `OrderUpToPolicy` are wrapped in the same `PriceReplayPolicy`.
- [ ] The test window is 2016-01-01 through 2016-04-24. The zero-shot row uses the synthetic weights and does not fit on either M5 window. The adaptation row refits on reward-free transitions from the 2015 window only, collected with the same two behavior policies. The windows are not pooled.
- [ ] The textbook anchor is an `OrderUpToPolicy` tuned on 2015 replay profit only. The 2016 window is not used to pick its hyperparameters.
- [ ] Both rows are reported. Neither row is a pass/fail gate.
