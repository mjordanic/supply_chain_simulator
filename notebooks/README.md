# Notebooks

A guided tour of the simulator. Notebooks 00–06 run **offline** against the committed
setup directories or a small synthetic catalog — no API key, no GPU. The analysis
companions (05a, 06a) and notebooks 10 and 13–15 read saved runs under the gitignored `runs/`
directory, and the M5 notebooks need the raw Kaggle files under `data/m5/`; each says
what it needs. Notebooks 09–14 are the earlier latent-replenishment study on the full
engine (`src/jepa/`); 15 is the perfect-supplier experiment that followed it, and 16 explores its track-T2 sealed-test result from committed fixtures. Run any of them with:

```bash
uv run jupyter lab            # then open a notebook
# or execute headless:
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/00-setup-directories.ipynb
```

Each notebook starts with a bootstrap cell that hops up to the repo root, so it works
regardless of where the kernel launches. Generated artifacts land under
`data/notebook_demos/` (gitignored).

| # | Notebook | What it shows |
|---|---|---|
| 00 | [`00-setup-directories.ipynb`](00-setup-directories.ipynb) | Anatomy of a **setup directory** (`catalog.csv` + `setup.yaml`); inspect catalog, nodes, edges, market; draw the DAG; author a new setup by scaffolding or round-tripping a `Scenario`. |
| 01 | [`01-generate-with-llm.ipynb`](01-generate-with-llm.ipynb) | Generate a catalog + market from a one-word domain prompt with the **LLM world generator** (gated on `OPENAI_API_KEY`); shows the on-disk output format and how to turn it into a runnable setup. |
| 02 | [`02-run-and-inspect.ipynb`](02-run-and-inspect.ipynb) | Run a simulation and **inspect every output**: run-log anatomy, per-tier cash/inventory/equity, market supply/demand, per-product series, and the exported parquet/JSON/PNG. |
| 02a | [`02a-inspect-a-saved-run.ipynb`](02a-inspect-a-saved-run.ipynb) | Load a **saved run** from disk (a `main.py run` output) and inspect it without re-running — rebuild the `Scenario` from the config snapshot, replay the inspect helpers, scorecard the run, and reconcile the exported parquet against the run log. Never overwrites a run directory you supply (it regenerates only the default demo run); pairs with 02. |
| 03 | [`03-topology-gallery.ipynb`](03-topology-gallery.ipynb) | Build, draw, run, and compare several **DAG topologies** — linear chain, supplier contention, fan-out, and a **lateral cross-stocking** peer link (`intermediate → intermediate`) — on shared KPIs. |
| 04 | [`04-policy-comparison.ipynb`](04-policy-comparison.ipynb) | Compare the four **textbook inventory policies** head-to-head under Common Random Numbers (same world, only the policy changes). |
| 05 | [`05-tune-a-policy.ipynb`](05-tune-a-policy.ipynb) | Tune a policy with **Optuna**; inspect the trial trajectory, parameter sensitivity, and the best-trial parameters. |
| 05a | [`05a-analyze-a-study.ipynb`](05a-analyze-a-study.ipynb) | Load a **completed tuning study** from disk and analyze it in depth — trajectory, parameter importance, leaderboard, seed-robustness, and the holdout confirmation. Read-only; pairs with 05. |
| 06 | [`06-rl-train-and-eval.ipynb`](06-rl-train-and-eval.ipynb) | Train a tiny **PPO** agent, read its TensorBoard learning curves, and evaluate it CRN-paired against the textbook baseline. |
| 06a | [`06a-analyze-a-trained-agent.ipynb`](06a-analyze-a-trained-agent.ipynb) | Load a **long-trained PPO run** from disk and analyze it in depth — training trajectory, best-checkpoint selection, deep CRN comparison vs the default, regime analysis, and behavioral signature. Read-only; pairs with 06. |
| 09 | [`09-jepa-train-and-plan.ipynb`](09-jepa-train-and-plan.ipynb) | Smoke **latent replenishment** walkthrough: a small transition log, a fit, and one `decide` (covers, priority ration, order). Published commands are in the root README. |
| 10 | [`10-jepa-synthetic-eval.ipynb`](10-jepa-synthetic-eval.ipynb) | Plot the synthetic generalisation tables, floors and oracle ceiling, rollout curve, capacity and data-scaling curves, and the holding-rate sweep. Prints the published eval command when `runs/jepa-replenishment/synthetic/` is absent. |
| 11 | [`11-jepa-m5.ipynb`](11-jepa-m5.ipynb) | **M5 replay exam** for the latent planner: slice rule, zero-shot row, adaptation row. Skips with the Kaggle download and the slice rule when raw files are absent. |
| 12 | [`12-jepa-drilldown.ipynb`](12-jepa-drilldown.ipynb) | **Drill into a JEPA fit** on a short log: the `LatentDynamics` modules `fit_latent` builds, its training loss, and the logged buyer, planner and tuned order-up-to rule deciding on the same held-out observations. Writes under `runs/jepa-drilldown/`. |
| 13 | [`13-jepa-training-log.ipynb`](13-jepa-training-log.ipynb) | Load the published **transition log** (`runs/jepa-replenishment/log/transitions.parquet`) with the fit's holdout split, and count reorders and arrivals per episode. |
| 14 | [`14-jepa-validation-decisions.ipynb`](14-jepa-validation-decisions.ipynb) | Replay the latent fit's **validation episodes** and record, per observation, the order the planner and each textbook rule would send. Needs `runs/jepa-replenishment/`; about five minutes per episode. |
| 15 | [`15-perfect-supplier.ipynb`](15-perfect-supplier.ipynb) | **Perfect-supplier** experiment (ADR 0033): the JEPA planner and its ablations vs tuned textbook and EOQ rules in a one-shop world where only demand is uncertain, plus the `one_product` and `no_fee` variants. Stored outputs are the pre-harness result; newer results are in `experiments/reports/`. Re-running needs `runs/perfect-supplier*/` (commands in `src/perfect_supplier/README.md`). |
| 16 | [`16-perfect-supplier-t2.ipynb`](16-perfect-supplier-t2.ipynb) | **Perfect-supplier track T2**, the sealed-test result: arms and pooled means from `experiments/frozen.json` and the T2 log, a variant and arm-pair picker over the per-world test scores (paired difference, BCa interval, holding-rate / CV / lag strata), and a day-by-day re-simulation of one dev or tune world with frozen arms. Reads the small fixture set in `notebooks/fixtures/t2/` (test scores and the seed-0 models); set `T2_DATA_ROOT` to a full T2 data copy for the other seeds. Never re-runs the sealed test. |
| M5 | [`m5_replay_example.ipynb`](m5_replay_example.ipynb) | End-to-end **M5 real-demand replay**: raw Kaggle files → quality report → invented upstream (factory + CA DC) → emitted CA shops → replay sinks → (s,S) policy run → exact-replay assertion → event-overlay plot → `PriceReplayPolicy` variant. Exits gracefully when raw files are absent. |

Start at **00** for the core model, then branch wherever your interest is.
