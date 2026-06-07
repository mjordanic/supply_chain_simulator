# Notebooks

A guided tour of the simulator. Every notebook runs **offline** against the committed
setup directories or a small synthetic catalog — no API key, no GPU. Run any of them with:

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
| 03 | [`03-topology-gallery.ipynb`](03-topology-gallery.ipynb) | Build, draw, run, and compare several **DAG topologies** — linear chain, supplier contention, fan-out — on shared KPIs. |
| 04 | [`04-policy-comparison.ipynb`](04-policy-comparison.ipynb) | Compare the four **textbook inventory policies** head-to-head under Common Random Numbers (same world, only the policy changes). |
| 05 | [`05-tune-a-policy.ipynb`](05-tune-a-policy.ipynb) | Tune a policy with **Optuna**; inspect the trial trajectory, parameter sensitivity, and the best-trial parameters. |
| 05a | [`05a-analyze-a-study.ipynb`](05a-analyze-a-study.ipynb) | Load a **completed tuning study** from disk and analyze it in depth — trajectory, parameter importance, leaderboard, seed-robustness, and the holdout confirmation. Read-only; pairs with 05. |
| 06 | [`06-rl-train-and-eval.ipynb`](06-rl-train-and-eval.ipynb) | Train a tiny **PPO** agent, read its TensorBoard learning curves, and evaluate it CRN-paired against the textbook baseline. |
| 06a | [`06a-analyze-a-trained-agent.ipynb`](06a-analyze-a-trained-agent.ipynb) | Load a **long-trained PPO run** from disk and analyze it in depth — training trajectory, best-checkpoint selection, deep CRN comparison vs the default, regime analysis, and behavioral signature. Read-only; pairs with 06. |

Start at **00** for the core model, then branch wherever your interest is.
