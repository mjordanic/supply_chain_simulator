# 07 — Notebook `notebooks/08-tune_textbook_policy.ipynb`

Status: ready-for-agent

## Parent

PRD: `.scratch/policy-tuning/PRD.md`
ADR: `docs/adr/0009-policy-hyperparameter-tuning-tool.md`

## What to build

Add the demonstration + analysis notebook for the tuning tool. The notebook is a tutorial + analysis hybrid: a short inline demo runs a ~10-trial study live (visible API call), then the remaining sections load `runs/tuning/order_up_to_v1/` from disk and produce the headline plots.

No committed full-study fixture. Cells that depend on `runs/tuning/order_up_to_v1/` display a friendly "run `uv run python -m src.tuning.study --policy order_up_to --trials 150 --study-name order_up_to_v1` to populate this section" message when the artifact directory is absent. The shipped `.ipynb` includes whatever executed outputs the implementing agent produced when running locally.

### Section layout

1. **Setup & motivation** — Imports, ADR 0009 framing-1 link, one paragraph on "measurement instrument, not baseline-construction tool".
2. **The tool in one screen + live 10-trial demo** — Show a 5-line example calling `run_study(order_up_to_space, ..., TuningConfig(n_trials=10, n_search_seeds=4, episode_length=60))` inline so the API is visible. Runtime budget ~1–2 min.
3. **Load pre-run study from `runs/tuning/order_up_to_v1/`** — `pd.read_parquet` + `json.load`. If the directory does not exist, the cell prints the CLI invocation and the rest of the notebook degrades to placeholders.
4. **Optimization trajectory plot** — `trial_id` vs `mean_normalised_return` scatter, with running-best line.
5. **Headline: tuned vs published default (paired CRN on held-out seeds)** — Read `holdout_summary.json`, display the headline uplift + CI as a one-line summary cell, and a paired bar chart from `holdout.parquet`.
6. **Parameter sensitivity scatters** — 4-panel grid: each tunable parameter on the x-axis, `mean_normalised_return` on the y-axis, points coloured by `trial_id` order. From `trials.parquet`.
7. **Param importances** — Reconstruct an in-memory `optuna.Study` from `trials.parquet` (use `optuna.create_study` + manual `_add_trial`-style replay, or `optuna.trial.create_trial` + `study.add_trial`); call `optuna.importance.get_param_importances(study)`; render as a horizontal bar chart.
8. **Pareto front: profit vs service level** — Scatter of `mean_service_level` vs `mean_net_profit` (from `trials.parquet`); highlight the Pareto-non-dominated set; annotate the search winner.
9. **Scale robustness: per-capacity-bucket paired bar** — From `per_seed.parquet`, bucket rows by `capacity` (small / medium / large terciles of the LogUniform sample); compare the search winner's per-bucket mean to the default's per-bucket mean (re-evaluate the default on the same per-seed rows from the search if available, or read from `holdout.parquet`); paired bar chart.
10. **Closing markdown** — Pointer to ADR 0009 out-of-scope list (per-world tuning, per-scale tuning, multi-objective Optuna) as the natural follow-ups.

### Graceful-fallback convention

Sections 4–9 each begin with:

```python
study_dir = pathlib.Path("runs/tuning/order_up_to_v1")
if not study_dir.exists():
    print("Run `uv run python -m src.tuning.study --policy order_up_to --trials 150 --study-name order_up_to_v1` to populate this section.")
else:
    # ... actual analysis
```

The notebook never raises a `FileNotFoundError` mid-execution.

### Plotting conventions

Match the existing notebook style in `notebooks/06-*.ipynb` and `notebooks/07-*.ipynb`: matplotlib, consistent figure sizes, axes labels and titles in plain English. No `seaborn` dependency unless already in `pyproject.toml`.

PRD user stories covered: 1, 12, 13, 14, 15, 16, 17.

## Acceptance criteria

- [ ] `notebooks/08-tune_textbook_policy.ipynb` exists with the ten sections in the listed order.
- [ ] Section 2 contains a live 10-trial `run_study(...)` call that completes in 1–2 minutes (small `n_search_seeds`, short `episode_length`).
- [ ] Sections 4–9 each handle the "no on-disk artifact" case by printing the CLI invocation rather than erroring.
- [ ] Section 7 reconstructs the Optuna study in-memory from `trials.parquet` (no SQLite read).
- [ ] Section 8's Pareto scatter highlights the non-dominated frontier and annotates the search winner.
- [ ] No `runs/tuning/order_up_to_v1/` artifact is committed to the repo (the directory is gitignored).
- [ ] The implementing agent runs `uv run python -m src.tuning.study --policy order_up_to --trials 150 --study-name order_up_to_v1` locally and commits the executed-output cells against those artifacts.
- [ ] `uv run jupyter nbconvert --to notebook --execute notebooks/08-tune_textbook_policy.ipynb --output /tmp/exec_check.ipynb` succeeds when the artifact directory is present; it also succeeds (gracefully) when the directory is absent.

## Blocked by

- `06-cli-entry-point.md` — the notebook documents the CLI invocation as the prerequisite for sections 3–9 and uses it to produce the executed-output artifacts.
