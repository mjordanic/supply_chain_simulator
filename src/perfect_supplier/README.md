# `src/perfect_supplier/` — one perfect supplier, learned demand

A lean single-shop replenishment experiment (ADR 0033). Only demand is
uncertain, and only demand is learned. Inventory mechanics are exact.

| module | role |
| --- | --- |
| `world.py` | samples a world (economics, opening state) and its demand path |
| `sim.py` | the daily loop, in `src/sim`'s tick order; logs every day × product |
| `textbook.py` | the four textbook rules, line for line with `src/sim/policy.py` |
| `behavior.py` | the behaviour mixture that writes the training log |
| `forecast.py` | oracle and moving-average demand forecasters |
| `jepa.py` | JEPA quantile forecaster, censored pinball loss, training |
| `planner.py` | path-sampling planner choosing a joint reorder plan |
| `data.py` | partitioned parquet storage and loaders |
| `experiment.py` | `collect`, `fit`, `tune`, `eval`, `shadow` commands |
| `analysis.py` | paired tables and plots for notebook 15 |

```
uv run python -m src.perfect_supplier collect --n-worlds 2000
uv run python -m src.perfect_supplier fit --steps 20000
uv run python -m src.perfect_supplier tune --n-trials 200 --n-worlds 64
uv run python -m src.perfect_supplier eval --n-worlds 500
uv run python -m src.perfect_supplier shadow --n-worlds 12 --driver jepa
```

Output lands under `runs/perfect-supplier/` (`timeseries/`, `summary/`,
`models/`, `tuning/`). `notebooks/15-perfect-supplier.ipynb` reads it.
