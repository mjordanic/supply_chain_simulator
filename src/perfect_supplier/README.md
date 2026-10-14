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

## Result (500 eval worlds, 2026-10-14)

Margin capture, each arm closed-loop on its own copy of every world, paired by world:

| arm | score | paired difference, pts, 95% CI |
| --- | --- | --- |
| planner + oracle forecast (ceiling) | 0.9629 | oracle − JEPA: +0.37 [+0.35, +0.40] |
| JEPA planner | 0.9592 | JEPA − tuned (R,S): +2.43 [+2.36, +2.49], wins 100% of worlds |
| direct-forecast planner (no JEPA loss) | 0.9592 | JEPA − direct: +0.00 [−0.01, +0.02] |
| planner + moving average | 0.9556 | JEPA − MA: +0.36 [+0.32, +0.39] |
| tuned (R,S) / (s,Q) / (s,S) / (R,s,S) | 0.935 / 0.929 / 0.929 / 0.929 | |

Most of the gain is the planner: it orders all products on one day (about 49 order days a year against about 91) and sizes stock against the holding rate. The learned forecast adds 0.36 points over a moving average, mostly fewer lost sales. The JEPA latent loss adds nothing over the same network trained on the quantile loss alone.
