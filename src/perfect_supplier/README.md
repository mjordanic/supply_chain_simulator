# `src/perfect_supplier/` — one perfect supplier, learned demand

A lean single-shop replenishment experiment (ADR 0033). Only demand is
uncertain, and only demand is learned. Inventory mechanics are exact.

| module | role |
| --- | --- |
| `world.py` | samples a world (economics, opening state) and its demand path |
| `sim.py` | the daily loop, in `src/sim`'s tick order; logs every day × product |
| `textbook.py` | the four textbook rules, line for line with `src/sim/policy.py`, and EOQ (s,Q) with a per-product or joint trigger |
| `behavior.py` | the behaviour mixture that writes the training log |
| `forecast.py` | oracle and moving-average demand forecasters |
| `jepa.py` | JEPA quantile forecaster, censored pinball loss, training |
| `planner.py` | path-sampling planner choosing a joint reorder plan |
| `data.py` | partitioned parquet storage and loaders |
| `experiment.py` | `collect`, `fit`, `tune`, `eval`, `shadow` commands |
| `analysis.py` | paired tables and plots for notebook 15 |
| `joint.py` | track T1: multi-product JEPA / direct forecaster over the whole shop |
| `regimes.py` | track T1: known-model Bayes regime filter and Baum-Welch HMM forecasters |
| `t1.py` | track T1 workflow: banks, fits, gates G1/G2 (result: `experiments/reports/T1-hidden-regimes.md`) |
| `bench/` | the pre-registered evaluation harness every new experiment runs through (ADR 0034, `bench/README.md`) |

```
uv run python -m src.perfect_supplier collect --n-worlds 2000
uv run python -m src.perfect_supplier fit --steps 20000
uv run python -m src.perfect_supplier tune --n-trials 200 --n-worlds 64
uv run python -m src.perfect_supplier eval --n-worlds 500
uv run python -m src.perfect_supplier shadow --n-worlds 12 --driver jepa
```

Output lands under `runs/perfect-supplier/` (`timeseries/`, `summary/`,
`models/`, `tuning/`). `notebooks/15-perfect-supplier.ipynb` reads it.

Two variants take the order fee out of the comparison (`world.py`):
`one_product` keeps product 0 of each base world; `no_fee` sets the fee to 0.
They reuse the base forecasters, which learn demand only:

```
for V in one_product no_fee; do
  uv run python -m src.perfect_supplier --variant $V tune
  uv run python -m src.perfect_supplier --variant $V --model-root runs/perfect-supplier eval --n-worlds 500
  uv run python -m src.perfect_supplier --variant $V --model-root runs/perfect-supplier shadow --n-worlds 12
done
```

Output for a variant goes under `runs/perfect-supplier-<variant>/`.

## Result (500 eval worlds per world type, 2026-10-15)

Margin capture, each arm closed-loop on its own copy of every world, paired by
world. The best textbook rule in all three worlds is tuned joint EOQ: joint
replenishment with a common EOQ cycle and `z·sd·sqrt(lag)` safety stock. It
reads the fee and holding rate, as the planner does.

| world | JEPA | best textbook | JEPA − best textbook, pts [95% CI] | JEPA wins | JEPA − direct |
| --- | --- | --- | --- | --- | --- |
| base (4 products, fee 50) | 0.9592 | 0.9568 | +0.24 [+0.22, +0.26] | 88% | −0.00 |
| one product, fee 50 | 0.9252 | 0.9246 | +0.06 [+0.02, +0.11] | 53% | −0.01 |
| 4 products, no fee | 0.9818 | 0.9823 | −0.05 [−0.09, −0.01] | 39% | −0.01 |

- **Engine rules.** Against the four engine rules (per-product triggers, no fee or holding-rate input), JEPA leads by 2.6–3.1 points in the base world. Most of that is ordering every product on one day: joint EOQ does the same without learning and closes 2.35 points.
- **Without the fee**, JEPA ties the best textbook rule.
- **The forecast.** The learned forecast beats a moving average inside the same planner (+0.24 to +0.65). The JEPA latent loss adds nothing over the direct ablation.
- **Headroom.** The oracle planner shows 0.2–0.7 points a better forecast could take.

Notebook 15, section 12, has the details.
