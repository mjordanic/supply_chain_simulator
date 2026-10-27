# `src/perfect_supplier/bench/` — the shared evaluation harness (H0)

Every perfect-supplier experiment from here on runs through this package, so
that results rest on a pre-registered protocol: a sealed test block, tuning
budgets that favour the textbook baselines, every run logged, paired
statistics with multiplicity control. ADR 0034 records the decision.

| module | role |
| --- | --- |
| `protocol.py` | loads `experiments/protocol.yaml` (constants, seed blocks, burnt seeds, budgets, search spaces); the sealed-split gate and `experiments/frozen.json` |
| `arms.py` | arm registry: a JSON config per arm (textbook families, contextual per-cell and parametric EOQ, the planner with oracle / moving-average / seasonal-ETS / direct / JEPA forecasts) |
| `tuner.py` | sequential TPE with a process pool per trial; budget and search space from the protocol; per-cell contextual tuning; tuning curves |
| `evaluate.py` | paired evaluation writer: every arm on its own copy of the same worlds → `scores.parquet` + `arms.json` |
| `stats.py` | BCa bootstrap, Wilcoxon, Holm, BH, TOST, two-level (world × training seed) bootstrap, cost reduction, protocol labels |
| `calibration.py` | lead-time coverage of a planner arm's imagined demand (exceedance at q0.90/0.95/0.99, PIT) |
| `log.py` | append-only `experiments/log/<track>.jsonl`; first line is the protocol hash; failed and refused runs are logged |
| `report.py` | markdown report: arm breakdown, primary family (Holm), secondary family, training seeds, strata (BH); families from the protocol |
| `backfill.py` | the pre-harness results as `kind: historical` log lines |

## Protocol constants (approved 2026-10-27)

- primary metric margin capture, cost reduction `1 − (1 − s_arm)/(1 − s_ref)` co-reported
- SESOI 0.10 pts margin capture
- seed blocks: train 1,000,000+, tune 2,000,000+, dev 4,000,000–4,000,499,
  **test 9,000,000–9,000,999 (sealed)**; burnt: 3,000,000–3,000,499 (old eval),
  3,100,000+ (shadow), 3,200,000–3,200,199 (audit), 12M–16M (engine line)
- budgets: learned arms 48 trials × 32 worlds; textbook arms 4× that (192 × 32),
  each cell of a contextual arm gets the full textbook budget
- 3 training seeds per learned arm; Holm at α = 0.05 per family; BH q = 0.10 for strata
- primary family: JEPA and direct planner arms (averaged over training seeds) −
  the textbook arm with the best tune score; secondary family: JEPA − direct,
  − seasonal ETS, − moving average, − oracle; strata holding rate, CV, lag
- stopping rules: T1 gates G1 (oracle beats best textbook by ≥ 1 pt on dev
  before training) and G2 (JEPA beats direct on dev forecast metrics before
  closed-loop runs); T3 oracle beats tuned capped base-stock by ≥ SESOI on dev;
  T2 no gate; one sealed test run per (arm, variant)

Change the protocol only by amendment: edit, bump `version`, and log a
`kind: amendment` line saying why. Every log line carries the file's sha256.

## Commands

```
B="uv run python -m src.perfect_supplier.bench --track T2-consolidation"

# arm configs for the stored arms of a run root (tuned params, model hashes)
$B import-legacy --root runs/perfect-supplier --variant base --out arms/base

# tune an arm with the protocol budget (per holding-rate cell here)
$B tune --kind eoq_joint --name ctx_eoq_joint --cell-keys holding_rate --out runs/bench/tune

# paired eval on the dev block, then a report
$B eval --arms arms/base/jepa.json runs/bench/tune/ctx_eoq_joint.json --split dev --n-worlds 300 --out runs/bench/dev1
$B report --scores runs/bench/dev1 --out runs/bench/dev1/report.md

# lead-time calibration of a planner arm (dev only)
$B calibrate --arm arms/base/jepa.json --n-worlds 60 --out runs/bench/cal

# the sealed test: freeze first, then one run per (arm, variant)
$B freeze --arms arms/base/jepa.json runs/bench/tune/ctx_eoq_joint.json --note "T2 final"
$B eval --arms ... --split test --n-worlds 1000 --out runs/bench/test
```

A `--split test` run is refused (and the refusal logged) unless every arm's
config hash is in `experiments/frozen.json` under the current protocol. A
second test run of the same (arm, variant) also needs `--rerun-reason`.
`PS_WORKERS` sets the process pool size (default 3 from the protocol).

The report's families come from `comparisons` in the protocol. Passing
`--reference`, `--primary`, or `--secondary` makes it exploratory: the report
says so and its log line records `families: exploratory`.

Arms that share a `seed_group` field in their configs (one config per training
seed) are averaged in the report, which adds a two-level bootstrap section.
