# T2: fair consolidation of the lean perfect-supplier result

Track T2 of the sc-jepa plan (audit §6). Protocol v4 (`experiments/protocol.yaml`,
sha256 `5b1785bc…`). Every run is a line in `experiments/log/T2-consolidation.jsonl`.
The harness reports are in `experiments/reports/T2/`. The sealed test ran once per
(arm, variant), with frozen configs only (`experiments/frozen.json`), and nothing was
re-tuned or re-run after it.

## Bottom line

The learned forecaster plus planner beats the best tuned textbook rule in all three
T2 worlds on the sealed test (seeds 9,000,000–9,000,999). All six primary comparisons
are **practically superior** under Holm. Holm here counts all 8 declared comparisons,
with T1's two not-yet-run comparisons set to p = 1, so the labels can only loosen
once T1 lands.

| world | reference (picked on tune worlds) | JEPA − ref, pts [95% BCa] | cost reduction | direct − ref, pts | label |
| --- | --- | --- | --- | --- | --- |
| base (4 products, order fee) | joint EOQ per holding rate × lag | **+0.233** [+0.218, +0.249] | 5.3% | +0.241 [+0.227, +0.257] | practically superior (both) |
| no_fee | joint EOQ per holding rate × lag | **+0.186** [+0.171, +0.202] | 11.2% | +0.170 [+0.156, +0.186] | practically superior (both) |
| one_product | joint EOQ, z mapped from the critical ratio | **+0.132** [+0.106, +0.160] | 1.7% | +0.135 [+0.110, +0.164] | practically superior (both) |

The audit's four checks:

1. **Contextual baselines.** The reference is the best of 9 tuned textbook arms. That
   includes per-holding-rate, per-holding-rate × lag and parametric (critical-ratio)
   joint EOQ, each tuned with 192 trials × 32 worlds *per cell*. The reference is
   picked by a selection score on 1,024 common tune worlds that no tuner used. The
   lead survives.
2. **A classical seasonal forecaster in the same planner.** Tuned seasonal ETS gets the
   same knobs and the same budget. It only ties the best textbook rule (+0.02 pts in
   every world). JEPA beats it by +0.21 (base), +0.16 (no_fee) and +0.11 (one_product)
   pts.
3. **Three training seeds.** The per-seed estimates against the reference span less
   than 0.01 pts: base +0.228–0.238, no_fee +0.184–0.188, one_product +0.130–0.133.
4. **The sealed test.** It matches dev (base +0.251, no_fee +0.154, one_product +0.134
   on 100 dev worlds) and the audit's +0.24 / 5.6%.

**Coherent path sampling turns the ties into wins.**
- In no_fee, the same seed-0 JEPA model with the original planner (`jepa_s0_plain`,
  an ablation outside the families) *loses* to the reference by −0.036 pts
  [−0.056, −0.014]. The tuned knobs add +0.223 [+0.209, +0.237].
- In one_product, the knobs add +0.058 [+0.038, +0.081].
- In base they add nothing (+0.002 [−0.008, +0.012]); the audit saw the same.
- The tuned path coherence ρ is high where holding is cheap and stock-outs are what
  cost: no_fee, h = 0.25/365, ρ ≈ 0.68–0.71. It is near 0 in base.

**The JEPA objective adds nothing here.** JEPA − direct is −0.008 pts (base),
+0.016 (no_fee) and −0.004 (one_product). Each is well inside ±SESOI (TOST p < 1e-49),
so the two are practically equivalent. The base and no_fee differences are
statistically non-zero (Holm labels "inferior" and "superior"), but tiny. The
headline therefore stays **"learned forecaster plus planner"**, not JEPA.

**The oracle ceiling.** The same planner with the true demand distribution beats the
reference by +0.624 (base), +0.654 (no_fee) and +0.339 (one_product) pts. JEPA
captures 37%, 28% and 39% of that headroom; JEPA − oracle is −0.39, −0.47 and
−0.21 pts.

## Where the lead comes from (exploratory strata, BH q = 0.10)

JEPA − reference, by holding rate:

| world | h = 0.25/365 | h = 1/365 | h = 0.01 |
| --- | --- | --- | --- |
| base | +0.084 | +0.183 | +0.437 |
| no_fee | +0.029 | +0.145 | +0.386 |
| one_product | −0.012 (n.s.) | +0.114 | +0.294 |

The lead grows with the holding cost: a sharper forecast buys a smaller safety stock.
Mean on-hand stock is 5% lower than the reference's in base and 8% lower in no_fee,
at a higher fill rate (99.59% vs 99.50% in base). In one_product, stock is about the
same and the gain comes from fewer lost sales. The lead is smallest where
over-stocking is nearly free.

## What changed (code)

- `planner.py`, with every default keeping the original decisions bit for bit:
  - `rho`: Gaussian-copula path coherence across days.
  - `service_max`: a service grid up to 0.9995.
  - `level_paths`: separate paths for the reorder and order-up-to quantiles.
  - The forecaster's own quantile levels.
  - `imagine()` is about 2.4× faster with identical results.
- `forecast.py`: `TAIL_QUANTILES` (adds 0.99, 0.995, 0.999). The oracle,
  moving-average and seasonal-ETS forecasters take `levels`. `JepaConfig.quantiles`
  carries a model's levels, and old models load with the original 11.
- Harness additions T2 needed:
  - `bench fit`: one logged (arm, seed) model.
  - `bench select`: a common-worlds selection score; `best_textbook` prefers it.
  - The tuner spreads a learned arm's budget over its cells.
  - An ablation `role` keeps an arm out of the families.
  - `pooled-primary --partial`: conservative Holm before T1's cell exists.
- Protocol v4 (logged amendment, before any T2 tuning or test look) declares the
  planner and ETS search spaces, the learned contextual budget and the selection rule.

## Budgets

| arm | budget | search space |
| --- | --- | --- |
| textbook, global (eoq, eoq_joint, param_eoq_joint, the 4 engine rules) | 192 trials × 32 worlds | protocol |
| textbook, contextual (per holding rate: 3 cells; per holding rate × lag: 18 cells) | 192 × 32 **per cell** | protocol |
| jepa_s0, direct_s0, seasonal ETS, MA planner, oracle planner, per holding-rate cell | 48 × 32 in all (16 × 32 per cell) | ρ, service_max, level_paths (+ α, γ for ETS) |
| jepa / direct seeds 1, 2 | none; seed 0's knobs, own model | — |

The textbook arms got 4× (global) to 12× (per cell) the learned budget. Seeds 1 and 2
reuse seed 0's planner knobs. That gives seed 0 a small selection advantage, but the
per-seed estimates show no visible effect.

## Caveats and honest limits

- **Small absolute effects.** The win is at most 0.23 pts of margin capture. That is
  5% of the reference's cost in base, 11% in no_fee and 2% in one_product. The
  controllable headroom (the oracle) is itself only 0.34–0.65 pts.
- **The learned upper tail under-covers.** On held-out behaviour worlds the trained
  0.999 quantile covers only about 95% of true demand (censoring). The closed-loop
  tuned service cap and ρ absorb it; the forecast itself is not tail-calibrated.
- **The reference choice.** The reference picked on tune worlds is also the best
  textbook arm on test in one_product (parametric joint EOQ, 0.92341, against 0.92335
  per holding rate). In base and no_fee, the other contextual arms are within
  0.0002 of it.
- **Dev looks.** Dev was looked at once (100 worlds per variant) as a sanity check
  before freezing. Nothing was chosen on dev.
- **The family is not complete yet.** T1's hidden-regime cell is not in the family yet.
  `bench pooled-primary` (without `--partial`) gives the final labels once it is.

## Reproduce

Paths are relative to the task data directory, which holds the behaviour log, models,
tuned configs and score files (not committed).

```
B="uv run python -m src.perfect_supplier.bench --track T2-consolidation"
uv run python -m src.perfect_supplier --root runs/train collect --n-worlds 2000
$B fit --root runs/train --arm {jepa,direct} --seed {0,1,2} --out models      # PS_WORKERS=1 each
$B tune --kind <textbook kind> [--cell-keys holding_rate [delivery_lag]] --variant V --out tune/V
$B select --variant V --offset 3072 --n-worlds 1024 --arms tune/V/<textbook arms>.json
$B tune --kind {planner,seasonal_ets} --cell-keys holding_rate --base-config <base> --variant V --out tune/V
$B freeze --arms arms/V/*.json --note ...
$B eval --arms arms/V/*.json --split test --n-worlds 1000 --variant V --out test/V
$B report --scores test/V --out test/V/report.md
$B pooled-primary --partial --out test/primary.md
```
