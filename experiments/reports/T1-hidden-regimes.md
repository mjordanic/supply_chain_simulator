# Track T1: hidden shop-wide demand regimes — result (negative; stopped at gate G2)

Hypothesis H1 (audit §4, plan §6 "T1"): when demand follows a hidden, persistent, shop-wide regime that
only joint sales reveal, a multi-product JEPA forecaster with regime-coherent path sampling in the
exact-recursion planner beats the best tuned textbook rule, and beats the same network without the JEPA
loss ("direct").

**Outcome.** Gate G1 (headroom) passed only at regime strength 2.0. Gate G2 (JEPA beats direct on dev
forecasts) **failed**: JEPA is significantly *worse* than direct, and both learned forecasters trail a
correctly specified Baum–Welch HMM by a wide margin. Per the pre-registered stopping rule and the
captain's decision on `t1-gate` (option A), T1 stops at G2. No closed-loop learned arm, no sealed-test
look; the protocol's primary cell `T1-hidden-regimes / hidden_regimes / base` has no test eval.

Every run, amendment, and gate line is in `experiments/log/T1-hidden-regimes.jsonl`. Harness reports
are copied under `experiments/reports/T1/`. Run artifacts (scores, models, banks) live outside the repo
under the firstmate home's `data/sc-jepa-t1/`, hashed in the log.

## Setup

- **World** (`hidden_regimes`, `src/perfect_supplier/world.py`): the 4-product, order-fee base world's
  draws times one hidden shop-wide regime multiplier. A 3-state Markov chain leaves its state with
  probability `switch_prob` per day (log-uniform 1/60–1/20 per world), to either other state uniformly.
  Mean multipliers 0.6 / 1.0 / 1.6 raised to `strength`; noise cv scaled by 1.2 / 1.0 / 0.8 per regime;
  the idiosyncratic per-product level shifts stay. Shelf and opening cash scale with the busiest regime so
  neither binds. No shared indicator (optional in the spec; not built).
- **Textbook arms**, each tuned on 32 tune worlds × 192 trials (4× the learned budget), re-tuned at every
  strength: per-product EOQ, joint EOQ, joint EOQ on an exponentially smoothed rate (`ets_eoq_joint`, new),
  parametric-z joint EOQ, the four engine rules, and per-holding-rate contextual joint EOQ and ETS joint EOQ.
- **Ceilings**: the planner with the clairvoyant oracle forecast (true mean path), and the known-model
  **Bayes** planner (`regimes.BayesRegimeForecaster`): it knows the regime model and each product's
  non-regime mean path, filters the regime from censored sales, and draws one regime trajectory per path.
  Untuned (64 paths, service grid capped at 0.995), so it is a conservative ceiling.
- **Learned forecasters** (`src/perfect_supplier/joint.py`): one multi-product context encoder (all
  products' last 35 days, scaled, with sold-out flags, plus per-product long-run level). The JEPA target is
  the EMA embedding of the joint future window. A regime-mixture head (3 components of per-day,
  per-product Gaussians on scaled demand) is trained by the censored, discretised mixture likelihood of the
  whole window; sampling one component per path gives coherent paths. Direct is the same network without
  the target encoder, predictor, and latent loss. 3 training seeds each, 12k steps, 2,000 behaviour-policy
  train shops (10% held out for early stopping on the forecast loss).
- **Classical model-based baseline**: a Gaussian HMM fitted by Baum–Welch on weekday-adjusted, centred log
  shop sales (sold-out days missing), order chosen on tune-bank likelihood (K = 4 of {2, 3, 4}; the edge
  of the grid, the true K is 3), used as a planner forecaster.

## (i) Boundary curve: regime strength vs headroom over the best textbook rule (G1, dev)

300 dev worlds (4,000,000–4,000,299), margin capture in points, 95% BCa. Gate G1 needs the Bayes planner
≥ 1.00 pt over **both** the protocol reference (textbook arm with the best tune score) and the best
textbook arm on dev (rule fixed in `t1.G1_RULE` before the first run). Each strength change was a logged
protocol amendment (v5, v6) on dev before any training; Firstmate capped the search at 2.0.

| strength | multipliers | Bayes − best textbook on dev | Bayes − tune reference | oracle − best textbook on dev | Bayes cost reduction vs tune ref | G1 |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 0.60 / 1.0 / 1.60 | +0.748 [+0.680, +0.825] | +0.838 | +1.315 | +16.5% | fail |
| 1.5 | 0.46 / 1.0 / 2.02 | +0.942 [+0.845, +1.062] | +1.011 | +2.093 | +17.4% | fail |
| 2.0 | 0.36 / 1.0 / 2.56 | +1.286 [+1.147, +1.444] | +1.376 [+1.239, +1.524] | +3.064 | +20.3% [+18.8, +21.7] | **pass** |

- The best textbook arm on dev was the per-holding-rate ETS joint EOQ at every strength; the protocol
  reference was parametric-z joint EOQ (1.0, 1.5) and ETS joint EOQ (2.0).
- Headroom grows with regime strength, holding rate, and delivery lag (harness strata). The clairvoyant
  oracle's lead grows about twice as fast as the Bayes planner's: most of what perfect foresight buys is
  not recoverable by any filter.
- Even with a known model, a ≥ 1-pt margin-capture headroom over tuned joint EOQ needs a ~7× swing between
  the slowest and busiest regime. At the audit's suggested 0.6 / 1.6 the ceiling is 0.75 pts — similar to
  the 0.62 pts in the perfect-supplier base world.

## (ii) JEPA is worse than direct on dev forecasts (G2)

300 dev shops run by behaviour policies (9% sold-out product-days), scored against true demand every 7th
decision day, 256 paths per forecast; mean pinball over levels 0.1/0.5/0.9/0.95/0.99, scaled by each
product's context mean (lower is better). Rule fixed in `t1.G2_RULE` before the run.

| forecaster | daily | lead time (lag + 7 days, per product) | shop total over lead time |
| --- | --- | --- | --- |
| JEPA (mean of 3 seeds) | 0.486 | 0.442 | 0.413 |
| direct (mean of 3 seeds) | 0.470 | 0.427 | 0.396 |
| HMM (Baum–Welch, K = 4) | 0.359 | 0.299 | 0.245 |
| Bayes (known model) | 0.221 | 0.146 | 0.139 |
| moving average (28 days) | 0.674 | 0.630 | 0.622 |

| JEPA − direct | mean | 95% BCa | JEPA better in |
| --- | --- | --- | --- |
| daily | +0.0161 | [+0.0096, +0.0278] | 28% of shops |
| lead time | +0.0152 | [+0.0096, +0.0251] | 22% of shops |
| shop total | +0.0165 | [+0.0101, +0.0272] | 30% of shops |

- Consistent over seeds: every JEPA seed (lead time 0.439–0.445) is worse than every direct seed
  (0.423–0.432).
- **Latent-loss evidence.** Training had saturated (val NLL flat by 12k steps). JEPA's best val NLL is
  2.713–2.717 against direct's 2.689–2.692, and JEPA's latent loss *rises* during training
  (2.42–2.50 at 4k steps → 2.83–2.93 at 12k) while the forecast loss falls: the latent target competes with
  the forecast objective rather than shaping a useful representation. This repeats the audit's §3.2(d)
  finding, now with a joint cross-product target, where the hypothesis said JEPA should help most.

## (iii) A fitted HMM beats both learned forecasters

The classical HMM, fitted by Baum–Welch on the same train sales, has a lead-time pinball of 0.299 against
direct's 0.427 and JEPA's 0.442 (30% and 32% lower), and a shop-total error 38–41% lower (0.245 vs 0.396 and 0.413). The
known-model Bayes filter (0.146) shows how much a correctly specified filter can still gain. The learned
models see a 35-day window plus a long-run level, while the HMM and the Bayes filter use the whole history;
the gap is the cost of a fixed context window against a recursive belief.

## Exploratory (option B, after G2 failed): model variants — not a T1 claim

**Post-failure exploratory look (captain's option B): it cannot feed any T1 primary or secondary claim,
does not re-open G2, and led to no closed-loop or sealed-test run.** Eight variants on the remaining JEPA
model budget, one training seed each, fitted on train only, dev forecasts re-scored once
(`t1 explore-fit`, `t1 explore-score`; `T1/explore-forecast.md`). The "ctx56"/"ctx112" variants keep the
35-day encoder window and add the last 8 or 16 weekly means per product over its long-run mean, read by the
predictor and head ("+ 8 / 16 weekly means" below). A longer encoder window itself (56–112 days) was not tested.

| forecaster (seed 0) | daily | lead time | shop total |
| --- | --- | --- | --- |
| JEPA, 35 days (G2 model) | 0.487 | 0.445 | 0.416 |
| direct, 35 days (G2 model) | 0.471 | 0.427 | 0.397 |
| JEPA, 35 days + 8 weekly means | 0.489 | 0.445 | 0.417 |
| JEPA, 35 days + 16 weekly means | 0.499 | 0.455 | 0.428 |
| JEPA, 35 days + 16 weekly means, latent weight 0.3 | 0.499 | 0.453 | 0.424 |
| JEPA, 35 days + 16 weekly means, latent weight 0.1 | 0.486 | 0.446 | 0.417 |
| JEPA, 35 days + 16 weekly means, latent weight 0.1, K = 6 | 0.520 | 0.462 | 0.430 |
| direct, 35 days + 8 weekly means | 0.470 | 0.421 | 0.391 |
| direct, 35 days + 16 weekly means | 0.471 | 0.426 | 0.396 |
| direct, 35 days + 16 weekly means, K = 6 | 0.491 | 0.432 | 0.398 |
| HMM | 0.359 | 0.299 | 0.245 |
| Bayes (known model) | 0.221 | 0.146 | 0.139 |

| JEPA − matching direct, lead time | mean | 95% BCa |
| --- | --- | --- |
| 35 days | +0.018 | [+0.011, +0.030] |
| + 8 weekly means | +0.024 | [+0.014, +0.041] |
| + 16 weekly means | +0.029 | [+0.019, +0.046] |
| + 16 weekly means, latent weight 0.3 (vs direct + 16) | +0.026 | [+0.017, +0.043] |
| + 16 weekly means, latent weight 0.1 (vs direct + 16) | +0.020 | [+0.012, +0.033] |
| + 16 weekly means, latent weight 0.1, K = 6 (vs direct + 16, K = 6) | +0.030 | [+0.017, +0.052] |

- JEPA is worse than its matching direct in every variant; a smaller latent weight shrinks the gap but
  never reverses it, and JEPA's latent loss still rises during training in every variant (to 3.09–3.24).
- Weekly long-horizon summaries do not close the gap to the HMM: the best learned forecaster (direct,
  35 days + 8 weekly means, lead time 0.421) is still 41% worse than the HMM (0.299). Whether a longer
  encoder window (56–112 days) would close it was not tested and remains an open follow-up. K = 6 lowers
  val NLL (2.57–2.59 vs 2.69–2.73) but worsens every pinball metric.
- Nothing here is striking enough to act on; it strengthens findings (ii) and (iii) as exploratory
  evidence only.

## What this supports, and what it does not

- **Supported (dev only; nothing here is held-out):** the boundary map in (i); "the JEPA objective hurts
  relative to direct even with a joint cross-product target" (ii); "a correctly specified fitted HMM beats
  both learned forecasters" (iii). These are the audit's "publishable if it fails" outcomes for T1.
- **Not supported:** any closed-loop learned-arm claim on hidden regimes, any sealed-test label. The T1 primary
  cell has no test eval, so `pooled-primary` will report it missing; the pooled family needs a
  decision on dropping the cell (a protocol amendment) once T2's cells are in.
- **Open follow-up:** a longer encoder window (56–112 days) was not tested; the exploratory "ctx56"/"ctx112"
  variants only add weekly-mean summaries to the 35-day window.

## Provenance and caveats

- Protocol amendments in this track: v4 (hidden_regimes parameters, `ets_eoq_joint` search space,
  secondary pairs `[jepa, hmm]`, `[jepa, bayes]`), v5 (strength 1.5), v6 (strength 2.0); each logged
  before the runs that use it. T2 amends the same file in parallel, so the version numbers will need
  reconciling at integration.
- Log lines carry `git_dirty_diff_sha256` because `.claude/settings.local.json` was modified in the
  worktree before the track started (not a code change).
- The option-B runs record git sha `e022158`; that commit was re-worded (message only, identical tree) as
  `402bce7` before push.
- The Bayes and oracle planners were not tuned; the textbook arms had 4× the learned budget. Contextual
  arms' tune scores average cells tuned on different worlds, so the G1 rule also required the margin over
  the best textbook arm on dev, which only makes the gate stricter.
