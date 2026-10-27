# H0 acceptance

Harness commit `2e6ea1d`, protocol sha256 `8aef3f9e96e33c50…`. Log lines are in
`experiments/log/H0-harness.jsonl`. Their `git_dirty_diff_sha256` is a local
`.claude/settings.local.json` edit, not code.

## 1. Re-scoring the stored base comparison on the dev block

The stored arms were the JEPA planner (`runs/perfect-supplier/models/jepa.pt`) and tuned
joint EOQ (`runs/perfect-supplier/tuning/best_params.json`). Both were run through the harness
on 500 dev worlds that nothing had touched before, seeds 4,000,000–4,000,499, variant `base`.

| | JEPA − tuned joint EOQ, pts |
| --- | --- |
| stored result, 500 burnt eval worlds (percentile bootstrap, notebook 15) | +0.242 [+0.219, +0.265] |
| stored result, recomputed by the harness (BCa, 10,000 resamples) | +0.242 [+0.220, +0.266] |
| **harness, dev block** | **+0.252 [+0.229, +0.275]** |

The dev estimate, +0.252, is inside the stored interval. **Accepted.** The cost reduction is
+5.7% [+5.3, +6.1] on dev, against 5.6% stored.

Parity check: on eval world 3,000,000 the harness reproduces both stored scores exactly
(jepa 0.964524, tuned joint EOQ 0.963701).

The scope of this result is limited. It shows the harness reproduces the stored comparison.
It is not a new claim, for three reasons:

- the arms are the old ones;
- the tuning budgets were unequal (planner 0 trials);
- only one training seed was used.

Tracks T2 and T1 make the claims.

## 2. A sealed run without a frozen config is refused

```
$ python -m src.perfect_supplier.bench eval --arms tuned_eoq_joint.json --split test --n-worlds 1000 ...
GateError: refusing test run: config of arm 'tuned_eoq_joint' (76ab480c1dae) is not frozen
```

No episode ran, and no output directory was created. The refusal is logged as
`status: refused`. Unit tests cover three more cases on a stand-in sealed block, never
the real test seeds:

- a frozen config runs once;
- a second run without `--rerun-reason` is refused;
- a config frozen under another protocol is refused.

## 3. Statistics unit tests

`tests/perfect_supplier/test_bench_stats.py` covers known cases:

- BCa against the t-interval and against scipy's BCa, including skewed data and the paired
  cost-reduction ratio;
- the Wilcoxon exact p = 0.0625;
- Holm and BH hand-computed values, also checked against statsmodels;
- TOST;
- the two-level bootstrap reducing to the world bootstrap;
- CVaR;
- the protocol labels.

`uv run pytest tests/perfect_supplier` gives 58 passed.

## 4. Diagnostic: lead-time calibration of the JEPA planner (dev, 60 worlds, a decision every 5 days)

| span | q0.90 exceed (nominal 0.10) | q0.95 exceed (0.05) | q0.99 exceed (0.01) | PIT sd (uniform 0.289) | n |
| --- | --- | --- | --- | --- | --- |
| lag | 0.117 | 0.076 | 0.035 | 0.291 | 17,396 |
| lag+7 | 0.154 | 0.109 | 0.063 | 0.313 | 17,080 |

This reproduces the audit (§3.2(e): 4.0% and 6.6% at q0.99). The planner's imagined lead-time
demand has too thin an upper tail. Fixing that belongs to T2.

## 5. Full report of the dev re-score


- split `dev` (not held-out), 500 worlds, seeds 4000000–4000499, variant `base`
- protocol sha256 `8aef3f9e96e33c50…`; SESOI 0.1 pts; BCa 10000 resamples; Holm at α = 0.05
- reference: `tuned_eoq_joint` (best tune score among tuned textbook arms)
- margin capture is the primary metric; cost reduction is `1 − (1 − s_arm)/(1 − s_ref)`, a ratio of means

## Arms

| arm | score | lost_sales_share | fee_share | holding_share | order_days | fill_rate_demand | mean_on_hand |
| --- | --- | --- | --- | --- | --- | --- | --- |
| jepa | 0.9585 | 0.0047 | 0.0153 | 0.0215 | 49.45 | 0.9953 | 75.3192 |
| tuned_eoq_joint | 0.956 | 0.0049 | 0.0163 | 0.0229 | 52.662 | 0.9951 | 79.8574 |

## Primary family

| comparison | diff, pts [95% BCa] | Holm-adjusted interval | cost reduction [95%] | win rate | Wilcoxon p | Holm p | TOST p | CVaR10, pts | label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jepa − tuned_eoq_joint | +0.252 [+0.229, +0.275] | [+0.229, +0.275] | +5.7% [+5.3, +6.1] | 87% | 4.9e-68 | 3.7e-71 | 1 | -0.088 | practically superior |

## Strata (exploratory, BH at q = 0.1)

| comparison | stratum | n | diff, pts | p (t) | BH q |
| --- | --- | --- | --- | --- | --- |
| jepa − tuned_eoq_joint | holding_rate = 0.000684932 | 173 | 0.085 | 3.1e-14 | 3.1e-14 |
| jepa − tuned_eoq_joint | holding_rate = 0.00273973 | 156 | 0.209 | 2.6e-30 | 5.1e-30 |
| jepa − tuned_eoq_joint | holding_rate = 0.01 | 171 | 0.46 | 1.5e-47 | 8.9e-47 |
| jepa − tuned_eoq_joint | cv = 0.1 | 172 | 0.268 | 9.5e-32 | 2.8e-31 |
| jepa − tuned_eoq_joint | cv = 0.3 | 164 | 0.186 | 3.2e-20 | 3.9e-20 |
| jepa − tuned_eoq_joint | cv = 0.6 | 164 | 0.302 | 6.3e-25 | 9.5e-25 |
