# T2 sealed test (one_product), 1000 worlds

- split `test` (sealed test), 1000 worlds, seeds 9000000–9000999, variant `one_product`
- protocol sha256 `5b1785bc0d979360…`; SESOI 0.1 pts; BCa 10000 resamples; Holm at α = 0.05
- reference: `param_eoq_joint` (best selection score on common tune worlds among tuned textbook arms)
- families: pre-registered (`comparisons` in the protocol)
- margin capture is the primary metric; cost reduction is `1 − (1 − s_arm)/(1 − s_ref)`, a ratio of means

## Arms

| arm | score | lost_sales_share | fee_share | holding_share | order_days | fill_rate_demand | mean_on_hand |
| --- | --- | --- | --- | --- | --- | --- | --- |
| planner_oracle | 0.9268 | 0.0027 | 0.0407 | 0.0298 | 30.091 | 0.9973 | 84.4138 |
| direct_s1 | 0.9248 | 0.0045 | 0.0402 | 0.0305 | 29.873 | 0.9955 | 87.3408 |
| direct_s2 | 0.9248 | 0.0045 | 0.0402 | 0.0305 | 29.883 | 0.9955 | 87.2814 |
| jepa_s2 | 0.9247 | 0.0039 | 0.0402 | 0.0311 | 29.953 | 0.9961 | 87.8457 |
| jepa_s0 | 0.9247 | 0.0038 | 0.0402 | 0.0312 | 29.935 | 0.9962 | 87.9556 |
| jepa_s1 | 0.9247 | 0.0039 | 0.0402 | 0.0312 | 29.944 | 0.9961 | 88.0369 |
| direct_s0 | 0.9246 | 0.0046 | 0.0402 | 0.0306 | 29.855 | 0.9954 | 87.4578 |
| jepa_s0_plain | 0.9241 | 0.0056 | 0.0403 | 0.03 | 29.591 | 0.9944 | 83.5945 |
| seasonal_ets | 0.9236 | 0.0036 | 0.0406 | 0.0322 | 30.319 | 0.9964 | 89.9257 |
| param_eoq_joint | 0.9234 | 0.0062 | 0.0392 | 0.0312 | 29.243 | 0.9938 | 87.8757 |
| ctx_eoq_joint | 0.9234 | 0.005 | 0.0384 | 0.0332 | 28.748 | 0.995 | 90.8525 |
| tuned_eoq | 0.9233 | 0.0064 | 0.0388 | 0.0315 | 28.918 | 0.9936 | 87.9916 |
| tuned_eoq_joint | 0.9233 | 0.0064 | 0.0388 | 0.0315 | 28.918 | 0.9936 | 87.9916 |
| ctx2_eoq_joint | 0.9232 | 0.0056 | 0.039 | 0.0323 | 29.171 | 0.9944 | 90.4574 |
| planner_moving_average | 0.923 | 0.0056 | 0.0404 | 0.0311 | 30.05 | 0.9944 | 88.5196 |
| tuned_periodic_reorder | 0.9167 | 0.0061 | 0.0378 | 0.0393 | 27.352 | 0.9939 | 95.3777 |
| tuned_order_up_to | 0.9163 | 0.0071 | 0.0404 | 0.0362 | 28.69 | 0.9929 | 89.1804 |
| tuned_reorder_point | 0.9162 | 0.0064 | 0.0395 | 0.0378 | 28.466 | 0.9936 | 92.5902 |
| tuned_periodic_order_up_to | 0.8665 | 0.0318 | 0.0692 | 0.0324 | 44.654 | 0.9682 | 80.6902 |

## Primary family (provisional: Holm within this report, not over the whole pre-registered family; final labels come from the pooled primary report)

| comparison | diff, pts [95% BCa] | Holm-adjusted interval | cost reduction [95%] | win rate | Wilcoxon p | Holm p | TOST p | CVaR10, pts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jepa − param_eoq_joint | +0.132 [+0.106, +0.160] | [+0.106, +0.160] | +1.7% [+1.4, +2.1] | 59% | 3.5e-16 | 6.1e-21 | 0.99 | -0.499 |
| direct − param_eoq_joint | +0.135 [+0.110, +0.164] | [+0.106, +0.168] | +1.8% [+1.4, +2.1] | 61% | 1.2e-17 | 1e-21 | 1 | -0.502 |

## Secondary family

| comparison | diff, pts [95% BCa] | Holm-adjusted interval | cost reduction [95%] | win rate | Wilcoxon p | Holm p | TOST p | CVaR10, pts | label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jepa − direct | -0.004 [-0.015, +0.008] | [-0.015, +0.008] | -0.0% [-0.2, +0.1] | 49% | 0.59 | 0.55 | 1.3e-50 | -0.366 | equivalent |
| jepa − seasonal_ets | +0.109 [+0.092, +0.125] | [+0.090, +0.128] | +1.4% [+1.2, +1.6] | 65% | 2.6e-33 | 3.6e-34 | 0.84 | -0.295 | practically superior |
| jepa − planner_moving_average | +0.174 [+0.152, +0.198] | [+0.147, +0.203] | +2.3% [+2.0, +2.6] | 69% | 1.1e-46 | 6.5e-44 | 1 | -0.371 | practically superior |
| jepa − planner_oracle | -0.207 [-0.229, -0.187] | [-0.234, -0.181] | -2.8% [-3.1, -2.6] | 22% | 4.2e-81 | 3.1e-68 | 1 | -0.946 | inferior |

## Training seeds (two-level bootstrap: worlds and seeds resampled)

| group − reference | mean, pts [95%] | per-seed estimates, pts |
| --- | --- | --- |
| direct − param_eoq_joint | +0.135 [+0.107, +0.165] | direct_s0: +0.124, direct_s1: +0.142, direct_s2: +0.141 |
| jepa − param_eoq_joint | +0.132 [+0.105, +0.160] | jepa_s0: +0.132, jepa_s1: +0.130, jepa_s2: +0.133 |

## Strata (exploratory, BH at q = 0.1)

| comparison | stratum | n | diff, pts | p (t) | BH q |
| --- | --- | --- | --- | --- | --- |
| jepa − param_eoq_joint | holding_rate = 0.000684932 | 329 | -0.012 | 0.53 | 0.55 |
| jepa − param_eoq_joint | holding_rate = 0.00273973 | 343 | 0.114 | 4.3e-09 | 2e-08 |
| jepa − param_eoq_joint | holding_rate = 0.01 | 328 | 0.294 | 1.4e-20 | 1.6e-19 |
| jepa − param_eoq_joint | cv = 0.1 | 331 | 0.181 | 8.7e-15 | 7e-14 |
| jepa − param_eoq_joint | cv = 0.3 | 326 | 0.107 | 3.4e-08 | 1.4e-07 |
| jepa − param_eoq_joint | cv = 0.6 | 343 | 0.108 | 0.00018 | 0.00027 |
| jepa − param_eoq_joint | delivery_lag = 2 | 148 | 0.057 | 0.0057 | 0.0065 |
| jepa − param_eoq_joint | delivery_lag = 3 | 182 | 0.145 | 1.2e-06 | 2.5e-06 |
| jepa − param_eoq_joint | delivery_lag = 4 | 174 | 0.162 | 2.8e-07 | 8.3e-07 |
| jepa − param_eoq_joint | delivery_lag = 5 | 169 | 0.106 | 0.00038 | 0.00047 |
| jepa − param_eoq_joint | delivery_lag = 6 | 159 | 0.132 | 0.0025 | 0.003 |
| jepa − param_eoq_joint | delivery_lag = 7 | 168 | 0.178 | 6.1e-05 | 9.7e-05 |
| direct − param_eoq_joint | holding_rate = 0.000684932 | 329 | -0.008 | 0.66 | 0.66 |
| direct − param_eoq_joint | holding_rate = 0.00273973 | 343 | 0.098 | 7.9e-07 | 2e-06 |
| direct − param_eoq_joint | holding_rate = 0.01 | 328 | 0.318 | 1e-24 | 2.5e-23 |
| direct − param_eoq_joint | cv = 0.1 | 331 | 0.178 | 3.2e-14 | 1.9e-13 |
| direct − param_eoq_joint | cv = 0.3 | 326 | 0.109 | 4.7e-08 | 1.6e-07 |
| direct − param_eoq_joint | cv = 0.6 | 343 | 0.12 | 2.5e-05 | 4.6e-05 |
| direct − param_eoq_joint | delivery_lag = 2 | 148 | 0.054 | 0.011 | 0.012 |
| direct − param_eoq_joint | delivery_lag = 3 | 182 | 0.147 | 1.5e-06 | 3e-06 |
| direct − param_eoq_joint | delivery_lag = 4 | 174 | 0.163 | 8.5e-07 | 2e-06 |
| direct − param_eoq_joint | delivery_lag = 5 | 169 | 0.108 | 0.00034 | 0.00047 |
| direct − param_eoq_joint | delivery_lag = 6 | 159 | 0.158 | 0.00038 | 0.00047 |
| direct − param_eoq_joint | delivery_lag = 7 | 168 | 0.173 | 2.8e-05 | 4.9e-05 |
