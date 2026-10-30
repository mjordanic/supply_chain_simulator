# T2 dev (one_product), 100 worlds

- split `dev` (not held-out), 100 worlds, seeds 4000000–4000099, variant `one_product`
- protocol sha256 `5b1785bc0d979360…`; SESOI 0.1 pts; BCa 10000 resamples; Holm at α = 0.05
- reference: `param_eoq_joint` (best selection score on common tune worlds among tuned textbook arms)
- families: pre-registered (`comparisons` in the protocol)
- margin capture is the primary metric; cost reduction is `1 − (1 − s_arm)/(1 − s_ref)`, a ratio of means

## Arms

| arm | score | lost_sales_share | fee_share | holding_share | order_days | fill_rate_demand | mean_on_hand |
| --- | --- | --- | --- | --- | --- | --- | --- |
| planner_oracle | 0.9262 | 0.0025 | 0.0409 | 0.0305 | 30.21 | 0.9975 | 82.9887 |
| direct_s2 | 0.9249 | 0.004 | 0.0399 | 0.0312 | 29.7 | 0.996 | 86.0759 |
| direct_s0 | 0.9248 | 0.004 | 0.0397 | 0.0315 | 29.54 | 0.996 | 86.3438 |
| jepa_s0 | 0.9243 | 0.0035 | 0.0401 | 0.0321 | 29.86 | 0.9965 | 86.8606 |
| direct_s1 | 0.9243 | 0.0045 | 0.04 | 0.0313 | 29.7 | 0.9955 | 85.8987 |
| jepa_s1 | 0.924 | 0.004 | 0.0399 | 0.0321 | 29.79 | 0.996 | 86.8971 |
| jepa_s2 | 0.9239 | 0.0041 | 0.0399 | 0.032 | 29.77 | 0.9959 | 86.7283 |
| jepa_s0_plain | 0.9239 | 0.0053 | 0.0403 | 0.0305 | 29.57 | 0.9947 | 82.1014 |
| ctx2_eoq_joint | 0.9231 | 0.0052 | 0.0392 | 0.0325 | 29.34 | 0.9948 | 88.0292 |
| tuned_eoq_joint | 0.9231 | 0.0063 | 0.0388 | 0.0318 | 29.01 | 0.9937 | 86.1961 |
| tuned_eoq | 0.9231 | 0.0063 | 0.0388 | 0.0318 | 29.01 | 0.9937 | 86.1961 |
| seasonal_ets | 0.9228 | 0.0037 | 0.0403 | 0.0333 | 30.07 | 0.9963 | 88.7865 |
| param_eoq_joint | 0.9227 | 0.0065 | 0.0392 | 0.0316 | 29.26 | 0.9935 | 86.2776 |
| ctx_eoq_joint | 0.9227 | 0.0052 | 0.0384 | 0.0337 | 28.67 | 0.9948 | 89.6787 |
| planner_moving_average | 0.9221 | 0.006 | 0.0401 | 0.0318 | 29.85 | 0.994 | 86.8917 |
| tuned_periodic_reorder | 0.9154 | 0.0051 | 0.0374 | 0.0421 | 26.98 | 0.9949 | 96.1953 |
| tuned_order_up_to | 0.9148 | 0.0071 | 0.0396 | 0.0384 | 28.15 | 0.9929 | 89.3766 |
| tuned_reorder_point | 0.9146 | 0.0067 | 0.0386 | 0.0402 | 27.65 | 0.9933 | 92.9298 |
| tuned_periodic_order_up_to | 0.8684 | 0.0295 | 0.0676 | 0.0345 | 44.63 | 0.9705 | 80.6991 |

## Primary family (provisional: Holm within this report, not over the whole pre-registered family; final labels come from the pooled primary report)

| comparison | diff, pts [95% BCa] | Holm-adjusted interval | cost reduction [95%] | win rate | Wilcoxon p | Holm p | TOST p | CVaR10, pts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jepa − param_eoq_joint | +0.134 [+0.042, +0.234] | [+0.042, +0.234] | +1.7% [+0.5, +3.0] | 59% | 0.02 | 0.008 | 0.75 | -0.644 |
| direct − param_eoq_joint | +0.192 [+0.104, +0.293] | [+0.089, +0.308] | +2.5% [+1.4, +3.7] | 68% | 0.00022 | 0.00031 | 0.97 | -0.528 |

## Secondary family

| comparison | diff, pts [95% BCa] | Holm-adjusted interval | cost reduction [95%] | win rate | Wilcoxon p | Holm p | TOST p | CVaR10, pts | label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jepa − direct | -0.058 [-0.098, -0.027] | [-0.098, -0.027] | -0.8% [-1.3, -0.4] | 40% | 0.0058 | 0.0015 | 0.0089 | -0.447 | inferior |
| jepa − seasonal_ets | +0.132 [+0.077, +0.188] | [+0.068, +0.196] | +1.7% [+1.0, +2.3] | 70% | 1.6e-06 | 1.9e-05 | 0.87 | -0.298 | practically superior |
| jepa − planner_moving_average | +0.195 [+0.121, +0.278] | [+0.103, +0.297] | +2.5% [+1.6, +3.5] | 74% | 4.5e-07 | 1.5e-05 | 0.99 | -0.413 | practically superior |
| jepa − planner_oracle | -0.209 [-0.283, -0.123] | [-0.301, -0.100] | -2.8% [-3.7, -1.7] | 23% | 6.5e-09 | 3.8e-06 | 1 | -0.913 | inferior |

## Training seeds (two-level bootstrap: worlds and seeds resampled)

| group − reference | mean, pts [95%] | per-seed estimates, pts |
| --- | --- | --- |
| direct − param_eoq_joint | +0.192 [+0.091, +0.294] | direct_s0: +0.203, direct_s1: +0.156, direct_s2: +0.216 |
| jepa − param_eoq_joint | +0.134 [+0.034, +0.235] | jepa_s0: +0.158, jepa_s1: +0.128, jepa_s2: +0.116 |

## Strata (exploratory, BH at q = 0.1)

| comparison | stratum | n | diff, pts | p (t) | BH q |
| --- | --- | --- | --- | --- | --- |
| jepa − param_eoq_joint | holding_rate = 0.000684932 | 32 | -0.019 | 0.81 | 0.89 |
| jepa − param_eoq_joint | holding_rate = 0.00273973 | 34 | 0.206 | 0.0014 | 0.017 |
| jepa − param_eoq_joint | holding_rate = 0.01 | 34 | 0.206 | 0.064 | 0.15 |
| jepa − param_eoq_joint | cv = 0.1 | 36 | 0.222 | 0.018 | 0.073 |
| jepa − param_eoq_joint | cv = 0.3 | 30 | 0.035 | 0.5 | 0.57 |
| jepa − param_eoq_joint | cv = 0.6 | 34 | 0.128 | 0.21 | 0.34 |
| jepa − param_eoq_joint | delivery_lag = 2 | 16 | 0.083 | 0.16 | 0.27 |
| jepa − param_eoq_joint | delivery_lag = 3 | 16 | 0.119 | 0.36 | 0.48 |
| jepa − param_eoq_joint | delivery_lag = 4 | 15 | 0.127 | 0.39 | 0.49 |
| jepa − param_eoq_joint | delivery_lag = 5 | 13 | 0.198 | 0.27 | 0.39 |
| jepa − param_eoq_joint | delivery_lag = 6 | 25 | 0.216 | 0.064 | 0.15 |
| jepa − param_eoq_joint | delivery_lag = 7 | 15 | 0.02 | 0.87 | 0.91 |
| direct − param_eoq_joint | holding_rate = 0.000684932 | 32 | 0.005 | 0.95 | 0.95 |
| direct − param_eoq_joint | holding_rate = 0.00273973 | 34 | 0.212 | 0.0003 | 0.0072 |
| direct − param_eoq_joint | holding_rate = 0.01 | 34 | 0.346 | 0.0021 | 0.017 |
| direct − param_eoq_joint | cv = 0.1 | 36 | 0.228 | 0.024 | 0.083 |
| direct − param_eoq_joint | cv = 0.3 | 30 | 0.097 | 0.071 | 0.16 |
| direct − param_eoq_joint | cv = 0.6 | 34 | 0.236 | 0.012 | 0.062 |
| direct − param_eoq_joint | delivery_lag = 2 | 16 | 0.11 | 0.048 | 0.15 |
| direct − param_eoq_joint | delivery_lag = 3 | 16 | 0.148 | 0.24 | 0.36 |
| direct − param_eoq_joint | delivery_lag = 4 | 15 | 0.124 | 0.41 | 0.49 |
| direct − param_eoq_joint | delivery_lag = 5 | 13 | 0.31 | 0.14 | 0.26 |
| direct − param_eoq_joint | delivery_lag = 6 | 25 | 0.279 | 0.013 | 0.062 |
| direct − param_eoq_joint | delivery_lag = 7 | 15 | 0.145 | 0.13 | 0.26 |
