# T2 sealed test (no_fee), 1000 worlds

- split `test` (sealed test), 1000 worlds, seeds 9000000–9000999, variant `no_fee`
- protocol sha256 `5b1785bc0d979360…`; SESOI 0.1 pts; BCa 10000 resamples; Holm at α = 0.05
- reference: `ctx2_eoq_joint` (best selection score on common tune worlds among tuned textbook arms)
- families: pre-registered (`comparisons` in the protocol)
- margin capture is the primary metric; cost reduction is `1 − (1 − s_arm)/(1 − s_ref)`, a ratio of means

## Arms

| arm | score | lost_sales_share | fee_share | holding_share | order_days | fill_rate_demand | mean_on_hand |
| --- | --- | --- | --- | --- | --- | --- | --- |
| planner_oracle | 0.99 | 0.0024 | 0.0 | 0.0076 | 360.536 | 0.9976 | 22.4952 |
| jepa_s1 | 0.9853 | 0.005 | 0.0 | 0.0097 | 359.429 | 0.995 | 34.1531 |
| jepa_s0 | 0.9853 | 0.005 | 0.0 | 0.0097 | 359.645 | 0.995 | 33.9682 |
| jepa_s2 | 0.9853 | 0.005 | 0.0 | 0.0097 | 359.411 | 0.995 | 34.1196 |
| direct_s0 | 0.9852 | 0.0045 | 0.0 | 0.0103 | 357.513 | 0.9955 | 35.2911 |
| direct_s1 | 0.9852 | 0.0046 | 0.0 | 0.0103 | 357.231 | 0.9954 | 35.2842 |
| direct_s2 | 0.9851 | 0.0046 | 0.0 | 0.0103 | 357.324 | 0.9954 | 35.335 |
| seasonal_ets | 0.9837 | 0.0048 | 0.0 | 0.0115 | 360.607 | 0.9952 | 39.0037 |
| ctx2_eoq_joint | 0.9834 | 0.0053 | 0.0 | 0.0113 | 364.948 | 0.9947 | 36.9595 |
| ctx_eoq_joint | 0.9834 | 0.0051 | 0.0 | 0.0115 | 364.881 | 0.9949 | 37.7483 |
| param_eoq_joint | 0.9833 | 0.0048 | 0.0 | 0.0119 | 364.879 | 0.9952 | 36.6451 |
| jepa_s0_plain | 0.9831 | 0.0087 | 0.0 | 0.0083 | 345.611 | 0.9913 | 24.8201 |
| planner_moving_average | 0.9827 | 0.005 | 0.0 | 0.0123 | 361.115 | 0.995 | 40.481 |
| tuned_eoq | 0.982 | 0.0045 | 0.0 | 0.0135 | 364.894 | 0.9955 | 35.1055 |
| tuned_eoq_joint | 0.982 | 0.0045 | 0.0 | 0.0135 | 364.894 | 0.9955 | 35.1055 |
| tuned_periodic_order_up_to | 0.9785 | 0.0069 | 0.0 | 0.0147 | 364.575 | 0.9931 | 38.4391 |
| tuned_periodic_reorder | 0.9782 | 0.0074 | 0.0 | 0.0144 | 364.842 | 0.9926 | 37.5587 |
| tuned_reorder_point | 0.9777 | 0.0056 | 0.0 | 0.0166 | 338.782 | 0.9944 | 43.4322 |
| tuned_order_up_to | 0.9771 | 0.0059 | 0.0 | 0.0171 | 364.727 | 0.9941 | 44.6642 |

## Primary family (provisional: Holm within this report, not over the whole pre-registered family; final labels come from the pooled primary report)

| comparison | diff, pts [95% BCa] | Holm-adjusted interval | cost reduction [95%] | win rate | Wilcoxon p | Holm p | TOST p | CVaR10, pts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jepa − ctx2_eoq_joint | +0.186 [+0.171, +0.202] | [+0.169, +0.204] | +11.2% [+10.5, +12.1] | 81% | 4.4e-107 | 1.2e-99 | 1 | -0.097 |
| direct − ctx2_eoq_joint | +0.170 [+0.156, +0.186] | [+0.156, +0.186] | +10.3% [+9.5, +11.1] | 79% | 4.7e-97 | 1.1e-91 | 1 | -0.107 |

## Secondary family

| comparison | diff, pts [95% BCa] | Holm-adjusted interval | cost reduction [95%] | win rate | Wilcoxon p | Holm p | TOST p | CVaR10, pts | label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jepa − direct | +0.016 [+0.011, +0.020] | [+0.011, +0.020] | +1.1% [+0.8, +1.3] | 66% | 1.5e-18 | 2.4e-12 | 2e-198 | -0.088 | superior |
| jepa − seasonal_ets | +0.160 [+0.148, +0.173] | [+0.147, +0.175] | +9.8% [+9.3, +10.4] | 88% | 6.1e-131 | 1.5e-111 | 1 | -0.060 | practically superior |
| jepa − planner_moving_average | +0.260 [+0.243, +0.278] | [+0.239, +0.282] | +15.0% [+14.2, +15.9] | 90% | 6.3e-144 | 3.6e-134 | 1 | -0.044 | practically superior |
| jepa − planner_oracle | -0.468 [-0.491, -0.446] | [-0.498, -0.441] | -46.7% [-49.3, -44.2] | 0% | 3.3e-165 | 1.9e-222 | 1 | -1.256 | inferior |

## Training seeds (two-level bootstrap: worlds and seeds resampled)

| group − reference | mean, pts [95%] | per-seed estimates, pts |
| --- | --- | --- |
| direct − ctx2_eoq_joint | +0.170 [+0.155, +0.186] | direct_s0: +0.174, direct_s1: +0.171, direct_s2: +0.166 |
| jepa − ctx2_eoq_joint | +0.186 [+0.171, +0.202] | jepa_s0: +0.187, jepa_s1: +0.188, jepa_s2: +0.184 |

## Strata (exploratory, BH at q = 0.1)

| comparison | stratum | n | diff, pts | p (t) | BH q |
| --- | --- | --- | --- | --- | --- |
| jepa − ctx2_eoq_joint | holding_rate = 0.000684932 | 329 | 0.029 | 3.4e-06 | 3.6e-06 |
| jepa − ctx2_eoq_joint | holding_rate = 0.00273973 | 343 | 0.145 | 2.2e-54 | 1.7e-53 |
| jepa − ctx2_eoq_joint | holding_rate = 0.01 | 328 | 0.386 | 5e-74 | 1.2e-72 |
| jepa − ctx2_eoq_joint | cv = 0.1 | 331 | 0.26 | 7.1e-47 | 3.4e-46 |
| jepa − ctx2_eoq_joint | cv = 0.3 | 326 | 0.15 | 2e-27 | 5.4e-27 |
| jepa − ctx2_eoq_joint | cv = 0.6 | 343 | 0.149 | 2.6e-31 | 8.8e-31 |
| jepa − ctx2_eoq_joint | delivery_lag = 2 | 148 | 0.158 | 2.5e-17 | 3.5e-17 |
| jepa − ctx2_eoq_joint | delivery_lag = 3 | 182 | 0.193 | 2.5e-20 | 3.7e-20 |
| jepa − ctx2_eoq_joint | delivery_lag = 4 | 174 | 0.231 | 7.2e-23 | 1.4e-22 |
| jepa − ctx2_eoq_joint | delivery_lag = 5 | 169 | 0.207 | 1.3e-24 | 3.1e-24 |
| jepa − ctx2_eoq_joint | delivery_lag = 6 | 159 | 0.161 | 2.9e-14 | 3.6e-14 |
| jepa − ctx2_eoq_joint | delivery_lag = 7 | 168 | 0.16 | 5.9e-12 | 7e-12 |
| direct − ctx2_eoq_joint | holding_rate = 0.000684932 | 329 | 0.024 | 0.00019 | 0.00019 |
| direct − ctx2_eoq_joint | holding_rate = 0.00273973 | 343 | 0.134 | 9.6e-51 | 5.7e-50 |
| direct − ctx2_eoq_joint | holding_rate = 0.01 | 328 | 0.356 | 6e-68 | 7.1e-67 |
| direct − ctx2_eoq_joint | cv = 0.1 | 331 | 0.244 | 9e-46 | 3.6e-45 |
| direct − ctx2_eoq_joint | cv = 0.3 | 326 | 0.152 | 2.7e-29 | 8.2e-29 |
| direct − ctx2_eoq_joint | cv = 0.6 | 343 | 0.117 | 2.6e-22 | 4.8e-22 |
| direct − ctx2_eoq_joint | delivery_lag = 2 | 148 | 0.154 | 7.5e-17 | 1e-16 |
| direct − ctx2_eoq_joint | delivery_lag = 3 | 182 | 0.2 | 7.7e-21 | 1.2e-20 |
| direct − ctx2_eoq_joint | delivery_lag = 4 | 174 | 0.227 | 5.7e-22 | 9.8e-22 |
| direct − ctx2_eoq_joint | delivery_lag = 5 | 169 | 0.173 | 2.9e-24 | 6.3e-24 |
| direct − ctx2_eoq_joint | delivery_lag = 6 | 159 | 0.13 | 1.5e-10 | 1.7e-10 |
| direct − ctx2_eoq_joint | delivery_lag = 7 | 168 | 0.13 | 3e-10 | 3.3e-10 |
