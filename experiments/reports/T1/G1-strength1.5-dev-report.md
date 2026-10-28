# T1 G1 dev (strength 1.5)

- split `dev` (not held-out), 300 worlds, seeds 4000000–4000299, variant `base`
- protocol sha256 `8035cf158a829e2d…`; SESOI 0.1 pts; BCa 10000 resamples; Holm at α = 0.05
- reference: `tuned_param_eoq_joint` (best tune score among tuned textbook arms)
- families: **exploratory**, chosen by the caller; not the pre-registered ones
- margin capture is the primary metric; cost reduction is `1 − (1 − s_arm)/(1 − s_ref)`, a ratio of means

## Arms

| arm | score | lost_sales_share | fee_share | holding_share | order_days | fill_rate_demand | mean_on_hand |
| --- | --- | --- | --- | --- | --- | --- | --- |
| planner_oracle | 0.9635 | 0.0053 | 0.0127 | 0.0185 | 46.3967 | 0.9947 | 81.6908 |
| planner_bayes | 0.952 | 0.0095 | 0.0121 | 0.0264 | 44.7633 | 0.9905 | 112.5979 |
| ctx_ets_eoq_joint | 0.9425 | 0.0152 | 0.0125 | 0.0297 | 45.75 | 0.9848 | 131.7133 |
| tuned_param_eoq_joint | 0.9418 | 0.0139 | 0.0106 | 0.0337 | 38.7867 | 0.9861 | 146.0211 |
| tuned_ets_eoq_joint | 0.9416 | 0.0127 | 0.0121 | 0.0336 | 44.2733 | 0.9873 | 136.6339 |
| ctx_eoq_joint | 0.9415 | 0.0139 | 0.0122 | 0.0323 | 44.9267 | 0.9861 | 148.2972 |
| tuned_eoq_joint | 0.9411 | 0.0137 | 0.0114 | 0.0339 | 41.6633 | 0.9863 | 138.4248 |
| tuned_eoq | 0.9263 | 0.0147 | 0.0231 | 0.0359 | 83.8267 | 0.9853 | 146.3349 |
| tuned_reorder_point | 0.9106 | 0.0138 | 0.024 | 0.0517 | 83.1333 | 0.9862 | 152.0332 |
| tuned_periodic_reorder | 0.9092 | 0.0143 | 0.023 | 0.0535 | 79.6467 | 0.9857 | 156.3767 |
| tuned_order_up_to | 0.9087 | 0.0114 | 0.0234 | 0.0566 | 81.41 | 0.9886 | 164.5696 |
| tuned_periodic_order_up_to | 0.9051 | 0.0243 | 0.0268 | 0.0438 | 92.3267 | 0.9757 | 128.0665 |
| never_order | 0.0106 | 0.9893 | 0.0 | 0.0001 | 0.0 | 0.0107 | 0.3767 |

## Primary family (exploratory: families chosen by the caller) (provisional: Holm within this report, not over the whole pre-registered family; final labels come from the pooled primary report)

| comparison | diff, pts [95% BCa] | Holm-adjusted interval | cost reduction [95%] | win rate | Wilcoxon p | Holm p | TOST p | CVaR10, pts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_bayes − tuned_param_eoq_joint | +1.011 [+0.909, +1.125] | [+0.909, +1.125] | +17.4% [+16.1, +18.6] | 95% | 6.9e-47 | 4.1e-50 | 1 | -0.111 |
| planner_oracle − tuned_param_eoq_joint | +2.162 [+1.955, +2.407] | [+1.927, +2.443] | +37.2% [+35.7, +38.8] | 100% | 6.1e-51 | 7.8e-52 | 1 | +0.267 |

## Strata (exploratory, BH at q = 0.1)

| comparison | stratum | n | diff, pts | p (t) | BH q |
| --- | --- | --- | --- | --- | --- |
| planner_bayes − tuned_param_eoq_joint | holding_rate = 0.000684932 | 108 | 0.405 | 1e-17 | 2.3e-17 |
| planner_bayes − tuned_param_eoq_joint | holding_rate = 0.00273973 | 95 | 0.913 | 1.4e-22 | 6.8e-22 |
| planner_bayes − tuned_param_eoq_joint | holding_rate = 0.01 | 97 | 1.783 | 1.6e-28 | 1.3e-27 |
| planner_bayes − tuned_param_eoq_joint | cv = 0.1 | 106 | 0.875 | 4.9e-15 | 9.7e-15 |
| planner_bayes − tuned_param_eoq_joint | cv = 0.3 | 94 | 0.925 | 3.9e-18 | 1.1e-17 |
| planner_bayes − tuned_param_eoq_joint | cv = 0.6 | 100 | 1.237 | 5.4e-21 | 2.2e-20 |
| planner_bayes − tuned_param_eoq_joint | delivery_lag = 2 | 57 | 0.67 | 2.6e-14 | 4.1e-14 |
| planner_bayes − tuned_param_eoq_joint | delivery_lag = 3 | 46 | 0.771 | 4e-08 | 4e-08 |
| planner_bayes − tuned_param_eoq_joint | delivery_lag = 4 | 45 | 0.83 | 2.7e-08 | 2.8e-08 |
| planner_bayes − tuned_param_eoq_joint | delivery_lag = 5 | 49 | 1.252 | 2.5e-11 | 3.3e-11 |
| planner_bayes − tuned_param_eoq_joint | delivery_lag = 6 | 62 | 1.133 | 7.7e-13 | 1.1e-12 |
| planner_bayes − tuned_param_eoq_joint | delivery_lag = 7 | 41 | 1.482 | 2e-08 | 2.2e-08 |
| planner_oracle − tuned_param_eoq_joint | holding_rate = 0.000684932 | 108 | 0.652 | 1.4e-26 | 8.3e-26 |
| planner_oracle − tuned_param_eoq_joint | holding_rate = 0.00273973 | 95 | 1.664 | 2.9e-34 | 3.5e-33 |
| planner_oracle − tuned_param_eoq_joint | holding_rate = 0.01 | 97 | 4.332 | 3.5e-38 | 8.4e-37 |
| planner_oracle − tuned_param_eoq_joint | cv = 0.1 | 106 | 2.144 | 2.8e-18 | 8.4e-18 |
| planner_oracle − tuned_param_eoq_joint | cv = 0.3 | 94 | 2.068 | 1e-17 | 2.3e-17 |
| planner_oracle − tuned_param_eoq_joint | cv = 0.6 | 100 | 2.27 | 4e-19 | 1.4e-18 |
| planner_oracle − tuned_param_eoq_joint | delivery_lag = 2 | 57 | 1.028 | 1.6e-14 | 2.7e-14 |
| planner_oracle − tuned_param_eoq_joint | delivery_lag = 3 | 46 | 1.455 | 1.6e-09 | 1.9e-09 |
| planner_oracle − tuned_param_eoq_joint | delivery_lag = 4 | 45 | 1.724 | 2.2e-09 | 2.5e-09 |
| planner_oracle − tuned_param_eoq_joint | delivery_lag = 5 | 49 | 2.557 | 4.9e-14 | 7.4e-14 |
| planner_oracle − tuned_param_eoq_joint | delivery_lag = 6 | 62 | 2.835 | 1.2e-14 | 2.2e-14 |
| planner_oracle − tuned_param_eoq_joint | delivery_lag = 7 | 41 | 3.525 | 1.5e-09 | 1.8e-09 |
