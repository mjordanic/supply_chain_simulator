# T1 G1 dev (strength 1.0)

- split `dev` (not held-out), 300 worlds, seeds 4000000–4000299, variant `base`
- protocol sha256 `7e454de6b8f41b1d…`; SESOI 0.1 pts; BCa 10000 resamples; Holm at α = 0.05
- reference: `tuned_param_eoq_joint` (best tune score among tuned textbook arms)
- families: **exploratory**, chosen by the caller; not the pre-registered ones
- margin capture is the primary metric; cost reduction is `1 − (1 − s_arm)/(1 − s_ref)`, a ratio of means

## Arms

| arm | score | lost_sales_share | fee_share | holding_share | order_days | fill_rate_demand | mean_on_hand |
| --- | --- | --- | --- | --- | --- | --- | --- |
| planner_oracle | 0.9632 | 0.0042 | 0.0134 | 0.0192 | 45.4467 | 0.9958 | 78.5891 |
| planner_bayes | 0.9575 | 0.0064 | 0.0133 | 0.0228 | 45.4167 | 0.9936 | 92.6323 |
| ctx_ets_eoq_joint | 0.95 | 0.0088 | 0.0134 | 0.0277 | 45.5067 | 0.9912 | 114.6162 |
| ctx_eoq_joint | 0.9494 | 0.0088 | 0.0131 | 0.0287 | 44.7333 | 0.9912 | 119.7591 |
| tuned_ets_eoq_joint | 0.9492 | 0.0069 | 0.0135 | 0.0304 | 45.91 | 0.9931 | 114.9211 |
| tuned_param_eoq_joint | 0.9491 | 0.0101 | 0.0149 | 0.0258 | 50.5833 | 0.9899 | 106.0876 |
| tuned_eoq_joint | 0.949 | 0.0082 | 0.0137 | 0.0291 | 46.5667 | 0.9918 | 112.7266 |
| tuned_eoq | 0.9321 | 0.0078 | 0.0227 | 0.0374 | 76.5067 | 0.9922 | 143.5464 |
| tuned_reorder_point | 0.9191 | 0.0083 | 0.0278 | 0.0448 | 89.61 | 0.9917 | 122.4733 |
| tuned_periodic_reorder | 0.918 | 0.0063 | 0.0256 | 0.05 | 82.89 | 0.9937 | 136.7979 |
| tuned_order_up_to | 0.9177 | 0.0078 | 0.0242 | 0.0503 | 78.2767 | 0.9922 | 136.6372 |
| tuned_periodic_order_up_to | 0.9158 | 0.0188 | 0.0258 | 0.0397 | 82.9633 | 0.9812 | 108.0552 |
| never_order | 0.0113 | 0.9886 | 0.0 | 0.0001 | 0.0 | 0.0114 | 0.3399 |

## Primary family (exploratory: families chosen by the caller) (provisional: Holm within this report, not over the whole pre-registered family; final labels come from the pooled primary report)

| comparison | diff, pts [95% BCa] | Holm-adjusted interval | cost reduction [95%] | win rate | Wilcoxon p | Holm p | TOST p | CVaR10, pts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_bayes − tuned_param_eoq_joint | +0.838 [+0.758, +0.927] | [+0.747, +0.940] | +16.5% [+15.6, +17.5] | 98% | 6.5e-50 | 2.4e-53 | 1 | +0.033 |
| planner_oracle − tuned_param_eoq_joint | +1.406 [+1.270, +1.562] | [+1.270, +1.562] | +27.6% [+26.4, +29.1] | 100% | 6.1e-51 | 5.1e-52 | 1 | +0.178 |

## Strata (exploratory, BH at q = 0.1)

| comparison | stratum | n | diff, pts | p (t) | BH q |
| --- | --- | --- | --- | --- | --- |
| planner_bayes − tuned_param_eoq_joint | holding_rate = 0.000684932 | 108 | 0.296 | 1.2e-20 | 3.7e-20 |
| planner_bayes − tuned_param_eoq_joint | holding_rate = 0.00273973 | 95 | 0.739 | 3e-25 | 1.4e-24 |
| planner_bayes − tuned_param_eoq_joint | holding_rate = 0.01 | 97 | 1.54 | 2.8e-34 | 3.3e-33 |
| planner_bayes − tuned_param_eoq_joint | cv = 0.1 | 106 | 0.7 | 2.3e-16 | 4.6e-16 |
| planner_bayes − tuned_param_eoq_joint | cv = 0.3 | 94 | 0.772 | 1.4e-18 | 3.7e-18 |
| planner_bayes − tuned_param_eoq_joint | cv = 0.6 | 100 | 1.047 | 9.6e-23 | 3.8e-22 |
| planner_bayes − tuned_param_eoq_joint | delivery_lag = 2 | 57 | 0.48 | 1.8e-14 | 3.1e-14 |
| planner_bayes − tuned_param_eoq_joint | delivery_lag = 3 | 46 | 0.666 | 1.6e-09 | 1.7e-09 |
| planner_bayes − tuned_param_eoq_joint | delivery_lag = 4 | 45 | 0.601 | 6e-09 | 6e-09 |
| planner_bayes − tuned_param_eoq_joint | delivery_lag = 5 | 49 | 1.063 | 1.2e-12 | 1.6e-12 |
| planner_bayes − tuned_param_eoq_joint | delivery_lag = 6 | 62 | 0.987 | 1.1e-13 | 1.7e-13 |
| planner_bayes − tuned_param_eoq_joint | delivery_lag = 7 | 41 | 1.296 | 6.9e-10 | 8.2e-10 |
| planner_oracle − tuned_param_eoq_joint | holding_rate = 0.000684932 | 108 | 0.422 | 1.8e-28 | 1.1e-27 |
| planner_oracle − tuned_param_eoq_joint | holding_rate = 0.00273973 | 95 | 1.09 | 4.3e-30 | 3.5e-29 |
| planner_oracle − tuned_param_eoq_joint | holding_rate = 0.01 | 97 | 2.809 | 1.5e-38 | 3.5e-37 |
| planner_oracle − tuned_param_eoq_joint | cv = 0.1 | 106 | 1.376 | 2.3e-17 | 5.5e-17 |
| planner_oracle − tuned_param_eoq_joint | cv = 0.3 | 94 | 1.316 | 3.3e-17 | 7.2e-17 |
| planner_oracle − tuned_param_eoq_joint | cv = 0.6 | 100 | 1.52 | 1.2e-20 | 3.7e-20 |
| planner_oracle − tuned_param_eoq_joint | delivery_lag = 2 | 57 | 0.644 | 5.8e-14 | 9.3e-14 |
| planner_oracle − tuned_param_eoq_joint | delivery_lag = 3 | 46 | 0.958 | 6.5e-10 | 8.2e-10 |
| planner_oracle − tuned_param_eoq_joint | delivery_lag = 4 | 45 | 1.091 | 2.8e-09 | 2.9e-09 |
| planner_oracle − tuned_param_eoq_joint | delivery_lag = 5 | 49 | 1.712 | 1.5e-13 | 2.1e-13 |
| planner_oracle − tuned_param_eoq_joint | delivery_lag = 6 | 62 | 1.841 | 1.6e-14 | 3e-14 |
| planner_oracle − tuned_param_eoq_joint | delivery_lag = 7 | 41 | 2.287 | 7.1e-10 | 8.2e-10 |
