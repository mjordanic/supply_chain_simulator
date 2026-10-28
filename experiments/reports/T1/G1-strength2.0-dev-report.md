# T1 G1 dev (strength 2.0)

- split `dev` (not held-out), 300 worlds, seeds 4000000–4000299, variant `base`
- protocol sha256 `c3831f5f6e6d2c03…`; SESOI 0.1 pts; BCa 10000 resamples; Holm at α = 0.05
- reference: `tuned_ets_eoq_joint` (best tune score among tuned textbook arms)
- families: **exploratory**, chosen by the caller; not the pre-registered ones
- margin capture is the primary metric; cost reduction is `1 − (1 − s_arm)/(1 − s_ref)`, a ratio of means

## Arms

| arm | score | lost_sales_share | fee_share | holding_share | order_days | fill_rate_demand | mean_on_hand |
| --- | --- | --- | --- | --- | --- | --- | --- |
| planner_oracle | 0.9638 | 0.0066 | 0.0118 | 0.0178 | 47.7367 | 0.9934 | 86.6259 |
| planner_bayes | 0.946 | 0.0116 | 0.0108 | 0.0316 | 44.4033 | 0.9884 | 139.6788 |
| ctx_ets_eoq_joint | 0.9332 | 0.0207 | 0.0104 | 0.0358 | 42.4233 | 0.9793 | 191.2872 |
| ctx_eoq_joint | 0.933 | 0.0197 | 0.01 | 0.0373 | 40.8533 | 0.9803 | 191.3837 |
| tuned_eoq_joint | 0.9329 | 0.0183 | 0.0092 | 0.0396 | 37.43 | 0.9817 | 176.4756 |
| tuned_param_eoq_joint | 0.9326 | 0.0189 | 0.0096 | 0.0388 | 39.2533 | 0.9811 | 173.6847 |
| tuned_ets_eoq_joint | 0.9323 | 0.0184 | 0.0116 | 0.0378 | 47.1133 | 0.9816 | 165.2276 |
| tuned_eoq | 0.9184 | 0.0199 | 0.0206 | 0.0411 | 83.1067 | 0.9801 | 177.6749 |
| tuned_reorder_point | 0.9009 | 0.0209 | 0.0237 | 0.0546 | 90.0167 | 0.9791 | 176.7746 |
| tuned_order_up_to | 0.8981 | 0.0174 | 0.0191 | 0.0654 | 73.38 | 0.9826 | 210.5541 |
| tuned_periodic_reorder | 0.8979 | 0.026 | 0.02 | 0.0561 | 76.0467 | 0.974 | 181.4583 |
| tuned_periodic_order_up_to | 0.8902 | 0.0223 | 0.0293 | 0.0582 | 111.0 | 0.9777 | 187.6307 |
| never_order | 0.0097 | 0.9902 | 0.0 | 0.0001 | 0.0 | 0.0098 | 0.4271 |

## Primary family (exploratory: families chosen by the caller) (provisional: Holm within this report, not over the whole pre-registered family; final labels come from the pooled primary report)

| comparison | diff, pts [95% BCa] | Holm-adjusted interval | cost reduction [95%] | win rate | Wilcoxon p | Holm p | TOST p | CVaR10, pts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| planner_bayes − tuned_ets_eoq_joint | +1.376 [+1.239, +1.524] | [+1.239, +1.524] | +20.3% [+18.8, +21.7] | 94% | 2.5e-46 | 1.1e-50 | 1 | -0.255 |
| planner_oracle − tuned_ets_eoq_joint | +3.155 [+2.846, +3.518] | [+2.806, +3.574] | +46.6% [+44.9, +48.3] | 100% | 6.1e-51 | 1.1e-50 | 1 | +0.358 |

## Strata (exploratory, BH at q = 0.1)

| comparison | stratum | n | diff, pts | p (t) | BH q |
| --- | --- | --- | --- | --- | --- |
| planner_bayes − tuned_ets_eoq_joint | holding_rate = 0.000684932 | 108 | 0.707 | 9.4e-19 | 3.2e-18 |
| planner_bayes − tuned_ets_eoq_joint | holding_rate = 0.00273973 | 95 | 1.201 | 1.5e-18 | 3.9e-18 |
| planner_bayes − tuned_ets_eoq_joint | holding_rate = 0.01 | 97 | 2.293 | 4.2e-27 | 3.4e-26 |
| planner_bayes − tuned_ets_eoq_joint | cv = 0.1 | 106 | 1.133 | 5.1e-14 | 8.8e-14 |
| planner_bayes − tuned_ets_eoq_joint | cv = 0.3 | 94 | 1.22 | 7.1e-20 | 2.8e-19 |
| planner_bayes − tuned_ets_eoq_joint | cv = 0.6 | 100 | 1.78 | 1.6e-22 | 7.6e-22 |
| planner_bayes − tuned_ets_eoq_joint | delivery_lag = 2 | 57 | 0.663 | 1e-10 | 1.3e-10 |
| planner_bayes − tuned_ets_eoq_joint | delivery_lag = 3 | 46 | 1.086 | 4.5e-09 | 4.7e-09 |
| planner_bayes − tuned_ets_eoq_joint | delivery_lag = 4 | 45 | 1.104 | 2.1e-09 | 2.4e-09 |
| planner_bayes − tuned_ets_eoq_joint | delivery_lag = 5 | 49 | 1.742 | 1.4e-12 | 2e-12 |
| planner_bayes − tuned_ets_eoq_joint | delivery_lag = 6 | 62 | 1.713 | 1.2e-15 | 2.5e-15 |
| planner_bayes − tuned_ets_eoq_joint | delivery_lag = 7 | 41 | 2.046 | 3.4e-08 | 3.4e-08 |
| planner_oracle − tuned_ets_eoq_joint | holding_rate = 0.000684932 | 108 | 1.07 | 1.4e-25 | 8.5e-25 |
| planner_oracle − tuned_ets_eoq_joint | holding_rate = 0.00273973 | 95 | 2.295 | 1.4e-28 | 1.7e-27 |
| planner_oracle − tuned_ets_eoq_joint | holding_rate = 0.01 | 97 | 6.318 | 6.8e-37 | 1.6e-35 |
| planner_oracle − tuned_ets_eoq_joint | cv = 0.1 | 106 | 2.965 | 3.1e-18 | 7.4e-18 |
| planner_oracle − tuned_ets_eoq_joint | cv = 0.3 | 94 | 2.99 | 3.1e-17 | 6.8e-17 |
| planner_oracle − tuned_ets_eoq_joint | cv = 0.6 | 100 | 3.51 | 1.2e-18 | 3.6e-18 |
| planner_oracle − tuned_ets_eoq_joint | delivery_lag = 2 | 57 | 1.267 | 9.1e-13 | 1.4e-12 |
| planner_oracle − tuned_ets_eoq_joint | delivery_lag = 3 | 46 | 2.251 | 2.5e-09 | 2.8e-09 |
| planner_oracle − tuned_ets_eoq_joint | delivery_lag = 4 | 45 | 2.583 | 2.1e-10 | 2.6e-10 |
| planner_oracle − tuned_ets_eoq_joint | delivery_lag = 5 | 49 | 3.663 | 1.5e-15 | 2.7e-15 |
| planner_oracle − tuned_ets_eoq_joint | delivery_lag = 6 | 62 | 4.158 | 5.6e-14 | 8.9e-14 |
| planner_oracle − tuned_ets_eoq_joint | delivery_lag = 7 | 41 | 5.295 | 1.3e-09 | 1.6e-09 |
