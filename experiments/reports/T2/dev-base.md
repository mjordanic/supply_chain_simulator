# T2 dev (base), 100 worlds

- split `dev` (not held-out), 100 worlds, seeds 4000000–4000099, variant `base`
- protocol sha256 `5b1785bc0d979360…`; SESOI 0.1 pts; BCa 10000 resamples; Holm at α = 0.05
- reference: `ctx2_eoq_joint` (best selection score on common tune worlds among tuned textbook arms)
- families: pre-registered (`comparisons` in the protocol)
- margin capture is the primary metric; cost reduction is `1 − (1 − s_arm)/(1 − s_ref)`, a ratio of means

## Arms

| arm | score | lost_sales_share | fee_share | holding_share | order_days | fill_rate_demand | mean_on_hand |
| --- | --- | --- | --- | --- | --- | --- | --- |
| planner_oracle | 0.9611 | 0.0029 | 0.0149 | 0.0211 | 47.73 | 0.9971 | 69.0772 |
| jepa_s0_plain | 0.9569 | 0.0048 | 0.0155 | 0.0228 | 49.32 | 0.9952 | 73.9446 |
| direct_s2 | 0.9569 | 0.0045 | 0.0155 | 0.023 | 49.73 | 0.9955 | 74.8287 |
| direct_s0 | 0.9569 | 0.0045 | 0.0155 | 0.0231 | 49.61 | 0.9955 | 74.9234 |
| direct_s1 | 0.9569 | 0.0046 | 0.0156 | 0.023 | 49.72 | 0.9954 | 74.844 |
| jepa_s2 | 0.9568 | 0.0043 | 0.0156 | 0.0233 | 49.86 | 0.9957 | 75.7439 |
| jepa_s1 | 0.9568 | 0.0043 | 0.0156 | 0.0233 | 49.9 | 0.9957 | 75.6709 |
| jepa_s0 | 0.9568 | 0.0044 | 0.0156 | 0.0232 | 50.07 | 0.9956 | 75.6007 |
| planner_moving_average | 0.9545 | 0.0053 | 0.0159 | 0.0243 | 50.85 | 0.9947 | 80.2518 |
| seasonal_ets | 0.9544 | 0.0043 | 0.0163 | 0.0251 | 52.25 | 0.9957 | 79.6994 |
| param_eoq_joint | 0.9544 | 0.0049 | 0.0168 | 0.0239 | 53.65 | 0.9951 | 78.7464 |
| ctx_eoq_joint | 0.9543 | 0.0051 | 0.017 | 0.0236 | 54.14 | 0.9949 | 78.537 |
| ctx2_eoq_joint | 0.9543 | 0.0054 | 0.0162 | 0.0241 | 51.54 | 0.9946 | 80.3771 |
| tuned_eoq_joint | 0.9543 | 0.0053 | 0.0168 | 0.0237 | 53.54 | 0.9947 | 77.5758 |
| tuned_eoq | 0.9348 | 0.0048 | 0.0303 | 0.0302 | 95.71 | 0.9952 | 98.6459 |
| tuned_periodic_order_up_to | 0.9288 | 0.0181 | 0.0243 | 0.0289 | 72.94 | 0.9819 | 69.6801 |
| tuned_periodic_reorder | 0.9276 | 0.0126 | 0.0303 | 0.0295 | 90.96 | 0.9874 | 68.9793 |
| tuned_reorder_point | 0.9245 | 0.0048 | 0.0293 | 0.0414 | 88.63 | 0.9952 | 100.2913 |
| tuned_order_up_to | 0.9242 | 0.0054 | 0.0336 | 0.0368 | 101.4 | 0.9946 | 89.033 |

## Primary family (provisional: Holm within this report, not over the whole pre-registered family; final labels come from the pooled primary report)

| comparison | diff, pts [95% BCa] | Holm-adjusted interval | cost reduction [95%] | win rate | Wilcoxon p | Holm p | TOST p | CVaR10, pts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jepa − ctx2_eoq_joint | +0.251 [+0.199, +0.317] | [+0.199, +0.317] | +5.5% [+4.4, +6.6] | 91% | 3.3e-15 | 1.3e-13 | 1 | -0.084 |
| direct − ctx2_eoq_joint | +0.261 [+0.212, +0.326] | [+0.205, +0.337] | +5.7% [+4.7, +6.8] | 92% | 2.8e-16 | 1.2e-14 | 1 | -0.053 |

## Secondary family

| comparison | diff, pts [95% BCa] | Holm-adjusted interval | cost reduction [95%] | win rate | Wilcoxon p | Holm p | TOST p | CVaR10, pts | label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jepa − direct | -0.011 [-0.030, +0.008] | [-0.030, +0.008] | -0.2% [-0.7, +0.2] | 52% | 0.25 | 0.29 | 1.1e-14 | -0.196 | equivalent |
| jepa − seasonal_ets | +0.242 [+0.195, +0.299] | [+0.187, +0.312] | +5.3% [+4.6, +6.1] | 91% | 3.8e-16 | 1.1e-14 | 1 | -0.053 | practically superior |
| jepa − planner_moving_average | +0.234 [+0.187, +0.300] | [+0.181, +0.310] | +5.1% [+4.1, +6.4] | 90% | 6.5e-15 | 8.7e-13 | 1 | -0.083 | practically superior |
| jepa − planner_oracle | -0.426 [-0.505, -0.360] | [-0.531, -0.343] | -11.0% [-12.2, -9.8] | 1% | 4e-18 | 3.7e-19 | 1 | -1.304 | inferior |

## Training seeds (two-level bootstrap: worlds and seeds resampled)

| group − reference | mean, pts [95%] | per-seed estimates, pts |
| --- | --- | --- |
| direct − ctx2_eoq_joint | +0.261 [+0.207, +0.321] | direct_s0: +0.262, direct_s1: +0.259, direct_s2: +0.263 |
| jepa − ctx2_eoq_joint | +0.251 [+0.195, +0.312] | jepa_s0: +0.248, jepa_s1: +0.251, jepa_s2: +0.252 |

## Strata (exploratory, BH at q = 0.1)

| comparison | stratum | n | diff, pts | p (t) | BH q |
| --- | --- | --- | --- | --- | --- |
| jepa − ctx2_eoq_joint | holding_rate = 0.000684932 | 32 | 0.125 | 8.6e-07 | 3.5e-06 |
| jepa − ctx2_eoq_joint | holding_rate = 0.00273973 | 34 | 0.183 | 1.5e-07 | 1.2e-06 |
| jepa − ctx2_eoq_joint | holding_rate = 0.01 | 34 | 0.437 | 3.5e-07 | 1.7e-06 |
| jepa − ctx2_eoq_joint | cv = 0.1 | 36 | 0.302 | 1.1e-06 | 3.6e-06 |
| jepa − ctx2_eoq_joint | cv = 0.3 | 30 | 0.117 | 0.00014 | 0.00024 |
| jepa − ctx2_eoq_joint | cv = 0.6 | 34 | 0.314 | 4.9e-06 | 1.2e-05 |
| jepa − ctx2_eoq_joint | delivery_lag = 2 | 16 | 0.262 | 0.0028 | 0.0037 |
| jepa − ctx2_eoq_joint | delivery_lag = 3 | 16 | 0.212 | 0.0088 | 0.0096 |
| jepa − ctx2_eoq_joint | delivery_lag = 4 | 15 | 0.251 | 0.0033 | 0.0041 |
| jepa − ctx2_eoq_joint | delivery_lag = 5 | 13 | 0.237 | 0.034 | 0.034 |
| jepa − ctx2_eoq_joint | delivery_lag = 6 | 25 | 0.3 | 9e-05 | 0.00018 |
| jepa − ctx2_eoq_joint | delivery_lag = 7 | 15 | 0.209 | 0.0044 | 0.0053 |
| direct − ctx2_eoq_joint | holding_rate = 0.000684932 | 32 | 0.125 | 1.2e-06 | 3.6e-06 |
| direct − ctx2_eoq_joint | holding_rate = 0.00273973 | 34 | 0.188 | 2.3e-10 | 5.5e-09 |
| direct − ctx2_eoq_joint | holding_rate = 0.01 | 34 | 0.463 | 5.1e-08 | 6.1e-07 |
| direct − ctx2_eoq_joint | cv = 0.1 | 36 | 0.29 | 2.4e-06 | 6.4e-06 |
| direct − ctx2_eoq_joint | cv = 0.3 | 30 | 0.131 | 3.2e-05 | 7e-05 |
| direct − ctx2_eoq_joint | cv = 0.6 | 34 | 0.346 | 2.6e-07 | 1.6e-06 |
| direct − ctx2_eoq_joint | delivery_lag = 2 | 16 | 0.278 | 0.002 | 0.0028 |
| direct − ctx2_eoq_joint | delivery_lag = 3 | 16 | 0.229 | 0.0081 | 0.0092 |
| direct − ctx2_eoq_joint | delivery_lag = 4 | 15 | 0.263 | 0.0012 | 0.0019 |
| direct − ctx2_eoq_joint | delivery_lag = 5 | 13 | 0.299 | 0.0093 | 0.0097 |
| direct − ctx2_eoq_joint | delivery_lag = 6 | 25 | 0.276 | 0.00013 | 0.00024 |
| direct − ctx2_eoq_joint | delivery_lag = 7 | 15 | 0.217 | 0.0017 | 0.0025 |
