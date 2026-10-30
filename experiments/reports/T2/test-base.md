# T2 sealed test (base), 1000 worlds

- split `test` (sealed test), 1000 worlds, seeds 9000000–9000999, variant `base`
- protocol sha256 `5b1785bc0d979360…`; SESOI 0.1 pts; BCa 10000 resamples; Holm at α = 0.05
- reference: `ctx2_eoq_joint` (best selection score on common tune worlds among tuned textbook arms)
- families: pre-registered (`comparisons` in the protocol)
- margin capture is the primary metric; cost reduction is `1 − (1 − s_arm)/(1 − s_ref)`, a ratio of means

## Arms

| arm | score | lost_sales_share | fee_share | holding_share | order_days | fill_rate_demand | mean_on_hand |
| --- | --- | --- | --- | --- | --- | --- | --- |
| planner_oracle | 0.9623 | 0.0029 | 0.0145 | 0.0202 | 47.59 | 0.9971 | 72.4761 |
| direct_s2 | 0.9585 | 0.0044 | 0.0152 | 0.0219 | 49.649 | 0.9956 | 78.0924 |
| direct_s1 | 0.9585 | 0.0044 | 0.0151 | 0.0219 | 49.572 | 0.9956 | 78.0659 |
| jepa_s0 | 0.9585 | 0.0041 | 0.0152 | 0.0222 | 49.961 | 0.9959 | 78.9377 |
| direct_s0 | 0.9585 | 0.0044 | 0.0151 | 0.022 | 49.487 | 0.9956 | 78.3058 |
| jepa_s0_plain | 0.9584 | 0.0047 | 0.0151 | 0.0218 | 49.288 | 0.9953 | 77.5246 |
| jepa_s1 | 0.9584 | 0.0041 | 0.0152 | 0.0222 | 49.954 | 0.9959 | 79.0346 |
| jepa_s2 | 0.9584 | 0.0042 | 0.0152 | 0.0222 | 49.994 | 0.9958 | 79.0466 |
| seasonal_ets | 0.9563 | 0.0042 | 0.016 | 0.0235 | 52.565 | 0.9958 | 82.9007 |
| ctx2_eoq_joint | 0.9561 | 0.005 | 0.016 | 0.0229 | 52.258 | 0.995 | 83.3244 |
| ctx_eoq_joint | 0.956 | 0.005 | 0.0166 | 0.0225 | 54.19 | 0.995 | 81.3825 |
| param_eoq_joint | 0.9559 | 0.005 | 0.0164 | 0.0227 | 53.699 | 0.995 | 81.5394 |
| planner_moving_average | 0.9559 | 0.0054 | 0.0157 | 0.023 | 51.495 | 0.9946 | 83.1321 |
| tuned_eoq_joint | 0.9558 | 0.0054 | 0.0163 | 0.0225 | 53.5 | 0.9946 | 80.1576 |
| tuned_eoq | 0.9366 | 0.0047 | 0.03 | 0.0288 | 97.607 | 0.9953 | 101.3299 |
| tuned_periodic_order_up_to | 0.9317 | 0.0177 | 0.0231 | 0.0275 | 72.962 | 0.9823 | 72.0997 |
| tuned_periodic_reorder | 0.9305 | 0.0136 | 0.0288 | 0.0271 | 90.945 | 0.9864 | 70.8514 |
| tuned_order_up_to | 0.9273 | 0.0056 | 0.0321 | 0.035 | 101.701 | 0.9944 | 91.7138 |
| tuned_reorder_point | 0.9272 | 0.0053 | 0.0281 | 0.0394 | 89.451 | 0.9947 | 102.8854 |

## Primary family (provisional: Holm within this report, not over the whole pre-registered family; final labels come from the pooled primary report)

| comparison | diff, pts [95% BCa] | Holm-adjusted interval | cost reduction [95%] | win rate | Wilcoxon p | Holm p | TOST p | CVaR10, pts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jepa − ctx2_eoq_joint | +0.233 [+0.218, +0.249] | [+0.218, +0.249] | +5.3% [+5.0, +5.6] | 90% | 7e-140 | 1.1e-139 | 1 | -0.064 |
| direct − ctx2_eoq_joint | +0.241 [+0.227, +0.257] | [+0.225, +0.259] | +5.5% [+5.2, +5.8] | 91% | 1.1e-145 | 4.9e-148 | 1 | -0.053 |

## Secondary family

| comparison | diff, pts [95% BCa] | Holm-adjusted interval | cost reduction [95%] | win rate | Wilcoxon p | Holm p | TOST p | CVaR10, pts | label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jepa − direct | -0.008 [-0.013, -0.002] | [-0.013, -0.002] | -0.2% [-0.3, -0.1] | 48% | 0.0057 | 0.0061 | 2.8e-157 | -0.180 | inferior |
| jepa − seasonal_ets | +0.209 [+0.195, +0.224] | [+0.193, +0.226] | +4.8% [+4.5, +5.0] | 90% | 3e-141 | 6e-133 | 1 | -0.053 | practically superior |
| jepa − planner_moving_average | +0.253 [+0.238, +0.270] | [+0.234, +0.274] | +5.7% [+5.4, +6.1] | 89% | 2.2e-141 | 5e-143 | 1 | -0.062 | practically superior |
| jepa − planner_oracle | -0.391 [-0.410, -0.372] | [-0.416, -0.367] | -10.4% [-10.8, -10.0] | 0% | 4.4e-165 | 3.6e-208 | 1 | -1.072 | inferior |

## Training seeds (two-level bootstrap: worlds and seeds resampled)

| group − reference | mean, pts [95%] | per-seed estimates, pts |
| --- | --- | --- |
| direct − ctx2_eoq_joint | +0.241 [+0.225, +0.257] | direct_s0: +0.237, direct_s1: +0.243, direct_s2: +0.244 |
| jepa − ctx2_eoq_joint | +0.233 [+0.217, +0.250] | jepa_s0: +0.238, jepa_s1: +0.234, jepa_s2: +0.228 |

## Strata (exploratory, BH at q = 0.1)

| comparison | stratum | n | diff, pts | p (t) | BH q |
| --- | --- | --- | --- | --- | --- |
| jepa − ctx2_eoq_joint | holding_rate = 0.000684932 | 329 | 0.084 | 8.7e-35 | 1.7e-34 |
| jepa − ctx2_eoq_joint | holding_rate = 0.00273973 | 343 | 0.183 | 2.1e-67 | 1.3e-66 |
| jepa − ctx2_eoq_joint | holding_rate = 0.01 | 328 | 0.437 | 6.6e-86 | 1e-84 |
| jepa − ctx2_eoq_joint | cv = 0.1 | 331 | 0.259 | 3.3e-59 | 1.6e-58 |
| jepa − ctx2_eoq_joint | cv = 0.3 | 326 | 0.188 | 6e-42 | 1.4e-41 |
| jepa − ctx2_eoq_joint | cv = 0.6 | 343 | 0.252 | 2.2e-45 | 5.9e-45 |
| jepa − ctx2_eoq_joint | delivery_lag = 2 | 148 | 0.191 | 3.5e-21 | 3.8e-21 |
| jepa − ctx2_eoq_joint | delivery_lag = 3 | 182 | 0.245 | 3.1e-27 | 4.2e-27 |
| jepa − ctx2_eoq_joint | delivery_lag = 4 | 174 | 0.271 | 9e-29 | 1.5e-28 |
| jepa − ctx2_eoq_joint | delivery_lag = 5 | 169 | 0.228 | 5.4e-25 | 6.4e-25 |
| jepa − ctx2_eoq_joint | delivery_lag = 6 | 159 | 0.226 | 3.9e-21 | 4.1e-21 |
| jepa − ctx2_eoq_joint | delivery_lag = 7 | 168 | 0.232 | 2.2e-25 | 2.7e-25 |
| direct − ctx2_eoq_joint | holding_rate = 0.000684932 | 329 | 0.095 | 2.5e-40 | 5.4e-40 |
| direct − ctx2_eoq_joint | holding_rate = 0.00273973 | 343 | 0.193 | 5.4e-74 | 4.3e-73 |
| direct − ctx2_eoq_joint | holding_rate = 0.01 | 328 | 0.439 | 8.5e-86 | 1e-84 |
| direct − ctx2_eoq_joint | cv = 0.1 | 331 | 0.26 | 2.7e-57 | 1.1e-56 |
| direct − ctx2_eoq_joint | cv = 0.3 | 326 | 0.191 | 7.3e-47 | 2.2e-46 |
| direct − ctx2_eoq_joint | cv = 0.6 | 343 | 0.271 | 3.3e-51 | 1.1e-50 |
| direct − ctx2_eoq_joint | delivery_lag = 2 | 148 | 0.184 | 5.8e-21 | 5.8e-21 |
| direct − ctx2_eoq_joint | delivery_lag = 3 | 182 | 0.25 | 2.5e-27 | 3.5e-27 |
| direct − ctx2_eoq_joint | delivery_lag = 4 | 174 | 0.28 | 5.7e-30 | 1.1e-29 |
| direct − ctx2_eoq_joint | delivery_lag = 5 | 169 | 0.241 | 2.6e-28 | 4.1e-28 |
| direct − ctx2_eoq_joint | delivery_lag = 6 | 159 | 0.241 | 1.7e-23 | 2e-23 |
| direct − ctx2_eoq_joint | delivery_lag = 7 | 168 | 0.243 | 7.8e-28 | 1.2e-27 |
