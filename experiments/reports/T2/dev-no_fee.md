# T2 dev (no_fee), 100 worlds

- split `dev` (not held-out), 100 worlds, seeds 4000000–4000099, variant `no_fee`
- protocol sha256 `5b1785bc0d979360…`; SESOI 0.1 pts; BCa 10000 resamples; Holm at α = 0.05
- reference: `ctx2_eoq_joint` (best selection score on common tune worlds among tuned textbook arms)
- families: pre-registered (`comparisons` in the protocol)
- margin capture is the primary metric; cost reduction is `1 − (1 − s_arm)/(1 − s_ref)`, a ratio of means

## Arms

| arm | score | lost_sales_share | fee_share | holding_share | order_days | fill_rate_demand | mean_on_hand |
| --- | --- | --- | --- | --- | --- | --- | --- |
| planner_oracle | 0.9891 | 0.0024 | 0.0 | 0.0085 | 360.48 | 0.9976 | 21.1583 |
| jepa_s0 | 0.9843 | 0.005 | 0.0 | 0.0107 | 359.57 | 0.995 | 31.6364 |
| jepa_s1 | 0.9842 | 0.005 | 0.0 | 0.0108 | 359.64 | 0.995 | 31.7783 |
| jepa_s2 | 0.9841 | 0.0051 | 0.0 | 0.0107 | 359.51 | 0.9949 | 31.7436 |
| direct_s0 | 0.9841 | 0.0044 | 0.0 | 0.0115 | 357.3 | 0.9956 | 32.9797 |
| direct_s1 | 0.9841 | 0.0045 | 0.0 | 0.0114 | 357.13 | 0.9955 | 33.0043 |
| direct_s2 | 0.9841 | 0.0045 | 0.0 | 0.0115 | 357.52 | 0.9955 | 32.9714 |
| ctx2_eoq_joint | 0.9827 | 0.005 | 0.0 | 0.0123 | 364.94 | 0.995 | 34.6494 |
| ctx_eoq_joint | 0.9825 | 0.0047 | 0.0 | 0.0127 | 364.88 | 0.9953 | 35.5896 |
| param_eoq_joint | 0.9824 | 0.0044 | 0.0 | 0.0132 | 364.9 | 0.9956 | 34.9927 |
| seasonal_ets | 0.9823 | 0.0049 | 0.0 | 0.0128 | 359.59 | 0.9951 | 36.6091 |
| jepa_s0_plain | 0.9821 | 0.0088 | 0.0 | 0.0091 | 346.5 | 0.9912 | 23.2732 |
| planner_moving_average | 0.9816 | 0.0047 | 0.0 | 0.0137 | 360.53 | 0.9953 | 38.6704 |
| tuned_eoq | 0.9808 | 0.0041 | 0.0 | 0.0151 | 364.9 | 0.9959 | 34.4234 |
| tuned_eoq_joint | 0.9808 | 0.0041 | 0.0 | 0.0151 | 364.9 | 0.9959 | 34.4234 |
| tuned_periodic_order_up_to | 0.9774 | 0.0068 | 0.0 | 0.0158 | 364.5 | 0.9932 | 36.9243 |
| tuned_periodic_reorder | 0.9773 | 0.007 | 0.0 | 0.0157 | 364.83 | 0.993 | 36.4556 |
| tuned_reorder_point | 0.977 | 0.0052 | 0.0 | 0.0178 | 339.27 | 0.9948 | 41.8499 |
| tuned_order_up_to | 0.9762 | 0.0051 | 0.0 | 0.0187 | 364.69 | 0.9949 | 43.8457 |

## Primary family (provisional: Holm within this report, not over the whole pre-registered family; final labels come from the pooled primary report)

| comparison | diff, pts [95% BCa] | Holm-adjusted interval | cost reduction [95%] | win rate | Wilcoxon p | Holm p | TOST p | CVaR10, pts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jepa − ctx2_eoq_joint | +0.154 [+0.115, +0.207] | [+0.110, +0.217] | +8.9% [+6.7, +11.7] | 77% | 1.3e-10 | 3.4e-09 | 0.99 | -0.094 |
| direct − ctx2_eoq_joint | +0.141 [+0.099, +0.198] | [+0.099, +0.198] | +8.2% [+5.6, +11.2] | 76% | 4.2e-09 | 1.3e-07 | 0.95 | -0.176 |

## Secondary family

| comparison | diff, pts [95% BCa] | Holm-adjusted interval | cost reduction [95%] | win rate | Wilcoxon p | Holm p | TOST p | CVaR10, pts | label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jepa − direct | +0.013 [-0.001, +0.030] | [-0.001, +0.030] | +0.8% [-0.1, +1.8] | 62% | 0.29 | 0.12 | 6.8e-19 | -0.100 | equivalent |
| jepa − seasonal_ets | +0.194 [+0.151, +0.251] | [+0.145, +0.260] | +10.9% [+9.1, +13.1] | 87% | 4.7e-15 | 2.2e-11 | 1 | -0.044 | practically superior |
| jepa − planner_moving_average | +0.264 [+0.209, +0.337] | [+0.198, +0.356] | +14.3% [+11.8, +17.7] | 92% | 7.2e-16 | 1.9e-12 | 1 | -0.042 | practically superior |
| jepa − planner_oracle | -0.490 [-0.562, -0.426] | [-0.584, -0.409] | -45.0% [-53.7, -38.0] | 0% | 3.9e-18 | 3.6e-25 | 1 | -1.227 | inferior |

## Training seeds (two-level bootstrap: worlds and seeds resampled)

| group − reference | mean, pts [95%] | per-seed estimates, pts |
| --- | --- | --- |
| direct − ctx2_eoq_joint | +0.141 [+0.095, +0.193] | direct_s0: +0.142, direct_s1: +0.142, direct_s2: +0.140 |
| jepa − ctx2_eoq_joint | +0.154 [+0.111, +0.203] | jepa_s0: +0.158, jepa_s1: +0.157, jepa_s2: +0.147 |

## Strata (exploratory, BH at q = 0.1)

| comparison | stratum | n | diff, pts | p (t) | BH q |
| --- | --- | --- | --- | --- | --- |
| jepa − ctx2_eoq_joint | holding_rate = 0.000684932 | 32 | 0.019 | 0.22 | 0.22 |
| jepa − ctx2_eoq_joint | holding_rate = 0.00273973 | 34 | 0.144 | 8.9e-10 | 1.1e-08 |
| jepa − ctx2_eoq_joint | holding_rate = 0.01 | 34 | 0.291 | 1.1e-05 | 8.9e-05 |
| jepa − ctx2_eoq_joint | cv = 0.1 | 36 | 0.232 | 2.2e-05 | 0.00013 |
| jepa − ctx2_eoq_joint | cv = 0.3 | 30 | 0.067 | 0.013 | 0.02 |
| jepa − ctx2_eoq_joint | cv = 0.6 | 34 | 0.149 | 0.00028 | 0.00096 |
| jepa − ctx2_eoq_joint | delivery_lag = 2 | 16 | 0.167 | 0.0093 | 0.015 |
| jepa − ctx2_eoq_joint | delivery_lag = 3 | 16 | 0.161 | 0.0021 | 0.0055 |
| jepa − ctx2_eoq_joint | delivery_lag = 4 | 15 | 0.18 | 0.061 | 0.073 |
| jepa − ctx2_eoq_joint | delivery_lag = 5 | 13 | 0.182 | 0.005 | 0.011 |
| jepa − ctx2_eoq_joint | delivery_lag = 6 | 25 | 0.15 | 0.0086 | 0.015 |
| jepa − ctx2_eoq_joint | delivery_lag = 7 | 15 | 0.089 | 0.037 | 0.05 |
| direct − ctx2_eoq_joint | holding_rate = 0.000684932 | 32 | 0.015 | 0.33 | 0.33 |
| direct − ctx2_eoq_joint | holding_rate = 0.00273973 | 34 | 0.141 | 6.1e-10 | 1.1e-08 |
| direct − ctx2_eoq_joint | holding_rate = 0.01 | 34 | 0.26 | 0.00028 | 0.00096 |
| direct − ctx2_eoq_joint | cv = 0.1 | 36 | 0.223 | 5e-05 | 0.00024 |
| direct − ctx2_eoq_joint | cv = 0.3 | 30 | 0.086 | 0.0034 | 0.0081 |
| direct − ctx2_eoq_joint | cv = 0.6 | 34 | 0.104 | 0.024 | 0.034 |
| direct − ctx2_eoq_joint | delivery_lag = 2 | 16 | 0.181 | 0.0058 | 0.012 |
| direct − ctx2_eoq_joint | delivery_lag = 3 | 16 | 0.155 | 0.0019 | 0.0055 |
| direct − ctx2_eoq_joint | delivery_lag = 4 | 15 | 0.195 | 0.052 | 0.065 |
| direct − ctx2_eoq_joint | delivery_lag = 5 | 13 | 0.186 | 0.0062 | 0.012 |
| direct − ctx2_eoq_joint | delivery_lag = 6 | 25 | 0.105 | 0.097 | 0.11 |
| direct − ctx2_eoq_joint | delivery_lag = 7 | 15 | 0.053 | 0.13 | 0.14 |
