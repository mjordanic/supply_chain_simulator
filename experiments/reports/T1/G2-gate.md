# T1 gate G2: dev forecast metrics (300 dev shops, behaviour policies, vs true demand)

Rule (fixed before the run): G2 passes when JEPA minus direct (each averaged over its training seeds) on the dev lead_time pinball has a world-paired 95% BCa interval entirely below 0, and the point estimates on daily and shop_lead_time are <= 0.

Mean pinball over levels 0.1, 0.5, 0.9, 0.95, 0.99, scaled; lower is better.

| forecaster | daily | lead_time | shop_lead_time |
| --- | --- | --- | --- |
| jepa | 0.48565 | 0.44220 | 0.41294 |
| direct | 0.46951 | 0.42698 | 0.39640 |
| jepa-s0 | 0.48743 | 0.44462 | 0.41576 |
| jepa-s1 | 0.48417 | 0.43883 | 0.40917 |
| jepa-s2 | 0.48536 | 0.44315 | 0.41390 |
| direct-s0 | 0.47090 | 0.42651 | 0.39651 |
| direct-s1 | 0.47984 | 0.43174 | 0.40035 |
| direct-s2 | 0.45780 | 0.42269 | 0.39233 |
| hmm | 0.35885 | 0.29858 | 0.24539 |
| bayes | 0.22073 | 0.14563 | 0.13949 |
| moving_average | 0.67392 | 0.62985 | 0.62166 |

| JEPA − direct | mean | 95% BCa | JEPA better in |
| --- | --- | --- | --- |
| daily | +0.01614 | [+0.00955, +0.02783] | 28% of shops |
| lead_time | +0.01522 | [+0.00958, +0.02511] | 22% of shops |
| shop_lead_time | +0.01654 | [+0.01009, +0.02724] | 30% of shops |

**G2 fails.**
