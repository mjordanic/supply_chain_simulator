# T1 exploratory forecast re-score (300 dev shops) — NOT a gate

**EXPLORATORY, after G2 failed: a post-failure look that cannot feed any T1 primary or secondary claim, does not re-open G2, and leads to no closed-loop or sealed-test run.** One training seed per variant, fitted on train only; scored once.

Mean pinball over levels 0.1, 0.5, 0.9, 0.95, 0.99, scaled; lower is better.

| forecaster | daily | lead_time | shop_lead_time |
| --- | --- | --- | --- |
| jepa-s0 | 0.48743 | 0.44462 | 0.41576 |
| direct-s0 | 0.47090 | 0.42651 | 0.39651 |
| jepa-ctx56-s0 | 0.48948 | 0.44528 | 0.41721 |
| jepa-ctx112-s0 | 0.49931 | 0.45511 | 0.42755 |
| jepa-ctx112-lw0.3-s0 | 0.49946 | 0.45275 | 0.42429 |
| jepa-ctx112-lw0.1-s0 | 0.48630 | 0.44620 | 0.41745 |
| jepa-ctx112-lw0.1-k6-s0 | 0.51971 | 0.46202 | 0.42975 |
| direct-ctx56-s0 | 0.47043 | 0.42135 | 0.39121 |
| direct-ctx112-s0 | 0.47059 | 0.42628 | 0.39596 |
| direct-ctx112-k6-s0 | 0.49134 | 0.43166 | 0.39838 |
| hmm | 0.35885 | 0.29858 | 0.24539 |
| bayes | 0.22073 | 0.14563 | 0.13949 |

| JEPA − direct (lead_time) | mean | 95% BCa | JEPA better in |
| --- | --- | --- | --- |
| jepa-s0 − direct-s0 | +0.01811 | [+0.01133, +0.02950] | 27% |
| jepa-ctx56-s0 − direct-ctx56-s0 | +0.02392 | [+0.01412, +0.04057] | 34% |
| jepa-ctx112-s0 − direct-ctx112-s0 | +0.02883 | [+0.01864, +0.04558] | 27% |
| jepa-ctx112-lw0.3-s0 − direct-ctx112-s0 | +0.02648 | [+0.01664, +0.04315] | 26% |
| jepa-ctx112-lw0.1-s0 − direct-ctx112-s0 | +0.01993 | [+0.01183, +0.03307] | 33% |
| jepa-ctx112-lw0.1-k6-s0 − direct-ctx112-s0 | +0.03574 | [+0.01627, +0.06806] | 62% |
| jepa-ctx112-lw0.1-k6-s0 − direct-ctx112-k6-s0 | +0.03035 | [+0.01696, +0.05236] | 35% |
