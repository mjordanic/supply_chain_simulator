# Primary family (pooled)

- protocol sha256 `5b1785bc0d979360…`; SESOI 0.1 pts; BCa 10000 resamples; Holm at α = 0.05 over all 8 comparisons
- cells (from the protocol): T2-consolidation/perfect_supplier/base, T2-consolidation/perfect_supplier/no_fee, T2-consolidation/perfect_supplier/one_product, T1-hidden-regimes/hidden_regimes/base; sealed test evals only; **partial**: not yet run, counted as p = 1 (T1-hidden-regimes/hidden_regimes/base: no test eval), so these labels are conservative and can only loosen once the whole family lands

| comparison | diff, pts [95% BCa] | Holm-adjusted interval | cost reduction [95%] | win rate | Wilcoxon p | Holm p | TOST p | CVaR10, pts | label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T2-consolidation/perfect_supplier/base: jepa − ctx2_eoq_joint | +0.233 [+0.218, +0.249] | [+0.213, +0.256] | +5.3% [+5.0, +5.6] | 90% | 7e-140 | 7.8e-139 | 1 | -0.064 | practically superior |
| T2-consolidation/perfect_supplier/base: direct − ctx2_eoq_joint | +0.241 [+0.227, +0.257] | [+0.221, +0.264] | +5.5% [+5.2, +5.8] | 91% | 1.1e-145 | 2e-147 | 1 | -0.053 | practically superior |
| T2-consolidation/perfect_supplier/no_fee: jepa − ctx2_eoq_joint | +0.186 [+0.171, +0.202] | [+0.166, +0.207] | +11.2% [+10.5, +12.1] | 81% | 4.4e-107 | 3.5e-99 | 1 | -0.097 | practically superior |
| T2-consolidation/perfect_supplier/no_fee: direct − ctx2_eoq_joint | +0.170 [+0.156, +0.186] | [+0.152, +0.190] | +10.3% [+9.5, +11.1] | 79% | 4.7e-97 | 5.3e-91 | 1 | -0.107 | practically superior |
| T2-consolidation/perfect_supplier/one_product: jepa − param_eoq_joint | +0.132 [+0.106, +0.160] | [+0.101, +0.166] | +1.7% [+1.4, +2.1] | 59% | 3.5e-16 | 1.8e-20 | 0.99 | -0.499 | practically superior |
| T2-consolidation/perfect_supplier/one_product: direct − param_eoq_joint | +0.135 [+0.110, +0.164] | [+0.103, +0.171] | +1.8% [+1.4, +2.1] | 61% | 1.2e-17 | 2e-21 | 1 | -0.502 | practically superior |
