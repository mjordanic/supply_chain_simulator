# A pre-registered evaluation harness for the perfect-supplier experiments

Status: Accepted

The lean result (ADR 0033) is +0.24 pts margin capture over tuned joint EOQ in the base world, and the direct ablation matches it. Its 500-world eval block was looked at over several development rounds, the planner was never tuned while the textbook rules had 200 trials × 64 worlds each, only one training seed was run, and variants were not logged. None of that supports a publishable claim.

**Decision — Every claim from here on goes through one harness, `src/perfect_supplier/bench/`, under a protocol fixed before any tuning.**

1. **Protocol.** `experiments/protocol.yaml` holds the approved constants: margin capture primary with cost reduction co-reported, SESOI 0.10 pts, seed blocks, the burnt-seed list, budgets, search spaces, 3 training seeds. It changes only by a logged amendment, and every log line carries its sha256.
2. **Seeds.** The harness hands out seeds only from registered blocks: tune 2,000,000+, dev 4,000,000–4,000,499, and a sealed test block 9,000,000–9,000,999. Burnt ranges (the old eval block, shadow runs, audit diagnostics, the engine line's blocks) are refused.
3. **Sealed test.** A test run is refused unless every arm's config hash is listed in `experiments/frozen.json` under the current protocol. A second test run of the same arm and variant needs a logged rerun reason.
4. **Budgets.** Learned arms get 48 trials × 32 tune worlds. Textbook arms get at least 4× that, and each cell of a contextual arm gets the full textbook budget. Tuning is sequential TPE with a process pool per trial; Optuna's thread mode is GIL-bound on these episodes.
5. **Baselines.** Besides the six textbook families, contextual joint EOQ (per observable cell, or a parametric map of the critical ratio) and the planner with a classical seasonal-ETS forecast separate learning from planning.
6. **Statistics.** Paired by world. BCa intervals (10,000 resamples) replace the percentile bootstrap; Wilcoxon as a robustness check; Holm across the primary family (learned arm − best textbook arm, the reference chosen on tune scores) and within the secondary family; BH over strata, labelled exploratory; TOST against the SESOI; a two-level bootstrap over worlds and training seeds. Labels: superior, practically superior, equivalent, inferior, inconclusive.
7. **Log.** `experiments/log/<track>.jsonl`, append-only, written by the harness for every run, including failed and refused ones. The pre-harness results are backfilled as `kind: historical`.

## Consequences

The harness re-scores the stored base comparison on the dev block within the stored interval (`experiments/reports/H0-acceptance.md`). Tracks T2 (fair consolidation) and T1 (hidden demand regimes) build on it in parallel; each logs to its own track file. The headline stays "learned forecaster plus planner" unless a track shows JEPA beating the direct ablation on the sealed test.
