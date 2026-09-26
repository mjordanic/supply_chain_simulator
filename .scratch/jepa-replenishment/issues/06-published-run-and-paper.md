# 06: Published synthetic run and paper

**What to build:** One published synthetic run, then a markdown paper whose method matches the PRD and whose result tables are copied from that run. The run collects the full log, fits, runs both `OrderUpToPolicy` tuning studies, and evaluates the holdout at the published seed count and search budget. The M5 section is filled from replay artifacts when they exist, and otherwise states the command and the slice rule. No cell is invented. The PPO stack is left untouched.

**Blocked by:** 04 — Training notebook, results notebook, and README commands. 05 — M5 replay exam.

**Status:** ready-for-agent

## Parent

`.scratch/jepa-replenishment/PRD.md`. ADR 0024 through ADR 0029.

## Stories

PRD stories 34–42 at published scale, 43–53 as the paper's M5 report, 60–63, 66–68, 70. Stories 34–42's entry point is ticket 03. Stories 43–53's notebook is ticket 05.

## Prior art

Ticket 03's comparison entry point and pass rules. Ticket 04's README commands. Ticket 05's slice rule and skip path. The tuner's published defaults: 150 trials, 16 search seeds, 32 holdout seeds. Business metrics `net_profit / initial_cash`.

## Artifact homes

- Raw run: `runs/jepa-replenishment/` (gitignored), including `log/`, `checkpoints/`, `synthetic/`, and `m5/` when M5 was run.
- Paper: `docs/jepa-replenishment.md` (tracked). Summary tables in the paper are copied from the run. They are the only tracked numbers.

## Acceptance criteria

- [ ] The published log is 64 episodes of each behavior at seed offset `14_000_000`, on 365-tick, five-SKU, lead-time-3 episodes.
- [ ] Both `OrderUpToPolicy` studies run at 150 trials, one at `holding_rate = 0.01` and one at `0.05`. The other three textbook policies stay at published defaults.
- [ ] The synthetic eval uses 32 holdout seeds at offset `13_000_000` and a search budget of 64. Weights for the retarget are the `0.01` fit, not a refit.
- [ ] `docs/jepa-replenishment.md` states the method, the censored-sales and `unit_cost` caveats, the success rules, and the future work (pricing policies, a joint multi-SKU model, policy control of engine Allocation, a harsher held-out disruption). The in-distribution claim is not "beats the tuned anchor."
- [ ] Every number in the paper is copied from `runs/jepa-replenishment/`. The in-distribution and retarget pass booleans are the ones ticket 03 computes. A failing boolean is reported as a fail. Thresholds are not edited to produce a pass.
- [ ] If `runs/jepa-replenishment/m5/` is absent, the M5 section contains the command and the ADR 0028 slice rule and no results. If it is present, the zero-shot row and the adaptation row are both copied, kept separate.
- [ ] The RL package is unchanged.
