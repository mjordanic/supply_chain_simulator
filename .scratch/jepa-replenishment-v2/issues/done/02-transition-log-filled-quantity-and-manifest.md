# 02: Log the filled quantity, the arrival-age profile, and a run manifest

**What to build:** A transition log that records the action that actually
happened, carries enough state to predict arrivals, and can be traced back to the
parameters that produced it.

Three defects today. First, the log reads the order quantity out of the buyer's
emitted action dict — the post-Ration *request*, before Allocation clamps it by
offer availability, buyer cash and buyer capacity. Measured over three holdout
seeds, `ExploratoryPolicy` under-fills on 109 of 364 ticks and the tuned anchor on
12, in a world where suppliers never stock out. The world model is trained on
transitions attributed to an action that was not taken. Second, in-transit units
are summed to a scalar, so arrival timing is not recoverable and `next_in_transit`
accuracy is capped near 0.75 for every method including ordinary least squares.
Third, the log carries no provenance at all: the parquet holds only pandas column
metadata and there is no sidecar, so the published run record was hand-transcribed
into the paper.

**Blocked by:** None (can start immediately).

**Status:** done

**Prior art:** existing transition-log tests; the `Transition log`, `Allocation`
and `Ration` entries in `CONTEXT.md`; ADR 0012 (central table FCFS allocation) for
the clamp order that `execute_buy` applies.

**Artifact homes:** the manifest is a run artifact and lives beside the log under
`runs/` (gitignored) — it is an input to the regenerated paper, not a committed
file itself. Anything that must be committed goes in `docs/`.

- [x] The recorded action is the quantity Allocation filled, not the quantity the
      buyer requested.
- [x] The requested quantity is retained as its own column, so the size of the
      clamp is measurable.
- [x] A test covers ticks where the request was clamped by offer availability, by
      buyer cash, and by buyer capacity, and asserts the two columns differ there.
- [x] In-transit units are recorded as an arrival-age profile rather than a single
      summed scalar.
- [x] A manifest sidecar records every parameter that produced the log: episode
      count and seed range, episode length, `K_active`, delivery lag, holding
      rate, order fee, catalog identity, behaviour mixture composition, and the
      git SHA.
- [x] The realised fraction of transitions carrying a non-zero order is computed
      and written to the manifest.
- [x] Any notebook, README section or `CONTEXT.md` text invalidated by the new
      columns is updated in this ticket.
