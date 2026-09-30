# 16: Unseen-SKU and unseen-disruption generalisation protocols

**What to build:** Transfer claims that test generalisation rather than seed
variation.

Today the only held-out axis is the world seed: the log uses seeds from
14,000,000 and evaluation uses 13,000,000. The catalog, topology, world
parameters and regime are identical, and 99 of the 100 synthetic SKUs appear in
training. That is a seed holdout, not a generalisation test, and it cannot
support the claim that a per-SKU model generalises rather than memorises.

A harsher held-out disruption was already listed as future work in the first
study's paper. It becomes a protocol here.

**Blocked by:** 15.

**Status:** done

**Prior art:** existing synthetic comparison tests; the `Synthetic comparison`,
`Active subset` and `EventEngine` entries in `CONTEXT.md`; ADR 0026 (replay
window adaptation) for how an existing transfer protocol is framed.

- [x] The catalog is partitioned so that training SKUs and evaluation SKUs never
      overlap, asserted by a test.
- [x] A harsher disruption regime is held entirely out of the training episodes
      and evaluated as its own protocol.
- [x] In-distribution, unseen-SKU and unseen-disruption results are reported as
      three separate tables rather than pooled, so each claim stands alone.
- [x] Each table carries the same floors, ceiling and operational columns as
      ticket 15 specifies.
- [x] Seed ranges for every protocol are disjoint and recorded in the manifest.
- [x] Any notebook, README section or `CONTEXT.md` text invalidated by the new
      protocols is updated in this ticket.
