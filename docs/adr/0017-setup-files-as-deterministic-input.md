# Setup files as the deterministic simulation input

Status: Accepted

The runnable unit was a `scenarios/*.py` file that imperatively did three jobs at once:
fetch/generate world data (LLM or synthetic), synthesise a graph topology, and attach
policy *objects* to nodes. This conflated data preparation with experiment wiring, left
no human-editable artifact (a scenario could only be expressed as Python), and made the
`Scenario` JSON round-trip lossy — policies survived only as a bare class-name string, so
a serialised scenario could not reconstruct a runnable experiment.

**Decision — split into two stages with a file-based boundary.** A *simulation setup* is a
directory of human-readable, human-editable data files that fully determine a run:

- `catalog.csv` — products (one row per SKU). LLM-generatable; the only file the world
  generator writes besides the market block. Scalar columns plus `related_products` encoded
  as a string (`pid:weight;…`).
- `setup.yaml` — everything else, human-owned: `market`, `disruption`, `nodes`
  (`id`, `type`, `region`, inventory, capacity, prices, `policy: {name, params}`), `edges`
  (`supplier`, `buyer`, `lead_time`), and `run` (`n_steps`, `start_date`, `world_seed`).

Running becomes `main.py run <setup-dir>` with **no Python scenario file**. Policy *code*
lives in `src/sim/policy.py` and is selected by name through a `{name: class}` registry;
`params` are splatted as constructor kwargs.

**Determinism contract.** A single `world_seed` lives in `setup.yaml`. Each node's init seed
and policy seed are *derived* from `(world_seed, node_id)` by the loader. Same files ⇒
bit-identical output. An A/B policy comparison on an identical world is "copy the setup,
change only the `policy` block."

**Node/edge facts are written once and injected into policies.** Capacity, unit cost, list
prices, supplier ids and lead times live on the node/edge only; the loader injects them into
the policy at construction, so `params` carries *only* genuine hyperparameters. This removes
a hand-editing footgun (editing capacity on the node but not in the policy would silently
diverge).

## Considered options

- **Setup dir + thin Python entrypoint** (data in files, policy wiring still in Python).
  Rejected: "which policy" stays trapped in code, defeating the hand-editable goal.
- **Two artifacts: keep `World` + a separate topology file.** Rejected: keeps two formats
  and the `World` abstraction we're retiring.
- **Single all-in-one YAML** (catalog inline). Rejected: catalogs reach ~3000 rows; inline
  YAML is miserable to hand-edit. CSV opens in a spreadsheet and diffs cleanly.
- **Explicit seeds on every node/policy.** Rejected: more surface, more footguns; derivation
  from `(world_seed, node_id)` gives the same control with one knob.

## Consequences

- **`World` removed entirely** (`src/sim/world.py`, `world.json` cache, `src/sim/world_loader.py`).
  The setup directory *is* the cache and the artifact. The LLM generator writes setup files
  directly; `load_or_build_world` becomes "load the setup dir if present, else generate it."
- **`world_to_graph` becomes a deterministic scaffolder** that emits a starter `nodes:`/`edges:`
  block into `setup.yaml` (e.g. `main.py scaffold`), not a runtime expansion step. Small graphs
  are hand-written; large catalogs use the scaffolder, then hand-edit. Both produce identical YAML.
- **Stores fully retired** — completing the migration ADR 0011 documented but did not finish.
  `StoreTemplate`, `StoreInstance`, `make_stores`, `stores_df`, `Scenario.stores`,
  `Scenario.is_graph`, `Scenario.from_world` deleted. Output `stores.parquet` → `nodes.parquet`.
- **RL/tuning** load `catalog` + `market` from a setup dir (replacing `world_loader`'s
  cache/archetype/synthetic resolution) and keep a programmatic per-episode graph builder in
  `NodeInstance`/`EdgeSpec` terms. The old `StoreTemplate` randomisation fields
  (`capacity_dist`, `balance_dist`, `delivery_lag`, `holding_rate`, `order_fee`) move into
  `RLConfig` / tuning `Config`. Rationale: RL's topology varies per episode with the sampled
  active-SKU subset, so a static file topology cannot drive it.
- **Output snapshot** is a verbatim copy of the input setup files (`config/catalog.csv`,
  `config/setup.yaml`), so a run is reproducible by construction. The lossy
  `Scenario.to_json`/`from_json` path is removed.
- **Freshness curve and product life-cycle are removed** for the PoC (see `TODO.md`);
  ADR 0001 and ADR 0002 become superseded by this ADR. Market otherwise stays intact.
- A single `Runner` / `build_world` remains; the `GraphRunner` / `build_graph_world` aliases
  and dead policy classes (`Policy` alias, `HeuristicPolicy`, `_HeuristicPolicyOld`, `NoopPolicy`)
  are deleted.
- **Out of scope (deferred):** collapsing the textbook-policy two-hierarchy bridge
  (`_OrderUpToCore(TextbookReorderPolicy(Policy))` wrapped by
  `OrderUpToPolicy(MultiSupplierTextbookPolicy(IntermediatePolicy))`) into a single hierarchy.
