"""Core discrete-event simulator package.

This sub-package contains every module needed to define and run one
retail market simulation:

- ``scenario``         — flat ``Scenario`` dataclass + authoring helpers
                         (``load_catalog``, ``make_stores``,
                         ``load_scenario_from_path``).
- ``runner``           — ``Runner`` that owns the per-tick observe →
                         decide → advance → log loop.
- ``store``            — ``Store`` with full per-product accounting.
- ``store_initializer``— pure ``init_store_state`` seam pinning the
                         bit-identity contract for step 0.
- ``policy``           — ``Policy`` ABC + ``HeuristicPolicy`` + textbook family.
- ``market``           — regional demand/supply environment.
- ``event_engine``     — stochastic disruption events + scheduled
                         delivery callbacks.
- ``item_registry``    — catalog + per-item lifecycle state.
- ``lifecycle_clock``  — pure ``advance_stage`` over the canonical
                         five-stage product lifecycle.
- ``freshness_curve``  — pure per-(store, product) hype multiplier
                         ``m(τ) = 1 + α · exp(−τ/β)``.
- ``distributions``    — ``Constant`` / ``Uniform`` / ``Normal`` /
                         ``Choice`` lazy sample objects.
- ``data_exporter``    — parquet + JSON + PNG writer for a finished
                         run.

The package intentionally has no top-level ``__init__`` exports so
that import side-effects stay scoped to the module you actually use.
"""
