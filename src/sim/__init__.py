"""Core discrete-event simulator package.

This sub-package contains every module needed to define and run one
retail market simulation on the multi-echelon graph engine.

Phase-4 (issue 11): the legacy single-Store engine has been retired.
All simulations now run on the graph engine.

Key modules:
- ``scenario``        — ``Scenario`` dataclass + authoring helpers
                        (``load_catalog``, ``make_nodes``,
                        ``load_scenario_from_path``).
- ``runner``          — ``Runner`` (graph engine) + ``Simulation`` +
                        ``build_world`` + ``TickResult``.
- ``node``            — ``FactoryNode``, ``IntermediateNode``,
                        ``DemandSinkNode`` ABCs.
- ``graph``           — ``Graph``, ``EdgeSpec``, ``build_graph``.
- ``policy``          — ``NodePolicy`` ABC + concrete policies:
                        ``StaticFactoryPolicy``, ``DefaultDemandSinkPolicy``,
                        ``MultiSupplierTextbookPolicy``,
                        ``OrderUpToPolicy``, etc.
- ``central_table``   — ``CentralTable`` live offer table.
- ``allocation``      — ``execute_buy``, ``shuffle_buyers``.
- ``market``          — regional demand/supply environment.
- ``event_engine``    — stochastic disruption events + delivery callbacks.
- ``item_registry``   — catalog + per-item lifecycle state.
- ``lifecycle_clock`` — pure ``advance_stage`` over the five-stage PLC.
- ``freshness_curve`` — per-(node, product) hype multiplier
                        ``m(τ) = 1 + α · exp(−τ/β)``.
- ``distributions``   — ``Constant`` / ``Uniform`` / ``Normal`` /
                        ``Choice`` lazy sample objects.
- ``data_exporter``   — parquet + JSON + PNG writer for a finished run.
- ``world``           — ``World`` LLM artifact + ``world_to_graph`` helper.

The package intentionally has no top-level ``__init__`` exports so
that import side-effects stay scoped to the module you actually use.
"""
