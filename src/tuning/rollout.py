"""Single-episode rollout primitives for tuning studies.

Self-contained — no imports from ``src.rl``. Delegates the full simulation
run to ``src.sim.runner.Runner`` and builds metrics via the shared
DataFrame-native path (ADR 0019).

Graph-mode (issue 06)
----------------------
``run_policy_episode`` now receives a ``TuningEpisodeSpec`` whose
``scenario`` is already a graph-mode Scenario (built by ``episode.py``):

    FactoryNode(F_<pid>) → IntermediateNode("S") → DemandSinkNode(D_<pid>)

The policy (a ``MultiSupplierTextbookPolicy`` or any ``IntermediatePolicy``)
is attached to the ``IntermediateNode`` by node id "S". No store-mode
conversion is needed.

``holding_rate`` and ``order_fee`` are read from the ``IntermediateNode``
"S" in the scenario (they are carried there by ``episode.py`` from the
``TuningConfig`` — ADR 0019 Rule 1–2).
"""

from __future__ import annotations

from src.tuning.episode import TuningEpisodeSpec
from src.sim.policy import Policy


# ---------------------------------------------------------------------------
# run_policy_episode — main public entry point
# ---------------------------------------------------------------------------


def run_policy_episode(policy: Policy, spec: TuningEpisodeSpec) -> dict[str, float]:
    """Roll out ``policy`` on the graph-mode scenario in ``spec``.

    ``spec.scenario`` is a graph-mode Scenario built by ``episode.sample_episode``:

        FactoryNode(F_<pid>) → IntermediateNode("S") → DemandSinkNode(D_<pid>)

    The ``policy`` (an ``IntermediatePolicy`` / ``MultiSupplierTextbookPolicy``)
    is attached to the IntermediateNode "S" via ``policy_overrides={"S": policy}``.

    Uses ``Runner.run()`` → ``flow_frame`` / ``purchase_frame`` /
    ``closing_inventory_frame`` → ``business_metrics`` + ``profit_decomposition``
    (the shared DataFrame-native path per ADR 0019).

    ``holding_rate`` and ``order_fee`` are sourced from ``spec.scenario``
    (IntermediateNode "S") — not from any config module.

    Parameters
    ----------
    policy:
        Any policy implementing ``decide(obs_intermediate, central_table)``.
    spec:
        ``TuningEpisodeSpec`` from ``episode.sample_episode``.

    Returns
    -------
    dict[str, float]
        Aggregate KPIs: ``service_level``, ``stockout_rate``,
        ``inventory_turnover``, ``mean_price_pct_of_msrp``,
        ``revenue``, ``net_profit``.
    """
    from src.sim.runner import Runner
    from src.sim.inspect import (
        flow_frame,
        purchase_frame,
        closing_inventory_frame,
        node_timeseries_df,
    )
    from src.sim.metrics import business_metrics, profit_decomposition

    scenario = spec.scenario

    runner = Runner(scenario, policy_overrides={"S": policy})
    run_log = runner.run()

    ff = flow_frame(run_log)
    pf = purchase_frame(run_log)
    cif = closing_inventory_frame(run_log)
    ts_df = node_timeseries_df(run_log, scenario)

    bm = business_metrics(ff, ts_df, scenario)
    pd_df = profit_decomposition(ff, pf, cif, scenario)

    # Extract node "S" row from business_metrics.
    s_bm = bm[bm["node_id"] == "S"]
    if s_bm.empty:
        # Fallback: use _system row if "S" not present.
        s_bm = bm[bm["node_id"] == "_system"]

    service_level = float(s_bm["service_level"].iloc[0]) if not s_bm.empty else 0.0
    stockout_rate = float(s_bm["stockout_rate"].iloc[0]) if not s_bm.empty else 0.0
    inventory_turnover = float(s_bm["inventory_turnover"].iloc[0]) if not s_bm.empty else 0.0
    mean_price_pct_of_msrp = float(s_bm["mean_price_pct_of_msrp"].iloc[0]) if not s_bm.empty else 1.0

    # Extract node "S" row from profit_decomposition.
    s_pd = pd_df[pd_df["node_id"] == "S"]
    net_profit = float(s_pd["net_profit"].iloc[0]) if not s_pd.empty else 0.0
    revenue = float(s_pd["revenue"].iloc[0]) if not s_pd.empty else 0.0

    return {
        "service_level": service_level,
        "stockout_rate": stockout_rate,
        "inventory_turnover": inventory_turnover,
        "mean_price_pct_of_msrp": mean_price_pct_of_msrp,
        "revenue": revenue,
        "net_profit": net_profit,
    }


__all__ = ["run_policy_episode"]
