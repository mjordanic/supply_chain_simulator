"""DataExporter for the new ``Scenario``-shaped runs (issue 07).

Replaces the previous ``src/simulator/data_exporter.py`` (705 lines)
with a focused module that consumes a ``Scenario`` plus a ``RunLog``
produced by ``Runner`` and writes:

- **parquet** — per-store / per-product time-series tables, plus static
  product and store tables.
- **JSON** — a serialisable view of the scenario (via
  ``Scenario.to_json``) and a flattened run-log payload that drops
  non-JSON-friendly objects (datetimes → ISO strings).
- **PNG** — a regional supply / demand overview chart.

Output folder layout (relative to ``output_folder`` argument)::

    config/scenario.json
    data/run_log.json
    data/products.parquet
    data/stores.parquet
    data/timeseries.parquet
    reports/overview.png

The old version's per-product detail plots, the lambda-string config
serializer, and the observation/action trace exporter are deleted —
they were tied to the deleted ``init_params`` / ``live_params``
config bundle. Adding them back is straightforward when needed.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.sim.scenario import Scenario


class DataExporter:
    """Persist a ``Scenario`` + ``RunLog`` pair to parquet / JSON / PNG.

    All public ``export_*`` / ``save_*`` methods accept an
    ``output_folder`` argument and create the necessary subdirectories.
    """

    def __init__(self, scenario: Scenario, run_log: dict[str, Any]) -> None:
        # Authoring artifact — used for the static catalog/store dataframes
        # and to JSON-dump the scenario into ``config/scenario.json``.
        self.scenario = scenario
        # Nested mutable dict produced by ``Runner.run`` — has the full
        # per-step / per-store / per-product time-series under
        # ``["stores"][store_id]["products"][pid][metric]``.
        self.run_log = run_log

    # ------------------------------------------------------------------ public

    def export_all(self, output_folder: str) -> None:
        """Write every supported output: scenario JSON, parquet tables, PNG."""
        self.save_scenario_json(output_folder)
        self.save_run_log_json(output_folder)
        self.save_products_parquet(output_folder)
        self.save_stores_parquet(output_folder)
        self.save_timeseries_parquet(output_folder)
        self.save_overview_plot(output_folder)

    def save_scenario_json(self, output_folder: str) -> str:
        """Write ``scenario.json`` via ``Scenario.to_json``. Returns the path."""
        # ``config/`` sub-directory keeps config separate from raw run data.
        folder = os.path.join(output_folder, "config")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "scenario.json")
        with open(path, "w") as f:
            # ``Scenario.to_json`` already omits policies — they're
            # Python objects, not JSON-serialisable, and policy choice
            # is wired up at experiment-authoring time anyway.
            f.write(self.scenario.to_json())
        return path

    def save_run_log_json(self, output_folder: str) -> str:
        """Write the full run-log as JSON. Datetimes are serialised as ISO strings."""
        folder = os.path.join(output_folder, "data")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "run_log.json")
        with open(path, "w") as f:
            # ``_jsonable`` recurses through the run log dropping
            # set/datetime objects; ``default=str`` is a last-ditch
            # fallback for anything ``_jsonable`` missed.
            json.dump(_jsonable(self.run_log), f, default=str)
        return path

    def save_products_parquet(self, output_folder: str) -> str:
        """Write the static product table as ``products.parquet``.

        Resolved ``freshness_alpha`` / ``freshness_decay`` columns
        (issue 04) and ``init_stock_share`` (issue 08) come from
        ``run_log["global"]["products"][pid]``, which the Runner
        populates from ``ItemRegistry.items[pid]`` once at construction.
        Distribution-typed Ware overrides are therefore captured as the
        actual scalars used by the run.
        """
        folder = os.path.join(output_folder, "data")
        os.makedirs(folder, exist_ok=True)
        # Pull the resolved per-product scalars the runner cached.
        product_log = self.run_log.get("global", {}).get("products", {})
        # Start from the scenario's static catalog dataframe (catalog
        # ordering preserved).
        base = self.scenario.catalog_df()
        # Overwrite the freshness / stock-share columns with the
        # *resolved* scalars from the run log so the parquet captures
        # what the run actually used (not an unresolved Distribution).
        products_df = base.assign(
            freshness_alpha=base["product_id"].map(
                lambda pid: product_log.get(pid, {}).get("freshness_alpha")
            ),
            freshness_decay=base["product_id"].map(
                lambda pid: product_log.get(pid, {}).get("freshness_decay")
            ),
            init_stock_share=base["product_id"].map(
                lambda pid: product_log.get(pid, {}).get("init_stock_share")
            ),
        )[
            [
                "product_id",
                "name",
                "category",
                "base_price",
                "unit_cost",
                "seasonality",
                "freshness_alpha",
                "freshness_decay",
                "init_stock_share",
            ]
        ]
        path = os.path.join(folder, "products.parquet")
        products_df.to_parquet(path)
        return path

    def save_stores_parquet(self, output_folder: str) -> str:
        """Write the static store table as ``stores.parquet``."""
        folder = os.path.join(output_folder, "data")
        os.makedirs(folder, exist_ok=True)
        # ``Scenario.stores_df`` exposes a ``policy_class`` column; the
        # output schema standardises on ``policy_type`` so downstream
        # consumers (notebooks/dashboards) get a single canonical name.
        stores_df = self.scenario.stores_df().rename(
            columns={"policy_class": "policy_type"}
        )
        path = os.path.join(folder, "stores.parquet")
        stores_df.to_parquet(path)
        return path

    def save_timeseries_parquet(self, output_folder: str) -> str:
        """Write per-store / per-product time-series as ``timeseries.parquet``."""
        folder = os.path.join(output_folder, "data")
        os.makedirs(folder, exist_ok=True)

        # Step / date axes are shared across every (store, product) row.
        sim_steps = self.run_log["global"]["time"]["simulation_step"]
        sim_dates = self.run_log["global"]["time"]["simulation_date"]
        # Accumulator for the long-form (one row per step × store × product) frame.
        rows: list[dict[str, Any]] = []
        for store_id, store_log in self.run_log["stores"].items():
            for pid, product_log in store_log["products"].items():
                for t, step in enumerate(sim_steps):
                    rows.append(
                        {
                            "simulation_step": step,
                            "simulation_date": sim_dates[t],
                            "store_id": store_id,
                            "product_id": pid,
                            "inventory": product_log["inventory"][t],
                            "demand": product_log["demand"][t],
                            "sales": product_log["sales"][t],
                            "order_quantity": product_log["order_quantity"][t],
                            "outstanding_orders": product_log["outstanding_orders"][t],
                            "promotion_status": product_log["promotion_status"][t],
                            "active_status": product_log["active_status"][t],
                            "price": product_log["price"][t],
                            "revenue": product_log["revenue"][t],
                            "total_cost": product_log["total_cost"][t],
                            "holding_cost": product_log["holding_cost"][t],
                            "profit": product_log["profit"][t],
                        }
                    )
        path = os.path.join(folder, "timeseries.parquet")
        pd.DataFrame(rows).to_parquet(path)
        return path

    def save_overview_plot(self, output_folder: str) -> str:
        """Render a regional supply / demand overview to ``overview.png``."""
        folder = os.path.join(output_folder, "reports")
        os.makedirs(folder, exist_ok=True)

        # Region keys come from the run log's market_supply dict.
        regions = list(self.run_log["global"]["market_supply"].keys())
        # X-axis: simple integer step index, same length as the per-region
        # supply/demand series.
        time_steps = list(range(len(self.run_log["global"]["time"]["simulation_step"])))

        # One sub-plot per region, stacked vertically. ``squeeze=False``
        # so ``axes`` is always a 2-D array regardless of region count.
        fig, axes = plt.subplots(
            len(regions), 1, figsize=(12, 4 * len(regions)), sharex=True, squeeze=False
        )
        for i, region in enumerate(regions):
            ax = axes[i, 0]
            # Supply line.
            ax.plot(
                time_steps,
                self.run_log["global"]["market_supply"][region],
                label="Supply",
                color="tab:blue",
            )
            # Demand line on the same axes — visual overlap is the point.
            ax.plot(
                time_steps,
                self.run_log["global"]["market_demand"][region],
                label="Demand",
                color="tab:red",
            )
            ax.set_title(f"Region {region}")
            ax.set_ylabel("Units")
            ax.legend(loc="upper right")
        # Only the last sub-plot needs the x-axis label.
        axes[-1, 0].set_xlabel("Simulation step")

        path = os.path.join(folder, "overview.png")
        fig.tight_layout()
        fig.savefig(path)
        # Always close the figure to release matplotlib's global state.
        plt.close(fig)
        return path


def _jsonable(value: Any) -> Any:
    """Recursively convert ``run_log`` payloads into JSON-friendly forms."""
    if isinstance(value, dict):
        # ``str(k)`` defends against int / tuple keys that JSON rejects.
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        # JSON has no native tuple — serialise as a list.
        return [_jsonable(v) for v in value]
    if isinstance(value, datetime):
        # ISO 8601 round-trips cleanly through ``datetime.fromisoformat``.
        return value.isoformat()
    if isinstance(value, set):
        # Sets are unordered — list-coercion is the best JSON can do.
        return [_jsonable(v) for v in value]
    return value


__all__ = ["DataExporter"]
