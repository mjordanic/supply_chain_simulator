"""Render the four simulator-section images for the README.

Outputs land under docs/images/:
    sim_market_supply_demand.png
    sim_equity_composition.png
    sim_revenue_vs_cost.png
    sim_product_P0333.png

Reuses the same logic as notebooks/04a-deep_dive_active_only.ipynb against
the canned run at data/llm_world_250/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parent.parent
RUN_DIR = REPO / "data" / "llm_world_250"
OUT_DIR = REPO / "docs" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STORE_ID = 0
TARGET_PID = "P0333"

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 130
plt.rcParams["lines.linewidth"] = 1.8
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 9


def load() -> dict:
    data_dir = RUN_DIR / "data"
    cfg_dir = RUN_DIR / "config"
    products = pd.read_parquet(data_dir / "products.parquet")
    ts = pd.read_parquet(data_dir / "timeseries.parquet")
    with open(data_dir / "run_log.json") as f:
        run_log = json.load(f)
    ts = ts.merge(
        products[["product_id", "name", "category", "base_price", "unit_cost", "seasonality"]],
        on="product_id",
        how="left",
    )
    ts["simulation_date"] = pd.to_datetime(ts["simulation_date"])
    ts = ts[ts["store_id"] == STORE_ID].sort_values(
        ["product_id", "simulation_step"]
    ).reset_index(drop=True)
    sim_steps = run_log["global"]["time"]["simulation_step"]
    return {"ts": ts, "run_log": run_log, "sim_steps": sim_steps}


def plot_market(state: dict) -> None:
    run_log = state["run_log"]
    sim_steps = state["sim_steps"]
    occurrences = run_log["global"]["events"]["occurrences"]

    def _sig(ev):
        return (ev["type"], float(ev["severity"]), tuple(ev["regions"]))

    events = []
    prev = set()
    for step_idx, active in enumerate(occurrences):
        curr = {_sig(ev) for ev in active}
        new = curr - prev
        for ev in active:
            if _sig(ev) not in new:
                continue
            end = sim_steps[min(step_idx + int(ev["duration"]) - 1, len(sim_steps) - 1)]
            events.append({
                "start_step": sim_steps[step_idx],
                "end_step": end,
                "type": ev["type"],
                "severity": float(ev["severity"]),
                "regions": list(ev["regions"]),
            })
        prev = curr
    events_df = pd.DataFrame(events)
    event_types = sorted({e["type"] for e in events})
    type_color = dict(zip(event_types, cm.Set2(np.linspace(0, 1, max(1, len(event_types))))))

    regions = list(run_log["global"]["market_supply"].keys())
    fig, axes = plt.subplots(len(regions), 1, figsize=(12, 3.2 * len(regions)),
                              sharex=True, squeeze=False)
    for i, region in enumerate(regions):
        ax = axes[i, 0]
        supply = run_log["global"]["market_supply"][region]
        demand = run_log["global"]["market_demand"][region]
        ax.plot(sim_steps, supply, color="#1f77b4", lw=2.0, label="Supply", alpha=0.95)
        ax.plot(sim_steps, demand, color="#d62728", lw=2.0, label="Demand", alpha=0.95)
        if not events_df.empty:
            for _, ev in events_df.iterrows():
                if region not in ev["regions"]:
                    continue
                ax.axvspan(ev["start_step"], ev["end_step"],
                           color=type_color[ev["type"]], alpha=0.22)
                ax.text(ev["start_step"], ax.get_ylim()[1] * 0.96 if ax.get_ylim()[1] else 1,
                        f"{ev['type']}\n×{ev['severity']:.1f}",
                        fontsize=7, va="top", ha="left",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                  edgecolor=type_color[ev["type"]], alpha=0.8))
        ax.set_title(f"Region: {region}")
        ax.set_ylabel("Units")
        ax.grid(True, alpha=0.35)
        handles, labels = ax.get_legend_handles_labels()
        for t in event_types:
            if (not events_df.empty) and any(
                t == ev["type"] and region in ev["regions"] for _, ev in events_df.iterrows()
            ):
                handles.append(Patch(color=type_color[t], alpha=0.4, label=t))
                labels.append(t)
        ax.legend(handles, labels, loc="upper right")
    axes[-1, 0].set_xlabel("Simulation step")
    fig.suptitle("Market supply / demand per region (disruption windows shaded)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sim_market_supply_demand.png")
    plt.close(fig)


def _store_step(state: dict) -> pd.DataFrame:
    ts = state["ts"]
    run_log = state["run_log"]
    sim_steps = state["sim_steps"]
    cash = run_log["stores"][str(STORE_ID)]["balance"]
    cash_df = pd.DataFrame({"simulation_step": sim_steps, "cash": cash,
                            "store_id": STORE_ID})
    val = ts.assign(
        inventory_value=ts["inventory"] * ts["unit_cost"],
        outstanding_value=ts["outstanding_orders"] * ts["unit_cost"],
    )
    g = (val.groupby(["store_id", "simulation_step"], as_index=False)
            .agg(inventory_value=("inventory_value", "sum"),
                 outstanding_value=("outstanding_value", "sum"),
                 step_profit=("profit", "sum"),
                 revenue=("revenue", "sum"),
                 total_cost=("total_cost", "sum"),
                 holding_cost=("holding_cost", "sum")))
    g = g.merge(cash_df, on=["store_id", "simulation_step"])
    g["equity"] = g["cash"] + g["inventory_value"] + g["outstanding_value"]
    g = g.sort_values("simulation_step")
    g["cumulative_profit"] = g["step_profit"].cumsum()
    return g


def plot_equity(state: dict) -> None:
    g = _store_step(state)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(
        g["simulation_step"],
        g["cash"], g["inventory_value"], g["outstanding_value"],
        labels=["Cash", "Inventory value (at cost)", "Outstanding orders (at cost)"],
        colors=["#1f77b4", "#2ca02c", "#ff7f0e"], alpha=0.85,
    )
    ax.plot(g["simulation_step"], g["equity"], color="black", lw=2.2, ls="--",
            label="Total equity")
    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Equity components")
    ax.set_title(f"Store {STORE_ID} — equity composition + cumulative P&L")
    ax2 = ax.twinx()
    ax2.plot(g["simulation_step"], g["cumulative_profit"], color="crimson", lw=2.2,
             label="Cumulative P&L")
    ax2.set_ylabel("Cumulative P&L", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sim_equity_composition.png")
    plt.close(fig)


def plot_revenue_costs(state: dict) -> None:
    g = _store_step(state)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(g["simulation_step"], g["revenue"], color="#2ca02c", alpha=0.9,
           label="Revenue")
    order_cost = g["total_cost"] - g["holding_cost"]
    ax.bar(g["simulation_step"], -order_cost, color="#d62728", alpha=0.9,
           label="Order cost")
    ax.bar(g["simulation_step"], -g["holding_cost"], bottom=-order_cost,
           color="#9467bd", alpha=0.9, label="Holding cost")
    ax.plot(g["simulation_step"], g["step_profit"], color="black", lw=2.0,
            marker="o", ms=3.2, label="Step P&L")
    ax.axhline(0, color="gray", lw=0.7)
    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Per-step amount")
    ax.set_title(f"Store {STORE_ID} — revenue vs costs per step")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sim_revenue_vs_cost.png")
    plt.close(fig)


def plot_product(state: dict) -> None:
    ts = state["ts"]
    run_log = state["run_log"]
    g = ts[ts["product_id"] == TARGET_PID].sort_values("simulation_step").reset_index(drop=True)
    if g.empty:
        raise SystemExit(f"product {TARGET_PID} not in run")
    store_total_capacity = float(run_log["stores"][str(STORE_ID)]["step0_capacity"])
    name = g["name"].iloc[0]
    cat = g["category"].iloc[0]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax_inv, ax_ds = axes

    ax_inv.plot(g["simulation_step"], g["inventory"], color="#1f77b4", lw=2.0,
                label="Inventory")
    ax_inv.plot(g["simulation_step"], g["outstanding_orders"], color="#ff7f0e",
                lw=2.0, label="Outstanding")
    ax_inv.bar(g["simulation_step"], g["order_quantity"], color="#2ca02c",
               alpha=0.85, label="Order qty")
    ax_inv.axhline(
        store_total_capacity, color="black", lw=1.8, ls=":", zorder=5,
        label=f"Store capacity ({store_total_capacity:.0f} units, total)",
    )
    ax_inv.set_ylabel("Units")
    ax_inv.grid(True, alpha=0.4)
    ax_inv.legend(loc="upper right")

    ax_ds.plot(g["simulation_step"], g["demand"], color="#9467bd", lw=2.0,
               ls="--", label="Demand (always sampled)")
    ax_ds.plot(g["simulation_step"], g["sales"], color="#2ca02c", lw=2.0,
               label="Sales")
    ax_ds.fill_between(g["simulation_step"], g["sales"], g["demand"],
                       where=(g["demand"] > g["sales"]),
                       color="#d62728", alpha=0.32, label="Unmet")
    inactive = (~g["active_status"]).astype(int).values
    if inactive.any():
        ymax = max(1, int(g["demand"].max()))
        ax_ds.fill_between(g["simulation_step"], 0, ymax * 1.05,
                           where=inactive.astype(bool), step="post",
                           color="lightgray", alpha=0.35, label="Inactive window")
    ax_ds.set_ylabel("Units")
    ax_ds.set_xlabel("Simulation step")
    ax_ds.grid(True, alpha=0.4)
    ax_ds.legend(loc="upper right")

    totals = (f'demand={int(g["demand"].sum())}  sales={int(g["sales"].sum())}  '
              f'active_steps={int(g["active_status"].sum())}/{len(g)}')
    fig.suptitle(f"{TARGET_PID}  —  {name}  ({cat})\n{totals}", y=0.995)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"sim_product_{TARGET_PID}.png")
    plt.close(fig)


def main() -> None:
    state = load()
    plot_market(state)
    plot_equity(state)
    plot_revenue_costs(state)
    plot_product(state)
    print(f"wrote 4 images under {OUT_DIR}")


if __name__ == "__main__":
    main()
