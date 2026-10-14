"""Tables and plots for the perfect-supplier runs (used by notebook 15).

Every eval world is run once per arm, each arm on its own copy of the world,
with the same demand path. ``paired_table`` therefore compares arms world by
world. Plots take the frames from :mod:`src.perfect_supplier.data`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = {
    "jepa": "JEPA planner",
    "direct": "direct-forecast planner",
    "planner_moving_average": "planner + moving average",
    "planner_oracle": "planner + oracle forecast",
    "tuned_order_up_to": "tuned (s,S)",
    "tuned_reorder_point": "tuned (s,Q)",
    "tuned_periodic_order_up_to": "tuned (R,S)",
    "tuned_periodic_reorder": "tuned (R,s,S)",
    "tuned_eoq": "tuned EOQ (s,Q)",
    "tuned_eoq_joint": "tuned joint EOQ",
    "default_order_up_to": "default (s,S)",
    "never_order": "never order",
}


def bootstrap_ci(x: np.ndarray, n: int = 5000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def _complete(m: pd.DataFrame) -> pd.DataFrame:
    """Fail rather than silently compare arms on different worlds."""
    missing = m.isna().sum()
    if missing.any():
        raise ValueError(f"arms missing eval worlds: {missing[missing > 0].to_dict()}")
    return m


def score_matrix(summary: pd.DataFrame, metric: str = "score", split: str = "eval") -> pd.DataFrame:
    s = summary[summary["split"] == split]
    return s.pivot_table(index="world_seed", columns="method", values=metric)


def paired_table(summary: pd.DataFrame, baseline: str, metric: str = "score", split: str = "eval") -> pd.DataFrame:
    """Per arm: mean metric, mean paired difference to ``baseline``, its 95% CI, and win rate."""
    m = _complete(score_matrix(summary, metric, split))
    rows = []
    for arm in m.columns:
        diff = (m[arm] - m[baseline]).to_numpy()
        lo, hi = bootstrap_ci(diff) if arm != baseline else (0.0, 0.0)
        rows.append(
            {
                "arm": arm,
                f"mean_{metric}": m[arm].mean(),
                "diff_vs_baseline": diff.mean(),
                "ci_low": lo,
                "ci_high": hi,
                "win_rate": float((diff > 0).mean()),
                "n_worlds": len(diff),
            }
        )
    return pd.DataFrame(rows).set_index("arm").sort_values(f"mean_{metric}", ascending=False)


def pairwise(summary: pd.DataFrame, pairs: Sequence[tuple[str, str]], split: str = "eval") -> pd.DataFrame:
    """Mean paired score difference ``a - b`` in points, its 95% CI, and a's win rate, per pair."""
    m = _complete(score_matrix(summary, split=split))
    rows = []
    for a, b in pairs:
        d = (m[a] - m[b]).to_numpy()
        lo, hi = bootstrap_ci(d)
        rows.append(
            {"arm": a, "vs": b, "diff (pts)": d.mean() * 100, "ci low": lo * 100, "ci high": hi * 100,
             "win rate": float((d > 0).mean()), "n_worlds": len(d)}
        )
    return pd.DataFrame(rows)


def best_textbook(root: str | Path) -> str:
    """The tuned textbook arm with the best score on the tune worlds.

    Chosen on tuning data, not on the eval worlds it is then compared on.
    """
    scores = json.loads((Path(root) / "tuning" / "best_scores.json").read_text())
    return "tuned_" + max(scores, key=lambda f: scores[f]["tune_score"])


def breakdown(summary: pd.DataFrame, split: str = "eval") -> pd.DataFrame:
    """Where each arm loses margin, as a share of the margin on all demand.

    ``score + lost_sales + fees_share + holding_share`` is exactly 1: equity
    values stock at cost, so only these three take margin away.
    """
    s = summary[summary["split"] == split].copy()
    margin = s["total_demand"] * (s["price"] - s["unit_cost"])
    s["lost_sales"] = s["lost_units"] * (s["price"] - s["unit_cost"]) / margin
    s["fees_share"] = s["fees"] / margin
    s["holding_share"] = s["holding_cost"] / margin
    cols = ["score", "lost_sales", "fees_share", "holding_share", "order_days", "fill_rate_demand", "mean_on_hand"]
    return s.groupby("method")[cols].mean().sort_values("score", ascending=False)


def plot_paired(summary: pd.DataFrame, baseline: str, arms: Sequence[str] | None = None, ax=None):
    """Box plot of world-by-world score differences to ``baseline``."""
    m = _complete(score_matrix(summary))
    arms = [a for a in (arms or m.columns) if a != baseline]
    diffs = [(m[a] - m[baseline]).to_numpy() * 100 for a in arms]
    ax = ax or plt.subplots(figsize=(9, 0.5 * len(arms) + 1.5))[1]
    ax.boxplot(diffs, vert=False, labels=[LABELS.get(a, a) for a in arms], showfliers=False)
    ax.axvline(0, color="grey", lw=1)
    ax.set_xlabel(f"score − {LABELS.get(baseline, baseline)} (points of margin captured, per world)")
    return ax


def plot_by_world_feature(summary: pd.DataFrame, baseline: str, feature: str, arms: Sequence[str], ax=None):
    """Mean paired difference to ``baseline`` for each value of a world feature."""
    s = summary[summary["split"] == "eval"]
    m = _complete(score_matrix(summary))
    feat = s.drop_duplicates("world_seed").set_index("world_seed")[feature].reindex(m.index)
    if feat.nunique() > 8:
        feat = pd.qcut(feat, 4)
    ax = ax or plt.subplots(figsize=(7, 4))[1]
    for arm in arms:
        d = ((m[arm] - m[baseline]) * 100).groupby(feat, observed=True).mean()
        ax.plot([str(i) for i in d.index], d.values, marker="o", label=LABELS.get(arm, arm))
    ax.axhline(0, color="grey", lw=1)
    ax.set_xlabel(feature)
    ax.set_ylabel(f"score − {LABELS.get(baseline, baseline)} (points)")
    ax.legend(fontsize=8)
    return ax


def plot_world(
    ts: pd.DataFrame,
    world_seed: int,
    product: int,
    methods: Sequence[str],
    days: tuple[int, int] | None = None,
):
    """One product of one world, one line per arm, each arm on its own run.

    Panels: demand and each arm's sales; closing on-hand and in-transit;
    units ordered (filled); shop equity.
    """
    g = ts[(ts["world_seed"] == world_seed) & (ts["product"] == product) & ts["method"].isin(methods)]
    if days:
        g = g[(g["day"] >= days[0]) & (g["day"] < days[1])]
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    first = g[g["method"] == methods[0]]
    axes[0].bar(first["day"], first["demand"], color="lightgrey", label="demand", width=1.0)
    axes[0].plot(first["day"], first["demand_mean"], color="black", lw=1, label="true mean")
    for m in methods:
        r = g[g["method"] == m]
        label = LABELS.get(m, m)
        axes[0].plot(r["day"], r["sales"], lw=1, label=f"sales {label}")
        line = axes[1].plot(r["day"], r["on_hand_close"], lw=1.2, label=f"on hand {label}")[0]
        axes[1].plot(r["day"], r["in_transit"], lw=0.8, ls=":", color=line.get_color())
        o = r[r["filled"] > 0]
        axes[2].vlines(o["day"], 0, o["filled"], color=line.get_color(), lw=1.5, label=label)
        eq = r.groupby("day")["equity_close"].first()
        axes[3].plot(eq.index, eq.values, color=line.get_color(), label=label)
    axes[0].set_ylabel("units / day")
    axes[1].set_ylabel("on hand (solid)\nin transit (dotted)")
    axes[2].set_ylabel("units ordered")
    axes[3].set_ylabel("equity")
    axes[3].set_xlabel("day")
    for ax in axes:
        ax.legend(fontsize=7, ncol=3, loc="upper left")
    fig.suptitle(f"world {world_seed}, product {product}")
    fig.tight_layout()
    return fig


def plot_shadow(ts: pd.DataFrame, world_seed: int, product: int, days: tuple[int, int] | None = None):
    """One shadow run: the driver's orders and what every shadow arm would have ordered on the same state."""
    g = ts[(ts["world_seed"] == world_seed) & (ts["product"] == product)].dropna(axis=1, how="all")
    if days:
        g = g[(g["day"] >= days[0]) & (g["day"] < days[1])]
    shadow_cols = [c for c in g.columns if c.startswith("shadow_")]
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes[0].plot(g["day"], g["position"], label="position (driver)")
    axes[0].plot(g["day"], g["on_hand_close"], label="on hand (driver)")
    axes[0].bar(g["day"], g["demand"], color="lightgrey", width=1.0, label="demand")
    axes[0].legend(fontsize=8)
    driver = g["method"].iloc[0]
    axes[1].vlines(g["day"], 0, g["requested"], color="black", lw=2, label=f"{LABELS.get(driver, driver)} (driver)")
    for i, c in enumerate(shadow_cols):
        name = c.removeprefix("shadow_").removesuffix("_requested")
        o = g[g[c] > 0]
        axes[1].scatter(o["day"] + 0.15 * (i + 1), o[c], s=12, label=LABELS.get(name, name))
    axes[1].set_ylabel("units requested")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"shadow run, world {world_seed}, product {product}")
    fig.tight_layout()
    return fig


def plot_forecast(ts: pd.DataFrame, world_seed: int, product: int, method: str = "jepa"):
    """The planner's 7-day median forecast against the 7 days of demand that followed."""
    g = ts[(ts["world_seed"] == world_seed) & (ts["product"] == product) & (ts["method"] == method)].sort_values("day")
    actual = g["demand"].shift(-1).rolling(7).sum().shift(-6)
    true_mean = g["demand_mean"].shift(-1).rolling(7).sum().shift(-6)
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(g["day"], g["dec_forecast_mean_7d"], label=f"{LABELS.get(method, method)} median, next 7 days")
    ax.plot(g["day"], actual, label="actual demand, next 7 days", alpha=0.6)
    ax.plot(g["day"], true_mean, label="true mean, next 7 days", color="black", lw=1)
    c = g[g["censored"]]
    ax.scatter(c["day"], np.zeros(len(c)), marker="|", color="red", label="sold out")
    ax.legend(fontsize=8)
    ax.set_xlabel("day")
    return fig
