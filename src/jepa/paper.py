"""Draft the replenishment paper from a run directory.

Result cells are means, the paired interval, the pass word, and the
held-out rollout curve. They are read from the synthetic tables, the
rollout file, the manifest, and the study files, or computed by the
comparison functions. This module does not run a study and does not
invent a cell. A refused profit table is reported as a refusal.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from src.jepa.commands import M5_COMMAND
from src.jepa.eval import (
    HOLDING_RATE_SWEEP_RULE,
    IN_DISTRIBUTION_RULE,
    in_distribution_passes,
    is_profit_column,
    paired_interval,
)

_PROTOCOL_FILES = (
    "in_distribution",
    "unseen_sku",
    "unseen_disruption",
    "holding_rate_sweep",
)
_MANIFEST_FIELDS = (
    "episode_count",
    "seed_min",
    "seed_max",
    "episode_length",
    "K_active",
    "K_min",
    "K_max_episode",
    "delivery_lag",
    "holding_rate",
    "order_fee",
    "catalog_identity",
    "evaluation_catalog_identity",
    "training_sku_count",
    "evaluation_sku_count",
    "git_sha",
    "nonzero_order_fraction",
)
_MANIFEST_OBJECTS = (
    "disruption",
    "harsh_disruption",
    "protocols",
    "state_coverage",
)
_OPERATIONAL = (
    "_order_volume",
    "_service_level",
    "_stockout_rate",
    "_fill_rate",
)


def _format_decimal(value: float) -> str:
    """Six decimal places, the same rounding for every table cell."""
    return f"{float(value):.6f}"


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_frame(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    return pd.read_parquet(path)


def _per_episode_counts(frame: pd.DataFrame, column: str) -> str:
    per_episode = frame.groupby(["behavior", "episode_seed"])[column].nunique()
    return ",".join(str(int(count)) for count in sorted(set(per_episode)))


def _seed_span(label: str, frame: pd.DataFrame) -> list[str]:
    return [
        f"{label}_seeds {len(frame)}",
        f"{label}_seed_min {int(frame['seed'].min())}",
        f"{label}_seed_max {int(frame['seed'].max())}",
    ]


def _means(frame: pd.DataFrame) -> list[str]:
    lines = [
        "| arm | mean net_profit / initial_cash |",
        "| --- | --- |",
    ]
    for column in frame.columns:
        if not is_profit_column(str(column)):
            continue
        lines.append(f"| {column} | {_format_decimal(frame[column].mean())} |")
    return lines


def _operational(frame: pd.DataFrame) -> list[str]:
    arms = [
        column[: -len("_order_volume")]
        for column in frame.columns
        if str(column).endswith("_order_volume")
    ]
    if not arms:
        return []
    lines = [
        "| arm | mean order volume | mean service level | mean stockout rate | mean fill rate |",
        "| --- | --- | --- | --- | --- |",
    ]
    for arm in arms:
        cells = []
        for suffix in _OPERATIONAL:
            column = f"{arm}{suffix}"
            if column in frame.columns:
                cells.append(_format_decimal(frame[column].mean()))
            else:
                cells.append("")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    return lines


def _mean_r2(values: list) -> float | None:
    finite: list[float] = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if math.isnan(number):
            continue
        finite.append(number)
    if not finite:
        return None
    return sum(finite) / len(finite)


def _rollout_means(accuracy: dict) -> list[str]:
    lines = ["Held-out rollout accuracy, mean R² over the planning horizon:"]
    for name in ("latent", "direct", "ols"):
        mean = _mean_r2(list(accuracy.get(name) or []))
        rendered = "none" if mean is None else _format_decimal(mean)
        lines.append(f"rollout_r2_{name}_mean {rendered}")
    return lines


def _rollout_steps(accuracy: dict) -> list[str]:
    lines = ["Per-step rollout R²:"]
    steps = list(accuracy.get("steps") or [])
    for index, step in enumerate(steps):
        parts = [f"step {int(step)}"]
        for name in ("latent", "direct", "ols"):
            series = list(accuracy.get(name) or [])
            value = series[index] if index < len(series) else None
            rendered = "none" if value is None else _format_decimal(value)
            parts.append(f"{name} {rendered}")
        lines.append(" ".join(parts))
    return lines


def render_paper(run_dir: str | Path) -> str:
    """Return the draft for ``run_dir``. Missing artifacts are named, not filled in."""
    run = Path(run_dir)
    synthetic = run / "synthetic"
    accuracy = _load_json(synthetic / "rollout_accuracy.json")
    frames = {name: _load_frame(synthetic / f"{name}.parquet") for name in _PROTOCOL_FILES}
    lines = [
        "# Latent replenishment",
        "",
        *_method(),
        "",
        *_run_record(run, frames),
        "",
        *_rollout_section(accuracy),
        "",
        *_profit_sections(
            frames,
            accuracy,
            _load_json(synthetic / "profit_table_refused.json"),
            _load_json(synthetic / "holding_rate_sweep_refused.json"),
        ),
        "",
        *_warnings(synthetic),
        *_m5(run),
        "",
        *_future(),
    ]
    text = "\n".join(lines)
    if not text.endswith("\n"):
        text += "\n"
    return text


def _profit_sections(
    frames: dict[str, pd.DataFrame | None],
    accuracy: dict | None,
    profit_refusal: dict | None,
    sweep_refusal: dict | None,
) -> list[str]:
    if profit_refusal is not None:
        return [
            "## Profit tables",
            "",
            "The profit table was refused.",
            f"refusal {json.dumps(profit_refusal, sort_keys=True)}",
            "",
            "This draft has no profit means. The rollout accuracy above is the",
            "measurement the gate used.",
        ]
    lines: list[str] = []
    lines.extend(_protocol_block(
        "In-distribution",
        frames["in_distribution"],
        "in_distribution.parquet",
        accuracy,
        in_distribution=True,
    ))
    lines.extend(_protocol_block(
        "Unseen SKU",
        frames["unseen_sku"],
        "unseen_sku.parquet",
        accuracy,
    ))
    lines.extend(_protocol_block(
        "Unseen disruption",
        frames["unseen_disruption"],
        "unseen_disruption.parquet",
        accuracy,
    ))
    lines.extend(_sweep_block(frames["holding_rate_sweep"], accuracy, sweep_refusal))
    return lines


def _beside(accuracy: dict | None) -> list[str]:
    if accuracy is None:
        return [
            "This section has no held-out rollout file, so no rollout R² is "
            "printed beside the profit means.",
        ]
    return _rollout_means(accuracy)


def _protocol_block(
    title: str,
    frame: pd.DataFrame | None,
    filename: str,
    accuracy: dict | None,
    *,
    in_distribution: bool = False,
) -> list[str]:
    lines = [f"## {title}", ""]
    if title == "In-distribution":
        lines.extend([
            IN_DISTRIBUTION_RULE,
            "",
            "The in-distribution claim is that rule.",
            "",
        ])
    elif title == "Unseen disruption":
        lines.extend([
            "Training SKUs, and a harsher disruption regime the training log",
            "never samples. This protocol is not future work.",
            "",
        ])
    if frame is None:
        lines.append(
            f"`synthetic/{filename}` is not in this run, so this draft has no numbers for it."
        )
        lines.append("")
        return lines
    lines.extend(_beside(accuracy))
    lines.append("")
    lines.extend(_means(frame))
    lines.append("")
    operational = _operational(frame)
    if operational:
        lines.extend(operational)
        lines.append("")
    if in_distribution and {"latent", "anchor"} <= set(frame.columns):
        low, high = paired_interval(frame["latent"], frame["anchor"])
        word = "pass" if in_distribution_passes(frame["latent"], frame["anchor"]) else "fail"
        lines.append(f"paired_interval {_format_decimal(low)} {_format_decimal(high)}")
        lines.append(f"in_distribution_passes {word}")
        lines.append("")
    elif in_distribution:
        lines.append(
            "in_distribution_passes unavailable; latent or anchor was omitted "
            "from the parquet as a collinear arm. See comparison_warnings.json."
        )
        lines.append("")
    return lines


def _sweep_block(
    frame: pd.DataFrame | None,
    accuracy: dict | None,
    sweep_refusal: dict | None,
) -> list[str]:
    lines = [
        "## Holding-rate sweep",
        "",
        HOLDING_RATE_SWEEP_RULE,
        "",
    ]
    if sweep_refusal is not None:
        lines.extend([
            "The holding-rate sweep was refused.",
            f"sweep_refusal {json.dumps(sweep_refusal, sort_keys=True)}",
            "",
            "This draft has no sweep means.",
            "",
        ])
        return lines
    if frame is None:
        lines.append(
            "`synthetic/holding_rate_sweep.parquet` is not in this run, "
            "so this draft has no sweep numbers."
        )
        lines.append("")
        return lines
    lines.extend(_beside(accuracy))
    lines.append("")
    keyed = frame["holding_rate"].map(lambda value: round(float(value), 6))
    rates = sorted(set(keyed))
    lines.append(
        "sweep_rates (" + ", ".join(f"{rate:g}" for rate in rates) + ")"
    )
    lines.append("")
    for rate in rates:
        group = frame[keyed == rate]
        lines.append(f"### holding_rate {rate:g}")
        lines.append("")
        lines.extend(_means(group))
        lines.append("")
        if {"latent", "ceiling"} <= set(group.columns):
            low, high = paired_interval(group["latent"], group["ceiling"])
            lines.append(
                f"paired_interval {_format_decimal(low)} {_format_decimal(high)}"
            )
            lines.append("")
    return lines


def _rollout_section(accuracy: dict | None) -> list[str]:
    lines = ["## Rollout accuracy", ""]
    if accuracy is None:
        lines.append(
            "This run has no `synthetic/rollout_accuracy.json`, so this draft "
            "reports no rollout R²."
        )
        return lines
    lines.extend(_rollout_means(accuracy))
    lines.append("")
    lines.extend(_rollout_steps(accuracy))
    return lines


def _warnings(synthetic: Path) -> list[str]:
    path = synthetic / "comparison_warnings.json"
    if not path.is_file():
        return []
    return [
        "## Comparison warnings",
        "",
        "Copied from `synthetic/comparison_warnings.json`.",
        "",
        path.read_text(encoding="utf-8").rstrip("\n"),
        "",
    ]


def _run_record(
    run: Path,
    frames: dict[str, pd.DataFrame | None],
) -> list[str]:
    """Counts and study fields read from the run. Missing files are omitted."""
    lines = [
        "## Run record",
        "",
        "Training-data parameters are copied from `log/transitions.manifest.json`.",
        "Episode counts below the manifest are copied from the transition log.",
        "Planner fields are copied from `checkpoints/planner.json`.",
        "Each study line is copied from that directory's `study.json`.",
        "`best_value` is that file's search objective on the search seeds.",
        "It is not a holdout mean. The transition log omits the final tick of",
        "each episode, because that tick has no next state.",
        "",
    ]
    manifest = _load_json(run / "log" / "transitions.manifest.json")
    if manifest is None:
        lines.append("This run has no `log/transitions.manifest.json`.")
    else:
        lines.extend(_manifest_lines(manifest))
    lines.append("")
    log = run / "log" / "transitions.parquet"
    if log.is_file():
        frame = pd.read_parquet(log)
        for behavior, group in frame.groupby("behavior", sort=True):
            episodes = int(group["episode_seed"].nunique())
            lines.append(f"{behavior}_episodes {episodes}")
        lines.append(f"log_seed_min {int(frame['episode_seed'].min())}")
        lines.append(f"log_seed_max {int(frame['episode_seed'].max())}")
        lines.append(f"products_per_episode {_per_episode_counts(frame, 'product_id')}")
        lines.append(f"recorded_tick_min {int(frame['tick'].min())}")
        lines.append(f"recorded_tick_max {int(frame['tick'].max())}")
        lines.append(f"recorded_tick_count {_per_episode_counts(frame, 'tick')}")
    planner = run / "checkpoints" / "planner.json"
    if planner.is_file():
        meta = json.loads(planner.read_text(encoding="utf-8"))
        lines.append(f"delivery_lag {int(meta['delivery_lag'])}")
        lines.append(f"budget {int(meta['budget'])}")
        lines.append(f"fit_holding_rate {_format_decimal(meta['holding_rate'])}")
        lines.append(f"order_fee {_format_decimal(meta['order_fee'])}")
    tuning = run / "tuning"
    if tuning.is_dir():
        for path in sorted(tuning.glob("*/study.json")):
            meta = json.loads(path.read_text(encoding="utf-8"))
            name = path.parent.name
            recorded = (meta.get("episode_distribution") or {}).get("holding_rate")
            if recorded is not None:
                lines.append(f"study {name} holding_rate {json.dumps(recorded)}")
            for key in (
                "n_trials",
                "n_search_seeds",
                "n_holdout_seeds",
                "episode_length",
                "seed_offset",
                "holdout_seed_offset",
            ):
                if key in meta:
                    lines.append(f"study {name} {key} {int(meta[key])}")
            if "best_value" in meta:
                lines.append(f"study {name} best_value {json.dumps(meta['best_value'])}")
            if "git_sha" in meta:
                lines.append(f"study {name} git_sha {meta['git_sha']}")
            if "best_params" in meta:
                params = json.dumps(meta["best_params"], sort_keys=True)
                lines.append(f"study {name} best_params {params}")
    for name, frame in frames.items():
        if frame is not None and "seed" in frame.columns:
            lines.extend(_seed_span(name, frame))
    lines.append(
        "Means, intervals, and rollout R² below are printed to 6 decimal places."
    )
    return lines


def _manifest_lines(manifest: dict) -> list[str]:
    lines = ["Training-data generation parameters:"]
    for key in _MANIFEST_FIELDS:
        if key in manifest:
            lines.append(f"manifest {key} {json.dumps(manifest[key], sort_keys=True)}")
    for key in _MANIFEST_OBJECTS:
        if key in manifest:
            lines.append(f"manifest {key} {json.dumps(manifest[key], sort_keys=True)}")
    for arm in manifest.get("behavior_mixture") or []:
        lines.append(
            "manifest mixture "
            f"{arm['behavior']} episodes {int(arm['episodes'])} policy {arm['policy']}"
        )
    lines.append(
        "Per-episode mixture hyperparameters stay in the manifest sidecar "
        "and are not retyped here."
    )
    return lines


def _method() -> list[str]:
    return [
        "## Method",
        "",
        "The learned buyer is ordering only. `decide` sets `list_price` from",
        "`base_price`, emits no promotions, and sets the minimum-order floor to",
        "zero. Inventory position is on-hand plus in-transit. The order before",
        "the ration is `max(0, round(cover × rate × delivery_lag) − position)`",
        "on every tick. The rate is the textbook censored-sales estimator over",
        "`delivery_lag`. Covers are chosen from `{0, 1, 2, 4}`.",
        "",
        "One learned buyer sits on the intermediate that faces the sinks.",
        "Factories stay on `StaticFactoryPolicy`. There is no second learned",
        "echelon. The anchor and the planner share cheapest-first routing and",
        "the same `per_supplier_min_order_floor`.",
        "",
        "The world model is per SKU. It is conditioned on the quantity",
        "Allocation filled. The planner imagines that quantity as the grant",
        "times the log's fill rate, and the emitted order stays the full grant.",
        "The encoder sees on-hand, an arrival-age",
        "profile, a sales window of its own length rather than the delivery",
        "lead time, cash, capacity, and that lead time. The Market",
        "multiplier, the disruption identity, and the calendar are not",
        "features. An",
        "exponential-moving-average target encoder supplies the regression",
        "target. The predictor rolls its embedding forward for the planning",
        "horizon, `model_unroll_steps`, which defaults to",
        "`horizon_multiple * delivery_lag` and is not the delivery lead time.",
        "The latent loss is the distance to that stopped",
        "target at each step. An anti-collapse penalty keeps embedding variance",
        "off the floor and penalizes cross-dimension correlation. Probes decode",
        "next on-hand, the next arrival-age profile, and sales over the step.",
        "The planner sums that profile for inventory position. Probe losses",
        "update the probes only. The planner's profit calculation does not",
        "update the encoder. Imagined profit is revenue minus order cost minus",
        "holding cost minus order fees, which is the engine's `net_profit`,",
        "plus a terminal credit of ending inventory position at unit cost.",
        "That credit is not cash the next ration can spend.",
        "There is no stockout penalty and no learned reward head.",
        "",
        "Each tick scores `budget` cover proposals. That budget is the",
        "`budget` line in the run record. The search is factored",
        "cross-entropy over a per-SKU cover, seeded by `policy_seed`. Priority",
        "is the marginal imagined profit of one unit of shared capacity. A",
        "greedy fill grants units in that order until free capacity or cash is",
        "exhausted. `ration=\"permutation\"` draws one order from `policy_rng`",
        "and is the ablation. `ration=\"fair_share\"` keeps the cover search",
        "and splits with textbook fair-share. The winner is the proposal with",
        "the highest imagined `net_profit`. The same `policy_seed` places the",
        "same orders on a repeated run. Search randomness is `policy_seed`.",
        "`OrderUpToPolicy` keeps fair-share. Engine Allocation stays",
        "`execute_buy`. The direct predictor uses the same search and the same",
        "ration, with no latent and no EMA target.",
        "",
        "The training log mixes all four textbook families, with hyperparameters",
        "sampled from the tuning search spaces, plus the exploratory policy and",
        "never-order and over-order arms. Planning depth varies across episodes.",
        "The log is collected at a single holding rate, on the training",
        "partition of the catalog. Untuned textbook policies",
        "stay at published defaults on the holdout tables. The in-distribution",
        "protocol uses the tuner holdout at `holding_rate = 0.01`, the training",
        "SKUs, and the default disruption. The anchor is an",
        "`OrderUpToPolicy` tuned at that rate. Unseen SKU uses the evaluation",
        "partition and its own seed block. Unseen disruption uses the training",
        "SKUs and a harsher disruption regime the log never samples.",
        "The holding-rate sweep keeps the predictor",
        "trained at `holding_rate = 0.01`. Imagined cash is rolled forward at",
        "each sweep rate and fed back into the ration. The weights are",
        "not refit, and behavior data is not recollected. At each rate the",
        "ceiling is the `OrderUpToPolicy` tuned at that same rate. The sweep",
        "writes no curve when the frozen planner's order sequence is identical",
        "across those rates.",
        "",
        "Every comparison table carries a never-order floor, a random-cover",
        "floor, and a least-squares model in the planner, plus an oracle model",
        "in the planner as a ceiling. Order volume, service level, stockout",
        "rate, and realised fill rate are reported on every arm. Held-out",
        "rollout R² for the latent model, the direct predictor, and ordinary",
        "least squares is printed beside those profit means.",
        "",
        "## Caveats",
        "",
        "Sales are censored. The censored-sales rate treats a stockout as low",
        "demand, so a cover learned from that rate is not a cover of true",
        "demand. Lost sales appear only as revenue that did not arrive.",
        "",
        "On the M5 arm, `unit_cost` is 40% of median price. That figure is the",
        "adapter's documented fiction, not a Walmart cost. Both ordering",
        "policies would face the same fiction, inside one `PriceReplayPolicy`,",
        "under `flat_world`.",
        "",
        "## Success rules",
        "",
        IN_DISTRIBUTION_RULE,
        "",
        HOLDING_RATE_SWEEP_RULE,
        "",
        "The textbook policy tuned at each sweep rate is reported and is not a",
        "pass bar. An M5 row is not a gate. A profit table is not written when",
        "held-out latent rollout R² fails to beat ordinary least squares on the",
        "same inputs.",
        "",
        "## Where this draft does not belong",
        "",
        "The claim worth writing down is whether the frozen planner's orders",
        "change across holding rates, and the split M5 result, against a",
        "textbook policy tuned at the same rate, with the direct predictor and",
        "the fair-share arm in the table. When the sweep refuses because those",
        "orders are identical, that refusal is the result for that claim.",
        "Venues that want a structural inventory proof, and venues that want a",
        "PPO or Dreamer bake-off, are the wrong homes for this draft.",
    ]


def _m5(run: Path) -> list[str]:
    from src.jepa.m5 import SLICE_RULE

    exam = run / "m5" / "exam.parquet"
    lines = [
        "## M5",
        "",
        "The published command is:",
        "",
        "```bash",
        M5_COMMAND,
        "```",
        "",
        SLICE_RULE,
        "",
        "The graph is one `StaticFactoryPolicy` factory feeding `shop-CA_1`",
        "at lead time 3, and one replay sink per item. `flat_world` is on.",
        "Both the learned policy and `OrderUpToPolicy` are wrapped in the same",
        "`PriceReplayPolicy`. Zero-shot scores the synthetic weights and does",
        "not fit on either window. Adaptation refits the same architecture on",
        "reward-free transitions from the 2015 window only. The test window is",
        "2016. The windows are not pooled. The textbook anchor is tuned on 2015",
        "replay profit only. Sales are censored, and `unit_cost` is a fiction:",
        "40% of median price. Neither row is a pass/fail gate.",
        "",
    ]
    if not exam.is_file():
        lines.extend(
            [
                "PINNED_ITEM_IDS is empty until a quality report runs.",
                "",
                "This run has no `m5/exam.parquet`, so there is no zero-shot row",
                "and no adaptation row.",
            ]
        )
        return lines
    frame = pd.read_parquet(exam)
    for label in ("zero_shot", "adaptation"):
        matched = frame[frame["label"] == label]
        if matched.empty:
            continue
        row = matched.iloc[0]
        lines.append(f"### {label}")
        lines.append("")
        for column in row.index:
            if column == "label":
                continue
            value = row[column]
            if isinstance(value, float):
                rendered = _format_decimal(value)
            else:
                rendered = str(value)
            lines.append(f"{column} {rendered}")
        lines.append("")
    return lines


def _future() -> list[str]:
    return [
        "## Future work",
        "",
        "Future work, deferred rather than forgotten:",
        "",
        "- pricing policies, with ordering held fixed",
        "- a joint multi-SKU model",
        "- policy control of engine Allocation",
    ]
