"""Draft the replenishment paper from a run directory.

Result cells are means, the paired interval, and the pass words. They
are read from the synthetic tables or computed by the comparison
functions. This module does not run a study and does not invent a cell.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.jepa.commands import M5_COMMAND
from src.jepa.eval import (
    IN_DISTRIBUTION_RULE,
    RETARGET_RULE,
    in_distribution_passes,
    paired_interval,
    retarget_passes,
)


def _format_decimal(value: float) -> str:
    """Six decimal places, the same rounding for every table cell."""
    return f"{float(value):.6f}"


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
        if column == "seed":
            continue
        lines.append(f"| {column} | {_format_decimal(frame[column].mean())} |")
    return lines


def render_paper(run_dir: str | Path) -> str:
    """Return the draft for ``run_dir``. Requires the synthetic tables."""
    run = Path(run_dir)
    in_dist = pd.read_parquet(run / "synthetic" / "in_distribution.parquet")
    retarget = pd.read_parquet(run / "synthetic" / "retarget.parquet")
    low, high = paired_interval(in_dist["latent"], in_dist["anchor"])
    in_word = "pass" if in_distribution_passes(in_dist["latent"], in_dist["anchor"]) else "fail"
    retarget_word = "pass" if retarget_passes(retarget["latent"], retarget["anchor"]) else "fail"
    lines = [
        "# Latent replenishment",
        "",
        *_method(),
        "",
        *_run_record(run, in_dist, retarget),
        "",
        "## In-distribution",
        "",
        IN_DISTRIBUTION_RULE,
        "",
        "The in-distribution claim is that rule.",
        "",
        "Per-seed rows stay in `synthetic/in_distribution.parquet`.",
        "The `anchor` column is the tuned `OrderUpToPolicy` when that study",
        "was passed to eval. `order_up_to` is the published-default policy.",
        "",
        *_means(in_dist),
        "",
        f"paired_interval {_format_decimal(low)} {_format_decimal(high)}",
        f"in_distribution_passes {in_word}",
        "",
        "## Retarget",
        "",
        RETARGET_RULE,
        "",
        "Per-seed rows stay in `synthetic/retarget.parquet`.",
        "`ceiling` is the `OrderUpToPolicy` tuned at holding rate 0.05.",
        "It is reported and it is not a pass bar.",
        "",
        *_means(retarget),
        "",
        f"retarget_passes {retarget_word}",
        "",
        *_m5(run),
        "",
        *_future(),
    ]
    text = "\n".join(lines)
    if not text.endswith("\n"):
        text += "\n"
    return text


def _run_record(
    run: Path,
    in_dist: pd.DataFrame,
    retarget: pd.DataFrame,
) -> list[str]:
    """Counts and study fields read from the run. Missing files are omitted."""
    lines = [
        "## Run record",
        "",
        "These lines are copied from the log, `checkpoints/planner.json`, and",
        "each `study.json`. `best_value` is that file's search objective on the",
        "search seeds. It is not the holdout table below. The transition log",
        "omits the final tick of each episode, because that tick has no next",
        "state. `episode_length` is the length stored on the study.",
        "",
    ]
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
    lines.extend(_seed_span("in_distribution", in_dist))
    lines.extend(_seed_span("retarget", retarget))
    lines.append(
        "Means and the paired interval below are printed to 6 decimal places."
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
        "echelon.",
        "",
        "The world model is per SKU. It is conditioned on the quantity the",
        "ration actually granted. The encoder sees on-hand, an arrival-age",
        "profile, a sales window of its own length rather than the delivery",
        "lead time, cash, capacity, and that lead time. The Market",
        "multiplier, the disruption identity, and the calendar are not",
        "features. An",
        "exponential-moving-average target encoder supplies the regression",
        "target. The predictor rolls its embedding forward for the edge lead",
        "time, open-loop, and the latent loss is the distance to that stopped",
        "target at each step. An anti-collapse penalty keeps embedding variance",
        "off the floor and penalizes cross-dimension correlation. Probes decode",
        "next on-hand, the next arrival-age profile, and sales over the step.",
        "The planner sums that profile for inventory position. Probe losses",
        "update the probes only. The planner's profit calculation does not",
        "update the encoder. Imagined profit is revenue minus order cost minus",
        "holding cost minus order fees, which is the engine's `net_profit`.",
        "There is no stockout penalty and no learned reward head.",
        "",
        "Each tick draws `budget` proposals from `policy_rng`, seeded by",
        "`policy_seed`. That budget is the `budget` line in the run record.",
        "Each proposal is one cover per SKU and one priority permutation. A",
        "greedy fill grants units in that order until free capacity or cash is",
        "exhausted, and the winner is the proposal with the highest imagined",
        "`net_profit`. The same `policy_seed` places the same orders on a",
        "repeated run. The world seed is not the search seed.",
        "`OrderUpToPolicy` keeps fair-share. Engine Allocation stays",
        "`execute_buy`. The direct predictor uses the same shooter and the same",
        "ration, with no latent and no EMA target. A fair-share arm takes the",
        "latent model's covers and rations them with textbook fair-share.",
        "",
        "The training log mixes all four textbook families, with hyperparameters",
        "sampled from the tuning search spaces, plus the exploratory policy and",
        "never-order and over-order arms. Planning depth varies across episodes.",
        "The log is collected at a single holding rate. Untuned textbook policies",
        "stay at published defaults on the holdout table. The in-distribution protocol uses",
        "the tuner holdout at `holding_rate = 0.01`. The anchor is an",
        "`OrderUpToPolicy` tuned at that rate. The retarget keeps the predictor",
        "trained at `holding_rate = 0.01`. Imagined cash is rolled forward at",
        "`holding_rate = 0.05` and fed back into the ration. The weights are",
        "not refit, and behavior data is not recollected. An `OrderUpToPolicy`",
        "tuned at `holding_rate = 0.05` is reported as a ceiling.",
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
        RETARGET_RULE,
        "",
        "The retuned textbook policy is reported and is not a pass bar. An M5",
        "row is not a gate.",
        "",
        "## Where this draft does not belong",
        "",
        "The claim worth writing down is the frozen holding-rate retarget and",
        "the split M5 result, against a tuned textbook anchor, with the direct",
        "predictor and the fair-share arm in the table. Venues that want a",
        "structural inventory proof, and venues that want a PPO or Dreamer",
        "bake-off, are the wrong homes for this draft.",
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
        "- a harsher held-out disruption, kept out of the training episodes",
    ]
