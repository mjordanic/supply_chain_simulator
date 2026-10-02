"""Paper draft whose result cells are copied from a run directory.

Seams, from the issue acceptance criteria:

1. Refused profit table. With ``profit_table_refused.json`` and
   ``rollout_accuracy.json``, and no protocol parquet, the draft says the
   table was refused, copies the rollout R², and has no profit mean.
2. Manifest. Training-data parameters in the run record are the manifest's
   own JSON values: episode count, seeds, length, ``K_active``, delivery
   lag, holding rate, order fee, catalog identity, mixture counts, git SHA.
3. Protocols and sweep. Means come from ``in_distribution.parquet``,
   ``unseen_sku.parquet``, ``unseen_disruption.parquet``, and
   ``holding_rate_sweep.parquet``. The pass word is
   ``in_distribution_passes``. Rollout means sit in each profit section.
   A harsher disruption is that protocol, not a future-work bullet. The
   sweep rule replaces the single-rate retarget. An identical-order refusal
   file contributes its reason and no sweep means.
4. Missing M5. With no ``m5/exam.parquet``, the section contains the
   published notebook command and the ADR 0028 slice rule, and it has no
   exam profit. ``PINNED_ITEM_IDS`` stays empty.
5. Present M5. With ``m5/exam.parquet``, the zero-shot row and the
   adaptation row are copied and kept separate. Neither row is a gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _rollout(directory: Path) -> None:
    synthetic = directory / "synthetic"
    synthetic.mkdir(parents=True, exist_ok=True)
    (synthetic / "rollout_accuracy.json").write_text(
        json.dumps(
            {
                "steps": [1, 2],
                "latent": [0.2, 0.4],
                "direct": [0.5, 0.5],
                "ols": [0.1, 0.3],
            }
        ),
        encoding="utf-8",
    )


def _tables(directory: Path) -> None:
    synthetic = directory / "synthetic"
    synthetic.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "seed": [13_000_000, 13_000_001],
            "latent": [0.125, 0.125],
            "direct": [0.25, 0.25],
            "anchor": [0.875, 0.875],
        }
    ).to_parquet(synthetic / "in_distribution.parquet", index=False)
    pd.DataFrame(
        {
            "seed": [15_000_000, 15_000_001],
            "latent": [0.201, 0.201],
            "anchor": [0.050, 0.050],
        }
    ).to_parquet(synthetic / "unseen_sku.parquet", index=False)
    pd.DataFrame(
        {
            "seed": [16_000_000, 16_000_001],
            "latent": [0.202, 0.202],
            "anchor": [0.040, 0.040],
        }
    ).to_parquet(synthetic / "unseen_disruption.parquet", index=False)
    pd.DataFrame(
        {
            "holding_rate": [0.05, 0.05],
            "seed": [13_000_000, 13_000_001],
            "latent": [-0.5, -0.5],
            "ceiling": [0.75, 0.75],
        }
    ).to_parquet(synthetic / "holding_rate_sweep.parquet", index=False)


def test_refused_profit_table_reports_rollout_accuracy_and_no_profit_numbers(tmp_path):
    from src.jepa.paper import render_paper

    _rollout(tmp_path)
    refusal = {
        "reason": "held-out rollout accuracy did not beat the ordinary-least-squares baseline",
        "refused": True,
    }
    (tmp_path / "synthetic" / "profit_table_refused.json").write_text(
        json.dumps(refusal),
        encoding="utf-8",
    )
    text = render_paper(tmp_path)
    assert "The profit table was refused." in text
    assert json.dumps(refusal, sort_keys=True) in text
    assert "rollout_r2_latent_mean 0.300000" in text
    assert "rollout_r2_direct_mean 0.500000" in text
    assert "rollout_r2_ols_mean 0.200000" in text
    assert "step 1 latent 0.200000 direct 0.500000 ols 0.100000" in text
    assert "| arm | mean net_profit / initial_cash |" not in text
    assert "in_distribution_passes" not in text
    assert "9.876543" not in text


def test_manifest_parameters_are_copied_into_the_run_record(tmp_path):
    from src.jepa.paper import render_paper

    _tables(tmp_path)
    manifest = {
        "episode_count": 896,
        "seed_min": 14_000_000,
        "seed_max": 14_000_895,
        "episode_length": 365,
        "K_active": None,
        "K_min": 1,
        "K_max_episode": 30,
        "delivery_lag": None,
        "holding_rate": 0.01,
        "order_fee": 50.0,
        "catalog_identity": "cafebabe",
        "git_sha": "abc123dead",
        "behavior_mixture": [
            {"behavior": "over_order", "episodes": 512, "policy": "OverOrderPolicy"},
        ],
        "protocols": {"unseen_disruption": {"seed_min": 16_000_000, "n_seeds": 32}},
    }
    log = tmp_path / "log"
    log.mkdir()
    (log / "transitions.manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    text = render_paper(tmp_path)
    record = text.split("## Run record", 1)[1].split("## Rollout accuracy", 1)[0]
    for key in (
        "episode_count",
        "seed_min",
        "seed_max",
        "episode_length",
        "K_active",
        "delivery_lag",
        "holding_rate",
        "order_fee",
        "catalog_identity",
        "git_sha",
    ):
        assert f"manifest {key} {json.dumps(manifest[key])}" in record
    assert "manifest mixture over_order episodes 512 policy OverOrderPolicy" in record
    assert "abc123dead" in record
    assert "cafebabe" in record
    assert "9.876543" not in record


def test_paper_copies_protocol_means_rollout_and_a_failing_pass_boolean(tmp_path):
    from src.jepa.eval import in_distribution_passes, paired_interval
    from src.jepa.paper import render_paper

    _tables(tmp_path)
    _rollout(tmp_path)
    text = render_paper(tmp_path)
    in_dist = pd.read_parquet(tmp_path / "synthetic" / "in_distribution.parquet")
    sweep = pd.read_parquet(tmp_path / "synthetic" / "holding_rate_sweep.parquet")
    low, high = paired_interval(in_dist["latent"], in_dist["anchor"])

    assert in_distribution_passes(in_dist["latent"], in_dist["anchor"]) is False
    for value in (
        float(in_dist["latent"].mean()),
        float(in_dist["direct"].mean()),
        float(in_dist["anchor"].mean()),
        float(sweep["latent"].mean()),
        float(sweep["ceiling"].mean()),
        low,
        high,
    ):
        assert f"{value:.6f}" in text
    assert "in_distribution_passes fail" in text
    assert "retarget_passes" not in text
    assert "9.876543" not in text
    bounds = (
        ("## In-distribution", "## Unseen SKU"),
        ("## Unseen SKU", "## Unseen disruption"),
        ("## Unseen disruption", "## Holding-rate sweep"),
        ("## Holding-rate sweep", "## M5"),
    )
    for heading, nxt in bounds:
        section = text.split(heading, 1)[1].split(nxt, 1)[0]
        assert "rollout_r2_latent_mean 0.300000" in section
        assert "rollout_r2_ols_mean 0.200000" in section


def test_sweep_refusal_copies_the_reason_and_invents_no_curve(tmp_path):
    from src.jepa.paper import render_paper

    _tables(tmp_path)
    (tmp_path / "synthetic" / "holding_rate_sweep.parquet").unlink()
    payload = {
        "reason": "frozen planner order sequence is identical across holding rates",
        "refused": True,
    }
    (tmp_path / "synthetic" / "holding_rate_sweep_refused.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    text = render_paper(tmp_path)
    section = text.split("## Holding-rate sweep", 1)[1].split("## M5", 1)[0]
    assert "The holding-rate sweep was refused." in section
    assert json.dumps(payload, sort_keys=True) in section
    assert "sweep_rates" not in section
    assert "-0.500000" not in section
    assert "0.750000" not in section


def test_paper_reports_a_passing_boolean_when_the_rules_hold(tmp_path):
    synthetic = tmp_path / "synthetic"
    synthetic.mkdir(parents=True)
    pd.DataFrame(
        {
            "seed": [13_000_000, 13_000_001],
            "latent": [0.8, 0.8],
            "anchor": [0.2, 0.2],
        }
    ).to_parquet(synthetic / "in_distribution.parquet", index=False)
    pd.DataFrame(
        {"seed": [1, 2], "latent": [0.4, 0.4], "anchor": [0.2, 0.2]}
    ).to_parquet(synthetic / "unseen_sku.parquet", index=False)
    pd.DataFrame(
        {"seed": [1, 2], "latent": [0.4, 0.4], "anchor": [0.1, 0.1]}
    ).to_parquet(synthetic / "unseen_disruption.parquet", index=False)
    pd.DataFrame(
        {
            "holding_rate": [0.01, 0.01],
            "seed": [1, 2],
            "latent": [0.4, 0.4],
            "ceiling": [0.2, 0.2],
        }
    ).to_parquet(synthetic / "holding_rate_sweep.parquet", index=False)

    from src.jepa.paper import render_paper

    text = render_paper(tmp_path)
    assert "in_distribution_passes pass" in text
    assert "retarget_passes" not in text


def test_paper_means_keep_operational_columns_out_of_the_profit_table(tmp_path):
    from src.jepa.paper import render_paper

    synthetic = tmp_path / "synthetic"
    synthetic.mkdir(parents=True)
    pd.DataFrame(
        {
            "seed": [1, 2],
            "latent": [0.5, 0.5],
            "anchor": [0.2, 0.2],
            "latent_order_volume": [94.0, 94.0],
            "latent_service_level": [0.5, 0.5],
            "latent_stockout_rate": [0.1, 0.1],
            "latent_fill_rate": [0.8, 0.8],
            "tightness": [1.0, 1.0],
            "latent_minus_anchor": [0.3, 0.3],
            "rollout_r2_latent": [0.2, 0.2],
        }
    ).to_parquet(synthetic / "in_distribution.parquet", index=False)
    text = render_paper(tmp_path)
    section = text.split("## In-distribution", 1)[1].split("## Unseen SKU", 1)[0]
    assert "| latent | 0.500000 |" in section
    assert "| latent_order_volume |" not in section
    assert "| tightness |" not in section
    assert "| rollout_r2_latent |" not in section
    assert "| latent_minus_anchor |" not in section
    assert "| latent | 94.000000 | 0.500000 | 0.100000 | 0.800000 |" in section


def test_in_distribution_claim_requires_a_win_whose_interval_excludes_zero(tmp_path):
    from src.jepa.eval import IN_DISTRIBUTION_RULE
    from src.jepa.paper import render_paper

    _tables(tmp_path)
    text = render_paper(tmp_path)
    section = text.split("## In-distribution", 1)[1].split("## Unseen SKU", 1)[0]
    assert IN_DISTRIBUTION_RULE in section
    assert "excludes zero" in section
    assert "95%" not in section


def test_paper_states_method_protocols_and_future_work(tmp_path):
    from src.jepa.eval import HOLDING_RATE_SWEEP_RULE, IN_DISTRIBUTION_RULE
    from src.jepa.paper import render_paper

    _tables(tmp_path)
    text = render_paper(tmp_path)
    assert "ordering only" in text
    assert "censored-sales" in text
    assert "per SKU" in text
    assert "direct predictor" in text
    assert "fair-share" in text
    assert "published defaults" in text
    assert "holding_rate = 0.01" in text
    assert "not refit" in text
    assert "unit_cost" in text
    assert "Dreamer" in text
    assert "terminal credit" in text
    assert "cheapest-first" in text
    assert IN_DISTRIBUTION_RULE in text
    assert HOLDING_RATE_SWEEP_RULE in text
    assert "retarget_passes" not in text
    disruption = text.split("## Unseen disruption", 1)[1].split("## Holding-rate sweep", 1)[0]
    assert "harsher disruption" in disruption
    future = text.split("## Future work", 1)[1]
    for phrase in (
        "pricing policies",
        "joint multi-SKU model",
        "policy control of engine Allocation",
    ):
        assert phrase in future
    assert "harsher held-out disruption" not in future
    assert "harsher" not in future


def test_missing_m5_section_has_the_command_and_slice_rule_without_results(tmp_path):
    from src.jepa.commands import M5_COMMAND
    from src.jepa.m5 import SLICE_RULE
    from src.jepa.paper import render_paper

    _tables(tmp_path)
    text = render_paper(tmp_path)
    section = text.split("## M5", 1)[1]
    assert M5_COMMAND in section
    assert SLICE_RULE in section
    assert "PINNED_ITEM_IDS is empty" in section
    assert "### zero_shot" not in section
    assert "### adaptation" not in section
    assert "FOODS_3_" not in section


def test_m5_rows_are_copied_separately_when_the_exam_exists(tmp_path):
    from src.jepa.paper import render_paper

    _tables(tmp_path)
    exam = tmp_path / "m5"
    exam.mkdir()
    pd.DataFrame(
        [
            {
                "label": "zero_shot",
                "planner_net_profit": 111.0,
                "anchor_net_profit": 222.0,
                "planner_normalised": 0.111,
                "anchor_normalised": 0.222,
                "test_window_start": "2016-01-01",
                "test_window_end": "2016-04-24",
                "adaptation_window_start": "2015-01-01",
                "adaptation_window_end": "2015-12-31",
                "anchor_selection_profit": 333.0,
            },
            {
                "label": "adaptation",
                "planner_net_profit": 444.0,
                "anchor_net_profit": 222.0,
                "planner_normalised": 0.444,
                "anchor_normalised": 0.222,
                "test_window_start": "2016-01-01",
                "test_window_end": "2016-04-24",
                "adaptation_window_start": "2015-01-01",
                "adaptation_window_end": "2015-12-31",
                "anchor_selection_profit": 333.0,
            },
        ]
    ).to_parquet(exam / "exam.parquet", index=False)
    text = render_paper(tmp_path)
    zero = text.split("### zero_shot", 1)[1].split("### adaptation", 1)[0]
    adapted = text.split("### adaptation", 1)[1]
    assert "111.000000" in zero
    assert "444.000000" not in zero
    assert "444.000000" in adapted
    assert "111.000000" not in adapted
    assert "pass/fail gate" in text
    assert "in_distribution_passes" not in adapted


def test_run_record_copies_log_checkpoint_and_study_numbers(tmp_path):
    from src.jepa.paper import render_paper

    _tables(tmp_path)
    log = tmp_path / "log"
    log.mkdir()
    pd.DataFrame(
        {
            "episode_seed": [14_000_000, 14_000_000, 14_000_001],
            "behavior": ["order_up_to", "order_up_to", "exploratory"],
            "tick": [1, 2, 1],
            "product_id": ["a", "b", "a"],
        }
    ).to_parquet(log / "transitions.parquet", index=False)
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "planner.json").write_text(
        json.dumps(
            {
                "delivery_lag": 2,
                "budget": 4,
                "holding_rate": 0.01,
                "order_fee": 50.0,
            }
        ),
        encoding="utf-8",
    )
    study = tmp_path / "tuning" / "order_up_to_holding_0_01"
    study.mkdir(parents=True)
    (study / "study.json").write_text(
        json.dumps(
            {
                "n_trials": 7,
                "n_search_seeds": 3,
                "n_holdout_seeds": 4,
                "episode_length": 9,
                "holdout_seed_offset": 13_000_000,
                "best_value": 0.424242,
                "best_params": {"cover_horizon_ticks": 5},
                "episode_distribution": {"holding_rate": 0.01},
            }
        ),
        encoding="utf-8",
    )
    text = render_paper(tmp_path)
    record = text.split("## Run record", 1)[1].split("## Rollout accuracy", 1)[0]
    assert "order_up_to_episodes 1" in record
    assert "exploratory_episodes 1" in record
    assert "log_seed_min 14000000" in record
    assert "budget 4" in record
    assert "delivery_lag 2" in record
    assert "n_trials 7" in record
    assert "episode_length 9" in record
    assert "0.424242" in record
    assert "cover_horizon_ticks" in record
    assert "study order_up_to_holding_0_01 holding_rate 0.01" in record
    assert "9.876543" not in record


def _published_run_ready(run: Path) -> bool:
    """True once this ticket's eval has written a rollout file and a profit outcome."""
    if not (run / "log" / "transitions.manifest.json").is_file():
        return False
    synthetic = run / "synthetic"
    if not (synthetic / "rollout_accuracy.json").is_file():
        return False
    refused = (synthetic / "profit_table_refused.json").is_file()
    swept = (synthetic / "holding_rate_sweep_refused.json").is_file()
    tables = all(
        (synthetic / name).is_file()
        for name in (
            "in_distribution.parquet",
            "unseen_sku.parquet",
            "unseen_disruption.parquet",
            "holding_rate_sweep.parquet",
        )
    )
    return tables or refused or swept


def test_tracked_paper_matches_the_renderer_when_the_published_run_is_present():
    """The committed draft is the renderer output of the published run.

    CI does not have ``runs/``. When that directory has this ticket's
    manifest, rollout file, and either the protocol tables or a refusal,
    the file must match ``render_paper`` exactly.
    """
    from src.jepa.commands import M5_COMMAND
    from src.jepa.eval import HOLDING_RATE_SWEEP_RULE, IN_DISTRIBUTION_RULE
    from src.jepa.m5 import SLICE_RULE
    from src.jepa.paper import render_paper

    paper = Path("docs/jepa-replenishment.md").read_text(encoding="utf-8")
    run = Path("runs/jepa-replenishment")
    if _published_run_ready(run):
        assert paper == render_paper(run)
    assert IN_DISTRIBUTION_RULE in paper
    assert HOLDING_RATE_SWEEP_RULE in paper
    assert "retarget_passes" not in paper
    assert "harsher held-out disruption" not in paper
    assert "censored-sales" in paper
    assert "unit_cost" in paper
    assert M5_COMMAND in paper
    assert SLICE_RULE in paper
    assert "PINNED_ITEM_IDS is empty" in paper
    assert "### zero_shot" not in paper
    assert "### adaptation" not in paper
    future = paper.split("## Future work", 1)[1]
    for phrase in (
        "pricing policies",
        "joint multi-SKU model",
        "policy control of engine Allocation",
    ):
        assert phrase in future
