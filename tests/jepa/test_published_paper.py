"""Paper draft whose result cells are copied from a run directory.

Seams, from the issue acceptance criteria:

1. Result cells. ``render_paper(run_dir)`` copies arm means from
   ``synthetic/in_distribution.parquet`` and ``synthetic/retarget.parquet``.
   The pass words are ``in_distribution_passes`` and ``retarget_passes``.
   A false boolean is the word ``fail``. A sentinel that is not in those
   tables is absent.
2. In-distribution claim. That section states the ticket-03 rule (the
   paired interval covers zero, or the latent mean is at least 95% of the
   anchor mean). It does not call that claim "beats the tuned anchor."
3. Method, caveats, and future work. The draft states ordering-only
   covers, the censored-sales rate, a per-SKU latent predictor, frozen
   weights from the 0.01 fit, the censored-sales caveat, the ``unit_cost``
   caveat, both success rules, and future work: pricing policies, a joint
   multi-SKU model, policy control of engine Allocation, and a harsher
   held-out disruption.
4. Missing M5. With no ``m5/exam.parquet``, the section contains the
   published notebook command and the ADR 0028 slice rule, and it has no
   exam profit. ``PINNED_ITEM_IDS`` stays empty.
5. Present M5. With ``m5/exam.parquet``, the zero-shot row and the
   adaptation row are copied and kept separate. Neither row is a gate.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _tables(directory: Path) -> None:
    synthetic = directory / "synthetic"
    synthetic.mkdir(parents=True)
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
            "seed": [13_000_000, 13_000_001],
            "latent": [-0.5, -0.5],
            "anchor": [-0.25, -0.25],
            "ceiling": [0.75, 0.75],
        }
    ).to_parquet(synthetic / "retarget.parquet", index=False)


def test_paper_copies_means_and_reports_a_failing_pass_boolean(tmp_path):
    from src.jepa.eval import in_distribution_passes, paired_interval, retarget_passes
    from src.jepa.paper import render_paper

    _tables(tmp_path)
    text = render_paper(tmp_path)
    in_dist = pd.read_parquet(tmp_path / "synthetic" / "in_distribution.parquet")
    retarget = pd.read_parquet(tmp_path / "synthetic" / "retarget.parquet")
    low, high = paired_interval(in_dist["latent"], in_dist["anchor"])

    assert in_distribution_passes(in_dist["latent"], in_dist["anchor"]) is False
    assert retarget_passes(retarget["latent"], retarget["anchor"]) is False
    for value in (
        float(in_dist["latent"].mean()),
        float(in_dist["direct"].mean()),
        float(in_dist["anchor"].mean()),
        float(retarget["latent"].mean()),
        float(retarget["ceiling"].mean()),
        low,
        high,
    ):
        assert f"{value:.6f}" in text
    assert "in_distribution_passes fail" in text
    assert "retarget_passes fail" in text
    assert "9.876543" not in text


def test_paper_reports_a_passing_boolean_when_the_rules_hold(tmp_path):
    synthetic = tmp_path / "synthetic"
    synthetic.mkdir(parents=True)
    pd.DataFrame(
        {
            "seed": [13_000_000, 13_000_001],
            "latent": [0.5, 0.5],
            "anchor": [0.5, 0.5],
        }
    ).to_parquet(synthetic / "in_distribution.parquet", index=False)
    pd.DataFrame(
        {
            "seed": [13_000_000, 13_000_001],
            "latent": [0.4, 0.4],
            "anchor": [0.2, 0.2],
        }
    ).to_parquet(synthetic / "retarget.parquet", index=False)

    from src.jepa.paper import render_paper

    text = render_paper(tmp_path)
    assert "in_distribution_passes pass" in text
    assert "retarget_passes pass" in text


def test_in_distribution_claim_is_the_interval_rule_not_a_win(tmp_path):
    from src.jepa.eval import IN_DISTRIBUTION_RULE
    from src.jepa.paper import render_paper

    _tables(tmp_path)
    text = render_paper(tmp_path)
    section = text.split("## In-distribution", 1)[1].split("## Retarget", 1)[0]
    assert IN_DISTRIBUTION_RULE in section
    assert "beats the tuned anchor" not in section


def test_paper_states_method_caveats_rules_and_future_work(tmp_path):
    from src.jepa.eval import IN_DISTRIBUTION_RULE, RETARGET_RULE
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
    assert IN_DISTRIBUTION_RULE in text
    assert RETARGET_RULE in text
    for phrase in (
        "pricing policies",
        "joint multi-SKU model",
        "policy control of engine Allocation",
        "harsher held-out disruption",
    ):
        assert phrase in text


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
    import json

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
            }
        ),
        encoding="utf-8",
    )
    text = render_paper(tmp_path)
    record = text.split("## Run record", 1)[1].split("## In-distribution", 1)[0]
    assert "order_up_to_episodes 1" in record
    assert "exploratory_episodes 1" in record
    assert "log_seed_min 14000000" in record
    assert "budget 4" in record
    assert "delivery_lag 2" in record
    assert "n_trials 7" in record
    assert "episode_length 9" in record
    assert "0.424242" in record
    assert "cover_horizon_ticks" in record
    assert "9.876543" not in record


def test_tracked_paper_matches_the_renderer_and_keeps_a_failing_result():
    """The committed draft is the renderer output of the published run.

    CI does not have ``runs/``. When that directory is present, the file
    must match ``render_paper`` exactly, so a hand-edited cell fails.
    The fail words are the published outcome and stay fail.
    """
    from src.jepa.commands import M5_COMMAND
    from src.jepa.eval import IN_DISTRIBUTION_RULE, RETARGET_RULE
    from src.jepa.m5 import SLICE_RULE
    from src.jepa.paper import render_paper

    paper = Path("docs/jepa-replenishment.md").read_text(encoding="utf-8")
    run = Path("runs/jepa-replenishment")
    if (run / "synthetic" / "in_distribution.parquet").is_file():
        assert paper == render_paper(run)
    section = paper.split("## In-distribution", 1)[1].split("## Retarget", 1)[0]
    assert IN_DISTRIBUTION_RULE in section
    assert "beats the tuned anchor" not in section
    assert "in_distribution_passes fail" in section
    assert RETARGET_RULE in paper
    assert "retarget_passes fail" in paper
    assert "censored-sales" in paper
    assert "unit_cost" in paper
    assert M5_COMMAND in paper
    assert SLICE_RULE in paper
    assert "PINNED_ITEM_IDS is empty" in paper
    assert "### zero_shot" not in paper
    assert "### adaptation" not in paper
    for phrase in (
        "pricing policies",
        "joint multi-SKU model",
        "policy control of engine Allocation",
        "harsher held-out disruption",
    ):
        assert phrase in paper
