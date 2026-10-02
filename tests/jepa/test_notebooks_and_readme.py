"""Notebooks and README commands for the latent replenishment study.

Seams, from the issue acceptance criteria:

1. Training walkthrough. ``notebooks/09-jepa-train-and-plan.ipynb``, run
   top to bottom on its smoke settings, builds a small log, fits, and
   prints one decision's covers, ration, and order.
2. Results beside the rules. When ``runs/jepa-replenishment/synthetic/``
   is present, ``notebooks/10-jepa-synthetic-eval.ipynb`` states the
   in-distribution rule and the holding-rate sweep rule next to the loaded
   tables and draws a bar or line chart for each. Rollout, capacity, and
   data-scaling files add their own curves.
3. Missing artifacts. When that directory is absent, the results notebook
   prints the published command that fills it and does not raise.
4. Root README. Exact commands for the published log (base count 64, with
   over-order weighted higher), the fit, the 150-trial ``OrderUpToPolicy``
   studies at each holding rate the transfer sweep uses (0.005, 0.01, 0.02,
   0.05, 0.1), and the synthetic eval on the tuner holdout. Raw outputs stay
   under ``runs/jepa-replenishment/`` and are not committed.
5. Package README. ``src/jepa/README.md`` points at that root section.

Assumption: the results notebook reads ``JEPA_SYNTHETIC_DIR`` when set,
otherwise ``runs/jepa-replenishment/synthetic``. Missing parquet counts as
absent, so an empty directory prints the command instead of a zero table.
The published commands are ``python -m src.jepa`` subcommands. The tuner
CLI cannot set ``holding_rate``, so the two studies are ``src.jepa tune``.
Smoke settings stay in the training notebook and do not launch the
published counts. Story 59's M5 command is the existing notebook execute;
that notebook is not rewritten.
"""

from __future__ import annotations

import ast
import io
import os
from contextlib import redirect_stdout
from pathlib import Path

import nbformat
import pytest

TRAINING = Path("notebooks/09-jepa-train-and-plan.ipynb")
RESULTS = Path("notebooks/10-jepa-synthetic-eval.ipynb")


def _run_notebook(path: Path) -> str:
    """Execute code cells in-process. Magics are not used."""
    notebook = nbformat.read(path, as_version=4)
    namespace: dict = {"__name__": "__main__"}
    stdout = io.StringIO()
    previous = Path.cwd()
    try:
        with redirect_stdout(stdout):
            for cell in notebook.cells:
                if cell.cell_type != "code":
                    continue
                source = cell.source if isinstance(cell.source, str) else "".join(cell.source)
                exec(compile(source, str(path), "exec"), namespace)
    finally:
        os.chdir(previous)
    return stdout.getvalue()


def _printed(output: str, label: str) -> str:
    prefix = label + " "
    for line in output.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):]
    raise AssertionError(f"{label} was not printed:\n{output}")


def test_training_notebook_shows_one_decision_on_smoke_settings():
    output = _run_notebook(TRAINING)
    assert int(_printed(output, "log_rows")) > 0
    assert "order_up_to" in output
    assert "exploratory" in output
    covers = ast.literal_eval(_printed(output, "covers"))
    ration = ast.literal_eval(_printed(output, "ration"))
    order = ast.literal_eval(_printed(output, "order"))
    assert covers
    assert set(covers.values()) <= {0, 1, 2, 4}
    assert set(ration) == set(covers)
    assert isinstance(order, dict)
    from src.jepa.dynamics import SALES_WINDOW, feature_dim
    from src.jepa.plan import DEFAULT_HORIZON_MULTIPLE

    assert _printed(output, "search") == "cem"
    assert int(_printed(output, "sales_window")) == SALES_WINDOW
    assert int(_printed(output, "feature_dim")) == feature_dim()
    assert int(_printed(output, "model_unroll_steps")) == DEFAULT_HORIZON_MULTIPLE * 2
    notebook = nbformat.read(TRAINING, as_version=4)
    code = "\n".join(
        cell.source if isinstance(cell.source, str) else "".join(cell.source)
        for cell in notebook.cells
        if cell.cell_type == "code"
    )
    assert '["chosen"]' in code


def test_results_notebook_prints_the_published_command_when_tables_are_absent(tmp_path, monkeypatch):
    from src.jepa.commands import EVAL_COMMAND
    from src.jepa.eval import HOLDING_RATE_SWEEP_RULE

    missing = tmp_path / "no-such-synthetic"
    empty = tmp_path / "empty-synthetic"
    empty.mkdir()
    for directory in (missing, empty):
        monkeypatch.setenv("JEPA_SYNTHETIC_DIR", str(directory))
        output = _run_notebook(RESULTS)
        assert EVAL_COMMAND in output
        assert "in_distribution_passes" not in output
        assert "retarget_passes" not in output
        assert HOLDING_RATE_SWEEP_RULE not in output


def test_results_notebook_states_the_rules_beside_the_loaded_tables(tmp_path, monkeypatch):
    import pandas as pd

    from src.jepa.eval import (
        HOLDING_RATE_SWEEP_RULE,
        IN_DISTRIBUTION_RULE,
        in_distribution_passes,
    )

    in_latent = [0.123, 0.123]
    in_anchor = [0.812, 0.812]
    sweep_latent = [-0.456, -0.456]
    sweep_ceiling = [-0.901, -0.901]
    in_dist = pd.DataFrame({
        "seed": [13_000_000, 13_000_001],
        "latent": in_latent,
        "direct": [0.111, 0.111],
        "anchor": in_anchor,
    })
    in_dist.to_parquet(tmp_path / "in_distribution.parquet", index=False)
    pd.DataFrame({
        "holding_rate": [0.01, 0.01, 0.05, 0.05],
        "seed": [13_000_000, 13_000_001, 13_000_000, 13_000_001],
        "latent": sweep_latent + sweep_latent,
        "ceiling": sweep_ceiling + sweep_ceiling,
    }).to_parquet(tmp_path / "holding_rate_sweep.parquet", index=False)
    pd.DataFrame({
        "seed": [15_000_000, 15_000_001],
        "latent": [0.201, 0.201],
        "anchor": [0.050, 0.050],
    }).to_parquet(tmp_path / "unseen_sku.parquet", index=False)
    pd.DataFrame({
        "seed": [16_000_000, 16_000_001],
        "latent": [0.202, 0.202],
        "anchor": [0.040, 0.040],
    }).to_parquet(tmp_path / "unseen_disruption.parquet", index=False)
    monkeypatch.setenv("JEPA_SYNTHETIC_DIR", str(tmp_path))

    output = _run_notebook(RESULTS)
    in_at = output.index(IN_DISTRIBUTION_RULE)
    loaded_in = output.index("0.123")
    sweep_at = output.index(HOLDING_RATE_SWEEP_RULE)
    loaded_out = output.index("-0.456")
    assert in_at < loaded_in < sweep_at < loaded_out
    assert loaded_in < output.index("protocol unseen_sku") < output.index("protocol unseen_disruption") < sweep_at
    assert "0.201" in output and "0.202" in output
    assert "9.876" not in output
    assert f"in_distribution_passes {in_distribution_passes(in_latent, in_anchor)}" in output
    assert "sweep_rates (0.01, 0.05)" in output
    assert "retarget_passes" not in output
    assert int(_printed(output, "figures")) >= 4


def test_root_readme_publishes_the_commands_and_leaves_runs_uncommitted():
    from src.jepa.commands import (
        COLLECT_COMMAND,
        EVAL_COMMAND,
        FIT_COMMAND,
        M5_COMMAND,
        SWEEP_HOLDING_RATES,
        TUNE_COMMANDS,
    )

    readme = Path("README.md").read_text()
    section = readme.split("## Latent replenishment", 1)[1].split("\n## ", 1)[0]
    assert SWEEP_HOLDING_RATES == (0.005, 0.01, 0.02, 0.05, 0.1)
    for command in (COLLECT_COMMAND, FIT_COMMAND, EVAL_COMMAND, M5_COMMAND, *TUNE_COMMANDS):
        assert command in section
    assert "Over-order" in section
    assert "contributes 512" in section
    assert "one `holding_rate` of 0.01" in section
    assert "OrderUpToPolicy" in section
    assert "150" in section
    assert "0.01" in section and "0.05" in section
    assert "runs/jepa-replenishment/" in section
    assert "not committed" in section
    package = Path("src/jepa/README.md").read_text()
    assert "../../README.md#latent-replenishment" in package


def _study_dir(directory: Path) -> None:
    directory.mkdir(parents=True)
    (directory / "study.json").write_text(
        '{"best_params": {'
        '"cover_horizon_ticks": 5, '
        '"safety_lead_pct_of_lag": 0.5, '
        '"stockout_safety_bonus_pct_of_lag": 0.0, '
        '"per_supplier_min_order_floor": 0, '
        '"routing_strategy": "cheapest_first"'
        "}}",
        encoding="utf-8",
    )


def test_smoke_eval_does_not_publish_an_undertrained_profit_table(tmp_path, monkeypatch):
    """Smoke fit is one epoch on a short log. The rollout gate withholds the table."""
    import json

    from src.jepa.commands import SWEEP_HOLDING_RATES, main, order_up_to_study_name
    from src.jepa.eval import RolloutGateError

    log = tmp_path / "transitions.parquet"
    checkpoints = tmp_path / "checkpoints"
    tuning = tmp_path / "tuning"
    out = tmp_path / "synthetic"
    for rate in SWEEP_HOLDING_RATES:
        directory = tuning / order_up_to_study_name(rate)
        _study_dir(directory)
        path = directory / "study.json"
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta["episode_distribution"] = {"holding_rate": rate}
        path.write_text(json.dumps(meta), encoding="utf-8")
    main([
        "collect",
        "--episodes-each", "1",
        "--episode-length", "5",
        "--k-active", "2",
        "--delivery-lag", "2",
        "--output", str(log),
    ])
    main([
        "fit",
        "--log", str(log),
        "--output", str(checkpoints),
        "--epochs", "1",
        "--budget", "4",
        "--delivery-lag", "2",
    ])
    try:
        main([
            "eval",
            "--checkpoints", str(checkpoints),
            "--log", str(log),
            "--tuning-dir", str(tuning),
            "--output", str(out),
            "--n-seeds", "1",
            "--episode-length", "4",
            "--k-active", "2",
            "--delivery-lag", "2",
        ])
    except RolloutGateError:
        refused = True
    else:
        refused = False
    assert refused
    assert not (out / "in_distribution.parquet").exists()
    assert not (out / "unseen_sku.parquet").exists()
    assert not (out / "unseen_disruption.parquet").exists()
    assert not (out / "holding_rate_sweep.parquet").exists()
    assert not (out / "retarget.parquet").exists()
    assert (out / "rollout_accuracy.json").is_file()
    assert (out / "profit_table_refused.json").is_file()
    monkeypatch.setenv("JEPA_SYNTHETIC_DIR", str(out))
    output = _run_notebook(RESULTS)
    assert "in_distribution_passes" not in output
    assert "retarget_passes" not in output


def test_results_notebook_plots_floors_rollout_and_scaling_curves(tmp_path, monkeypatch):
    import json

    import pandas as pd

    pd.DataFrame(
        {
            "seed": [1, 2],
            "never_order": [0.01, 0.01],
            "random_cover": [0.02, 0.02],
            "least_squares": [0.03, 0.03],
            "oracle": [0.9, 0.9],
            "latent": [0.4, 0.4],
            "anchor": [0.2, 0.2],
        }
    ).to_parquet(tmp_path / "in_distribution.parquet", index=False)
    for name, seed in (("unseen_sku", 15_000_000), ("unseen_disruption", 16_000_000)):
        pd.DataFrame(
            {"seed": [seed, seed + 1], "latent": [0.2, 0.2], "oracle": [0.8, 0.8]}
        ).to_parquet(tmp_path / f"{name}.parquet", index=False)
    pd.DataFrame(
        {
            "holding_rate": [0.01, 0.05],
            "seed": [1, 1],
            "latent": [0.3, 0.1],
            "ceiling": [0.5, 0.4],
        }
    ).to_parquet(tmp_path / "holding_rate_sweep.parquet", index=False)
    (tmp_path / "rollout_accuracy.json").write_text(
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
    scaling = tmp_path / "scaling"
    scaling.mkdir()
    (scaling / "capacity.json").write_text(
        json.dumps(
            {
                "points": [
                    {"parameters": 10, "latent_mean": 0.1, "direct_mean": 0.2, "ols_mean": 0.3},
                    {"parameters": 20, "latent_mean": 0.2, "direct_mean": 0.3, "ols_mean": 0.3},
                ]
            }
        ),
        encoding="utf-8",
    )
    (scaling / "data.json").write_text(
        json.dumps(
            {
                "points": [
                    {"episodes": 4, "latent_mean": 0.0, "direct_mean": 0.2, "ols_mean": 0.3},
                    {"episodes": 8, "latent_mean": 0.1, "direct_mean": 0.2, "ols_mean": 0.3},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("JEPA_SYNTHETIC_DIR", str(tmp_path))
    monkeypatch.setenv("JEPA_SCALING_DIR", str(scaling))
    output = _run_notebook(RESULTS)
    assert "never_order 0.010" in output
    assert "oracle 0.900" in output
    assert "rollout_r2_latent_mean 0.300000" in output
    assert "capacity_points 2" in output
    assert "data_points 2" in output
    # Three protocol bars, the sweep, the rollout curve, capacity, and data.
    assert int(_printed(output, "figures")) >= 7


def test_sweep_loader_rejects_a_study_that_does_not_record_its_rate(tmp_path):
    import json

    from src.jepa.commands import order_up_to_study_name, sweep_factories

    for rate in (0.005, 0.01, 0.02, 0.05, 0.1):
        directory = tmp_path / order_up_to_study_name(rate)
        _study_dir(directory)
        if rate == 0.05:
            continue
        path = directory / "study.json"
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta["episode_distribution"] = {"holding_rate": rate}
        path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="does not record"):
        sweep_factories(tmp_path)


def test_sweep_loader_rejects_a_study_tuned_at_another_rate(tmp_path):
    import json

    from src.jepa.commands import order_up_to_study_name, sweep_factories

    for rate in (0.005, 0.01, 0.02, 0.05, 0.1):
        directory = tmp_path / order_up_to_study_name(rate)
        _study_dir(directory)
        path = directory / "study.json"
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta["episode_distribution"] = {"holding_rate": 0.01 if rate == 0.05 else rate}
        path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="0.05"):
        sweep_factories(tmp_path)


def test_sweep_study_names_keep_the_published_directories():
    from src.jepa.commands import SWEEP_HOLDING_RATES, order_up_to_study_name

    assert [order_up_to_study_name(rate) for rate in SWEEP_HOLDING_RATES] == [
        "order_up_to_holding_0_005",
        "order_up_to_holding_0_01",
        "order_up_to_holding_0_02",
        "order_up_to_holding_0_05",
        "order_up_to_holding_0_1",
    ]


def test_tune_command_writes_an_order_up_to_study_at_the_requested_rate(tmp_path):
    import json

    from src.jepa.commands import main

    main([
        "tune",
        "--holding-rate", "0.05",
        "--trials", "1",
        "--output-dir", str(tmp_path),
        "--episode-length", "3",
        "--k-active", "1",
        "--n-search-seeds", "1",
        "--n-holdout-seeds", "1",
    ])
    study_dir = tmp_path / "order_up_to_holding_0_05"
    meta = json.loads((study_dir / "study.json").read_text(encoding="utf-8"))
    assert meta["n_trials"] == 1
    assert meta["episode_length"] == 3
    assert (study_dir / "trials.parquet").is_file()
