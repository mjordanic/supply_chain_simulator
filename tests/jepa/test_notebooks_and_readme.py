"""Notebooks and README commands for the latent replenishment study.

Seams, from the issue acceptance criteria:

1. Training walkthrough. ``notebooks/09-jepa-train-and-plan.ipynb``, run
   top to bottom on its smoke settings, builds a small log, fits, and
   prints one decision's covers, ration, and order.
2. Results beside the rules. When ``runs/jepa-replenishment/synthetic/``
   is present, ``notebooks/10-jepa-synthetic-eval.ipynb`` states the
   in-distribution rule and the retarget rule next to the loaded tables.
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
    notebook = nbformat.read(TRAINING, as_version=4)
    code = "\n".join(
        cell.source if isinstance(cell.source, str) else "".join(cell.source)
        for cell in notebook.cells
        if cell.cell_type == "code"
    )
    assert '["chosen"]' in code


def test_results_notebook_prints_the_published_command_when_tables_are_absent(tmp_path, monkeypatch):
    from src.jepa.commands import EVAL_COMMAND

    missing = tmp_path / "no-such-synthetic"
    empty = tmp_path / "empty-synthetic"
    empty.mkdir()
    for directory in (missing, empty):
        monkeypatch.setenv("JEPA_SYNTHETIC_DIR", str(directory))
        output = _run_notebook(RESULTS)
        assert EVAL_COMMAND in output
        assert "in_distribution_passes" not in output
        assert "retarget_passes" not in output


def test_results_notebook_states_the_rules_beside_the_loaded_tables(tmp_path, monkeypatch):
    import pandas as pd

    from src.jepa.eval import (
        IN_DISTRIBUTION_RULE,
        RETARGET_RULE,
        in_distribution_passes,
        retarget_passes,
    )

    in_latent = [0.123, 0.123]
    in_anchor = [0.812, 0.812]
    retarget_latent = [-0.456, -0.456]
    retarget_anchor = [-0.901, -0.901]
    in_dist = pd.DataFrame({
        "seed": [13_000_000, 13_000_001],
        "latent": in_latent,
        "direct": [0.111, 0.111],
        "anchor": in_anchor,
    })
    retarget = pd.DataFrame({
        "seed": [13_000_000, 13_000_001],
        "latent": retarget_latent,
        "anchor": retarget_anchor,
        "ceiling": [0.222, 0.222],
    })
    in_dist.to_parquet(tmp_path / "in_distribution.parquet", index=False)
    retarget.to_parquet(tmp_path / "retarget.parquet", index=False)
    monkeypatch.setenv("JEPA_SYNTHETIC_DIR", str(tmp_path))

    output = _run_notebook(RESULTS)
    in_at = output.index(IN_DISTRIBUTION_RULE)
    loaded_in = output.index("0.123")
    retarget_at = output.index(RETARGET_RULE)
    loaded_out = output.index("-0.456")
    assert in_at < loaded_in < retarget_at < loaded_out
    assert "9.876" not in output
    assert f"in_distribution_passes {in_distribution_passes(in_latent, in_anchor)}" in output
    assert f"retarget_passes {retarget_passes(retarget_latent, retarget_anchor)}" in output


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


def test_smoke_commands_write_tables_the_results_notebook_reads(tmp_path, monkeypatch):
    """The published subcommands, on smoke overrides, fill the notebook's files.

    Profit figures are whatever the run produced. This test checks that the
    notebook can state both rules beside those files, and that the numbers
    are finite.
    """
    import math

    import pandas as pd

    from src.jepa.commands import main
    from src.jepa.eval import IN_DISTRIBUTION_RULE, RETARGET_RULE

    log = tmp_path / "transitions.parquet"
    checkpoints = tmp_path / "checkpoints"
    anchor = tmp_path / "order_up_to_holding_0_01"
    ceiling = tmp_path / "order_up_to_holding_0_05"
    out = tmp_path / "synthetic"
    _study_dir(anchor)
    _study_dir(ceiling)
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
    main([
        "eval",
        "--checkpoints", str(checkpoints),
        "--anchor", str(anchor),
        "--ceiling", str(ceiling),
        "--output", str(out),
        "--n-seeds", "1",
        "--episode-length", "4",
        "--k-active", "2",
        "--delivery-lag", "2",
    ])
    in_dist = pd.read_parquet(out / "in_distribution.parquet")
    retarget = pd.read_parquet(out / "retarget.parquet")
    assert {"seed", "latent", "anchor"} <= set(in_dist.columns)
    assert {"seed", "latent", "anchor", "ceiling"} <= set(retarget.columns)
    assert math.isfinite(float(in_dist["latent"].iloc[0]))
    assert math.isfinite(float(retarget["latent"].iloc[0]))
    monkeypatch.setenv("JEPA_SYNTHETIC_DIR", str(out))
    output = _run_notebook(RESULTS)
    assert output.index(IN_DISTRIBUTION_RULE) < output.index(RETARGET_RULE)
    assert "in_distribution_passes" in output
    assert "retarget_passes" in output


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
