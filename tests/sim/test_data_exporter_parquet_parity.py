"""Issue 06: DataExporter parquet parity regression test.

Loads committed fixtures (products.parquet / stores.parquet) produced by the
pre-refactor DataExporter against the deterministic offline CannedClient
scenario, then re-runs the same scenario through the refactored DataExporter
and asserts byte-identical output via pd.testing.assert_frame_equal.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture(scope="module")
def _offline_run():
    """Run the offline scenario through DataExporter once per module."""
    import sys
    import tempfile

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from scenarios.example_llm_world_offline import scenario
    from src.sim.data_exporter import DataExporter
    from src.sim.runner import Runner

    run_log = Runner(scenario).run()
    tmp = tempfile.mkdtemp()
    DataExporter(scenario, run_log).export_all(tmp)
    return Path(tmp) / "data"


def test_products_parquet_matches_fixture(_offline_run):
    actual = pd.read_parquet(_offline_run / "products.parquet")
    fixture = pd.read_parquet(FIXTURES / "products.parquet")
    pd.testing.assert_frame_equal(actual, fixture, check_dtype=True, check_exact=True)


def test_stores_parquet_matches_fixture(_offline_run):
    actual = pd.read_parquet(_offline_run / "stores.parquet")
    fixture = pd.read_parquet(FIXTURES / "stores.parquet")
    pd.testing.assert_frame_equal(actual, fixture, check_dtype=True, check_exact=True)
