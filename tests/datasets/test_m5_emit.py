"""Tests for M5 setup-dir emission (issue 06).

All tests use small hand-built fixture files — no real M5 data.

Acceptance criteria:
1. One call takes raw-file dir + slice spec → emitted setup dir loadable by load_setup
2. Emitted scenario runs end-to-end and replays demand exactly under a flat world
3. Catalog fields derived as specified; unit_cost fraction documented and overridable
4. No insufficient_cash rejections at sinks in a fixture-based run (income_rate non-binding)
5. Prices and calendar parquets emitted alongside; events/SNAP not wired to the event engine
6. Quality report persisted in the emitted setup dir alongside the other artifacts
7. README committed next to the adapter (datasets package, not git-ignored data folder)
8. Raw and emitted M5 data paths are git-ignored
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.datasets.m5 import emit_m5_setup_dir, load_m5_slice
from src.sim.setup_io import load_setup


# ── fixture helpers ───────────────────────────────────────────────────────────


def _write_csv(path: Path, header: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _build_fixture_dir(tmp_path: Path) -> Path:
    """Build a minimal M5-shaped fixture directory.

    Timeline: 5 days, 2024-01-01 → 2024-01-05. One Walmart week (11201).

    Items: HOBBIES_1_001 (dept HOBBIES_1), FOODS_3_090 (dept FOODS_3)
    Stores: CA_1 (state CA), TX_2 (state TX)

    Sales (all non-zero to avoid zero-price corner cases):
      (HOBBIES_1_001, CA_1): [3, 5, 2, 4, 6]
      (HOBBIES_1_001, TX_2): [1, 2, 1, 3, 2]
      (FOODS_3_090, CA_1):   [10, 8, 12, 9, 11]
      (FOODS_3_090, TX_2):   [4, 5, 6, 3, 7]

    Prices (weekly, one week):
      (CA_1, HOBBIES_1_001): 10.0
      (TX_2, HOBBIES_1_001): 8.0
      (CA_1, FOODS_3_090):   4.0
      (TX_2, FOODS_3_090):   5.0
    """
    data_dir = tmp_path / "m5_fixture"
    data_dir.mkdir()

    # ── calendar.csv ─────────────────────────────────────────────────────────
    dates = [date(2024, 1, i) for i in range(1, 6)]
    cal_rows = [
        {
            "date": str(d),
            "wm_yr_wk": "11201",
            "weekday": "Monday",
            "wday": 2,
            "month": d.month,
            "year": d.year,
            "d": f"d_{i + 1}",
            "event_name_1": "",
            "event_type_1": "",
            "snap_CA": 0,
            "snap_TX": 0,
            "snap_WI": 0,
        }
        for i, d in enumerate(dates)
    ]
    cal_header = ["date", "wm_yr_wk", "weekday", "wday", "month", "year", "d",
                  "event_name_1", "event_type_1", "snap_CA", "snap_TX", "snap_WI"]
    _write_csv(data_dir / "calendar.csv", cal_header, cal_rows)

    # ── sales_train_fixture.csv ───────────────────────────────────────────────
    day_cols = [f"d_{i}" for i in range(1, 6)]
    sales_data = {
        ("HOBBIES_1_001", "CA_1", "HOBBIES_1", "HOBBIES", "CA"): [3, 5, 2, 4, 6],
        ("HOBBIES_1_001", "TX_2", "HOBBIES_1", "HOBBIES", "TX"): [1, 2, 1, 3, 2],
        ("FOODS_3_090", "CA_1", "FOODS_3", "FOODS", "CA"):        [10, 8, 12, 9, 11],
        ("FOODS_3_090", "TX_2", "FOODS_3", "FOODS", "TX"):        [4, 5, 6, 3, 7],
    }
    sales_rows = []
    for (item_id, store_id, dept_id, cat_id, state_id), vals in sales_data.items():
        row = {
            "id": f"{item_id}_{store_id}_validation",
            "item_id": item_id,
            "dept_id": dept_id,
            "cat_id": cat_id,
            "store_id": store_id,
            "state_id": state_id,
        }
        for col, v in zip(day_cols, vals):
            row[col] = v
        sales_rows.append(row)
    sales_header = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"] + day_cols
    _write_csv(data_dir / "sales_train_fixture.csv", sales_header, sales_rows)

    # ── sell_prices.csv ───────────────────────────────────────────────────────
    price_rows = [
        {"store_id": "CA_1", "item_id": "HOBBIES_1_001", "wm_yr_wk": "11201", "sell_price": 10.0},
        {"store_id": "TX_2", "item_id": "HOBBIES_1_001", "wm_yr_wk": "11201", "sell_price": 8.0},
        {"store_id": "CA_1", "item_id": "FOODS_3_090",   "wm_yr_wk": "11201", "sell_price": 4.0},
        {"store_id": "TX_2", "item_id": "FOODS_3_090",   "wm_yr_wk": "11201", "sell_price": 5.0},
    ]
    price_header = ["store_id", "item_id", "wm_yr_wk", "sell_price"]
    _write_csv(data_dir / "sell_prices.csv", price_header, price_rows)

    return data_dir


def _emit_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Build fixture dir, load slice, emit setup dir; return (data_dir, setup_dir)."""
    data_dir = _build_fixture_dir(tmp_path)
    m5 = load_m5_slice(
        data_dir,
        item_ids=["HOBBIES_1_001", "FOODS_3_090"],
        store_ids=["CA_1", "TX_2"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 5),
    )
    setup_dir = tmp_path / "emitted_setup"
    emit_m5_setup_dir(m5, data_dir, setup_dir)
    return data_dir, setup_dir


# ── AC1: one call → setup dir loadable by load_setup ─────────────────────────


class TestEmitLoadable:
    def test_emitted_setup_dir_loadable(self, tmp_path):
        """emit_m5_setup_dir produces a setup dir that load_setup accepts."""
        _, setup_dir = _emit_fixture(tmp_path)
        scenario = load_setup(setup_dir)
        assert scenario is not None

    def test_emitted_catalog_not_empty(self, tmp_path):
        """Emitted catalog has one Ware per item (not per item-store)."""
        _, setup_dir = _emit_fixture(tmp_path)
        scenario = load_setup(setup_dir)
        # 2 items → 2 wares
        assert len(scenario.catalog) == 2

    def test_emitted_nodes_include_shops_and_sinks(self, tmp_path):
        """Emitted topology includes one shop per store and one sink per (item, store)."""
        _, setup_dir = _emit_fixture(tmp_path)
        scenario = load_setup(setup_dir)

        node_types = {}
        for ni in scenario.nodes:
            node_types[ni.node.id] = type(ni.node).__name__

        intermediate_ids = [nid for nid, t in node_types.items() if t == "IntermediateNode"]
        replay_ids = [nid for nid, t in node_types.items() if t == "ReplayDemandSinkNode"]

        assert len(intermediate_ids) == 2, f"Expected 2 shops, got {intermediate_ids}"
        assert len(replay_ids) == 4, f"Expected 4 replay sinks (2 items × 2 stores), got {replay_ids}"

    def test_emitted_n_steps_matches_slice(self, tmp_path):
        """Emitted n_steps equals the slice window length."""
        _, setup_dir = _emit_fixture(tmp_path)
        scenario = load_setup(setup_dir)
        assert scenario.n_steps == 5  # 2024-01-01 to 2024-01-05


# ── AC2: exact replay under flat world ────────────────────────────────────────


class TestExactReplay:
    def test_replay_demand_matches_series(self, tmp_path):
        """Loaded emitted scenario runs and each sink replays demand exactly."""
        from src.sim.runner import Runner

        _, setup_dir = _emit_fixture(tmp_path)
        scenario = load_setup(setup_dir)
        log = Runner(scenario).run()

        # Expected demands per tick per (item, store):
        expected = {
            ("HOBBIES_1_001", "CA_1"): [3, 5, 2, 4, 6],
            ("HOBBIES_1_001", "TX_2"): [1, 2, 1, 3, 2],
            ("FOODS_3_090",   "CA_1"): [10, 8, 12, 9, 11],
            ("FOODS_3_090",   "TX_2"): [4, 5, 6, 3, 7],
        }

        # Map sink node ids to (item, store) — convention: "sink-<item>-<store>"
        # We identify sinks by checking their type and series matches expected series.
        from src.sim.replay_demand_sink import ReplayDemandSinkNode

        for ni in scenario.nodes:
            if not isinstance(ni.node, ReplayDemandSinkNode):
                continue
            node_id = ni.node.id
            node_series = list(ni.node.series)
            # Find which expected key matches this series.
            matched_key = None
            for key, exp_series in expected.items():
                if node_series == exp_series:
                    matched_key = key
                    break
            assert matched_key is not None, (
                f"Sink {node_id} series {node_series} not found in expected"
            )

            # Verify per-tick demand in the run log
            for tick_idx, tick_log in enumerate(log["ticks"]):
                sink_flows = [f for f in tick_log["node_flows"] if f["node_id"] == node_id]
                assert len(sink_flows) == 1
                assert sink_flows[0]["demand"] == expected[matched_key][tick_idx], (
                    f"Sink {node_id} tick {tick_idx}: "
                    f"expected {expected[matched_key][tick_idx]}, "
                    f"got {sink_flows[0]['demand']}"
                )


# ── AC3: catalog field derivation ─────────────────────────────────────────────


class TestCatalogDerivation:
    def test_base_price_is_median_across_stores(self, tmp_path):
        """base_price per item = median of observed prices over the slice (all stores)."""
        data_dir = _build_fixture_dir(tmp_path)
        m5 = load_m5_slice(
            data_dir,
            item_ids=["HOBBIES_1_001"],
            store_ids=["CA_1", "TX_2"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 5),
        )
        setup_dir = tmp_path / "setup"
        emit_m5_setup_dir(m5, data_dir, setup_dir)
        scenario = load_setup(setup_dir)

        # HOBBIES_1_001: CA_1 price = 10.0 (all days), TX_2 price = 8.0 (all days)
        # All observed prices = [10.0]*5 + [8.0]*5, median = 9.0
        ware = next(w for w in scenario.catalog if "HOBBIES_1_001" in w.name)
        import statistics
        observed_prices = [10.0] * 5 + [8.0] * 5
        expected_median = statistics.median(observed_prices)
        assert abs(ware.base_price - expected_median) < 1e-6

    def test_unit_cost_is_fraction_of_base_price(self, tmp_path):
        """unit_cost defaults to a documented fraction of base_price."""
        _, setup_dir = _emit_fixture(tmp_path)
        scenario = load_setup(setup_dir)
        for ware in scenario.catalog:
            # unit_cost must be < base_price (fraction < 1)
            assert ware.unit_cost < ware.base_price, (
                f"{ware.product_id}: unit_cost {ware.unit_cost} >= base_price {ware.base_price}"
            )
            # unit_cost must be > 0
            assert ware.unit_cost > 0.0

    def test_unit_cost_fraction_overridable(self, tmp_path):
        """emit_m5_setup_dir accepts a custom unit_cost_fraction parameter."""
        data_dir = _build_fixture_dir(tmp_path)
        m5 = load_m5_slice(
            data_dir,
            item_ids=["HOBBIES_1_001"],
            store_ids=["CA_1"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 5),
        )
        setup_dir = tmp_path / "setup_custom_cost"
        emit_m5_setup_dir(m5, data_dir, setup_dir, unit_cost_fraction=0.5)
        scenario = load_setup(setup_dir)
        ware = scenario.catalog[0]
        assert abs(ware.unit_cost - ware.base_price * 0.5) < 1e-6

    def test_category_derived_from_dept(self, tmp_path):
        """Ware.category matches the M5 dept_id of the item."""
        _, setup_dir = _emit_fixture(tmp_path)
        scenario = load_setup(setup_dir)
        categories = {w.name: w.category for w in scenario.catalog}
        # HOBBIES_1_001 → dept HOBBIES_1
        assert any("HOBBIES_1" in cat for cat in categories.values())
        # FOODS_3_090 → dept FOODS_3
        assert any("FOODS_3" in cat for cat in categories.values())

    def test_shop_region_derived_from_state(self, tmp_path):
        """Shop node region matches the M5 state_id of the store."""
        from src.sim.node import IntermediateNode

        _, setup_dir = _emit_fixture(tmp_path)
        scenario = load_setup(setup_dir)
        shop_regions = {
            ni.node.id: ni.node.region
            for ni in scenario.nodes
            if isinstance(ni.node, IntermediateNode)
        }
        # CA_1 → region CA, TX_2 → region TX
        assert any(r == "CA" for r in shop_regions.values())
        assert any(r == "TX" for r in shop_regions.values())


# ── AC4: income_rate non-binding (no insufficient_cash rejections) ────────────


class TestIncomeRateNonBinding:
    def test_no_insufficient_cash_rejections(self, tmp_path):
        """Sinks never fail to purchase due to cash constraints."""
        from src.sim.runner import Runner

        _, setup_dir = _emit_fixture(tmp_path)
        scenario = load_setup(setup_dir)
        log = Runner(scenario).run()

        for tick_log in log["ticks"]:
            for flow in tick_log["node_flows"]:
                # insufficient_cash key present means the sink was blocked by cash
                insufficient = flow.get("insufficient_cash", 0)
                assert insufficient == 0, (
                    f"Node {flow['node_id']} had insufficient_cash={insufficient}"
                )


# ── AC5: prices and calendar parquets emitted ─────────────────────────────────


class TestArtifactsEmitted:
    def test_prices_parquet_emitted(self, tmp_path):
        """emit_m5_setup_dir writes a prices parquet file into the setup dir."""
        _, setup_dir = _emit_fixture(tmp_path)
        assert (setup_dir / "prices.parquet").is_file()

    def test_prices_parquet_columns(self, tmp_path):
        """prices.parquet has item_id, store_id, date, price columns."""
        _, setup_dir = _emit_fixture(tmp_path)
        df = pd.read_parquet(setup_dir / "prices.parquet")
        required = {"item_id", "store_id", "date", "price"}
        assert required <= set(df.columns), f"Missing columns: {required - set(df.columns)}"

    def test_prices_parquet_row_count(self, tmp_path):
        """prices.parquet has n_ticks rows per (item, store) pair."""
        _, setup_dir = _emit_fixture(tmp_path)
        df = pd.read_parquet(setup_dir / "prices.parquet")
        # 2 items × 2 stores × 5 ticks
        assert len(df) == 20

    def test_calendar_parquet_emitted(self, tmp_path):
        """emit_m5_setup_dir writes a calendar parquet file into the setup dir."""
        _, setup_dir = _emit_fixture(tmp_path)
        assert (setup_dir / "calendar.parquet").is_file()

    def test_calendar_parquet_has_date_column(self, tmp_path):
        """calendar.parquet has at least a date column."""
        _, setup_dir = _emit_fixture(tmp_path)
        df = pd.read_parquet(setup_dir / "calendar.parquet")
        assert "date" in df.columns


# ── AC6: quality report persisted ─────────────────────────────────────────────


class TestQualityReportPersisted:
    def test_quality_report_parquet_emitted(self, tmp_path):
        """emit_m5_setup_dir writes quality_report.parquet into the setup dir."""
        _, setup_dir = _emit_fixture(tmp_path)
        assert (setup_dir / "quality_report.parquet").is_file()

    def test_quality_report_has_required_columns(self, tmp_path):
        """quality_report.parquet has the standard quality report columns."""
        _, setup_dir = _emit_fixture(tmp_path)
        df = pd.read_parquet(setup_dir / "quality_report.parquet")
        required = {
            "item_id", "store_id", "price_coverage_pct", "ffill_count",
            "bfill_count", "zero_sale_day_share", "longest_zero_run", "launch_date",
        }
        assert required <= set(df.columns)

    def test_quality_report_row_count(self, tmp_path):
        """quality_report.parquet has one row per (item, store) in the slice."""
        _, setup_dir = _emit_fixture(tmp_path)
        df = pd.read_parquet(setup_dir / "quality_report.parquet")
        assert len(df) == 4  # 2 items × 2 stores


# ── AC7: README exists next to adapter ────────────────────────────────────────


class TestReadme:
    def test_readme_exists_in_datasets_package(self):
        """README.md lives in src/datasets/ (not in git-ignored data/)."""
        readme = Path("src/datasets/README.md")
        assert readme.is_file(), "src/datasets/README.md not found"

    def test_readme_mentions_kaggle(self):
        """README mentions Kaggle (download instructions)."""
        readme = Path("src/datasets/README.md")
        content = readme.read_text(encoding="utf-8")
        assert "kaggle" in content.lower() or "Kaggle" in content

    def test_readme_mentions_emit_m5_setup_dir(self):
        """README references the emit_m5_setup_dir function."""
        readme = Path("src/datasets/README.md")
        content = readme.read_text(encoding="utf-8")
        assert "emit_m5_setup_dir" in content


# ── AC8: git-ignore coverage ──────────────────────────────────────────────────


class TestGitignore:
    def test_data_dir_already_gitignored(self):
        """data/ is already in .gitignore (blanket coverage for M5 raw/emitted data)."""
        gitignore = Path(".gitignore").read_text(encoding="utf-8")
        assert "data/" in gitignore
