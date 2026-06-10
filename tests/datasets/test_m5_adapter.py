"""Tests for the M5 adapter (issue 05).

All tests use small hand-built fixture files — no real M5 data.

Acceptance criteria:
- Parses sales, prices, and calendar files of the M5 schema.
- Slice selection by item ids, store ids, and date window.
- Weekly prices expand correctly across Walmart week boundaries.
- Gap ffill/bfill behave as specified, with counts surfaced.
- Quality report contains all required fields and flags long zero runs.
- No real M5 data in the repo or test suite.
"""

from __future__ import annotations

import csv
import os
import textwrap
from datetime import date
from pathlib import Path

import pytest

from src.datasets.m5 import M5Slice, load_m5_slice, quality_report


# ── fixture helpers ───────────────────────────────────────────────────────────


def _write_csv(path: Path, header: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _build_fixture_dir(tmp_path: Path) -> Path:
    """Build a minimal M5-shaped fixture directory.

    Timeline: 10 days, 2024-01-01 → 2024-01-10.
    Two Walmart weeks (wm_yr_wk):
      - "11201": 2024-01-01 … 2024-01-07
      - "11202": 2024-01-08 … 2024-01-10

    Items: ITEM_A, ITEM_B
    Stores: STORE_1, STORE_2

    Sales layout:
      (ITEM_A, STORE_1): [1,2,3,4,5,6,7,8,9,10]
      (ITEM_A, STORE_2): [0,0,0,1,2,3,4,5,6,7]  # 3 leading zeros
      (ITEM_B, STORE_1): [0,0,0,0,0,0,0,0,0,0]  # all zeros
      (ITEM_B, STORE_2): [5,0,0,0,0,5,0,0,0,5]  # scattered zeros

    Sell prices (weekly):
      (STORE_1, ITEM_A): week 11201 → 10.0, week 11202 → 12.0
      (STORE_2, ITEM_A): week 11202 only → 8.0   (leading bfill needed)
      (STORE_1, ITEM_B): week 11201 → 5.0 (no week 11202 price → ffill)
      # STORE_2, ITEM_B: no prices at all
    """
    data_dir = tmp_path / "m5_fixture"
    data_dir.mkdir()

    # ── calendar.csv ─────────────────────────────────────────────────────────
    dates = [date(2024, 1, i) for i in range(1, 11)]
    weeks = ["11201"] * 7 + ["11202"] * 3
    cal_rows = [
        {
            "date": str(d),
            "wm_yr_wk": w,
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
        for i, (d, w) in enumerate(zip(dates, weeks))
    ]
    cal_header = ["date", "wm_yr_wk", "weekday", "wday", "month", "year", "d",
                  "event_name_1", "event_type_1", "snap_CA", "snap_TX", "snap_WI"]
    _write_csv(data_dir / "calendar.csv", cal_header, cal_rows)

    # ── sales_train_fixture.csv ───────────────────────────────────────────────
    day_cols = [f"d_{i}" for i in range(1, 11)]
    sales_data = {
        ("ITEM_A", "STORE_1"): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ("ITEM_A", "STORE_2"): [0, 0, 0, 1, 2, 3, 4, 5, 6, 7],
        ("ITEM_B", "STORE_1"): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ("ITEM_B", "STORE_2"): [5, 0, 0, 0, 0, 5, 0, 0, 0, 5],
    }
    sales_rows = []
    for (item_id, store_id), vals in sales_data.items():
        row = {
            "id": f"{item_id}_{store_id}_validation",
            "item_id": item_id,
            "dept_id": "DEPT_1",
            "cat_id": "CAT_1",
            "store_id": store_id,
            "state_id": "CA",
        }
        for col, v in zip(day_cols, vals):
            row[col] = v
        sales_rows.append(row)
    sales_header = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"] + day_cols
    _write_csv(data_dir / "sales_train_fixture.csv", sales_header, sales_rows)

    # ── sell_prices.csv ───────────────────────────────────────────────────────
    price_rows = [
        {"store_id": "STORE_1", "item_id": "ITEM_A", "wm_yr_wk": "11201", "sell_price": 10.0},
        {"store_id": "STORE_1", "item_id": "ITEM_A", "wm_yr_wk": "11202", "sell_price": 12.0},
        {"store_id": "STORE_2", "item_id": "ITEM_A", "wm_yr_wk": "11202", "sell_price": 8.0},
        # ITEM_B STORE_1: only week 11201 → ffill needed for days 8-10
        {"store_id": "STORE_1", "item_id": "ITEM_B", "wm_yr_wk": "11201", "sell_price": 5.0},
        # ITEM_B STORE_2: no prices → all 0.0
    ]
    price_header = ["store_id", "item_id", "wm_yr_wk", "sell_price"]
    _write_csv(data_dir / "sell_prices.csv", price_header, price_rows)

    return data_dir


# ── tests ─────────────────────────────────────────────────────────────────────


def test_load_full_slice(tmp_path):
    """Load all items/stores for the full date range."""
    data_dir = _build_fixture_dir(tmp_path)
    m5 = load_m5_slice(
        data_dir,
        item_ids=["ITEM_A", "ITEM_B"],
        store_ids=["STORE_1", "STORE_2"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 10),
    )
    assert m5.n_ticks == 10
    assert set(m5.sales.keys()) == {
        ("ITEM_A", "STORE_1"),
        ("ITEM_A", "STORE_2"),
        ("ITEM_B", "STORE_1"),
        ("ITEM_B", "STORE_2"),
    }


def test_slice_by_items_stores(tmp_path):
    """Slicing by a subset of items/stores returns only the requested series."""
    data_dir = _build_fixture_dir(tmp_path)
    m5 = load_m5_slice(
        data_dir,
        item_ids=["ITEM_A"],
        store_ids=["STORE_1"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 10),
    )
    assert list(m5.sales.keys()) == [("ITEM_A", "STORE_1")]
    assert m5.sales[("ITEM_A", "STORE_1")] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_slice_by_date_window(tmp_path):
    """Slicing by a date sub-window returns only the ticks in that window."""
    data_dir = _build_fixture_dir(tmp_path)
    m5 = load_m5_slice(
        data_dir,
        item_ids=["ITEM_A"],
        store_ids=["STORE_1"],
        start_date=date(2024, 1, 3),
        end_date=date(2024, 1, 6),
    )
    assert m5.n_ticks == 4
    # d_3 .. d_6 → sales [3, 4, 5, 6]
    assert m5.sales[("ITEM_A", "STORE_1")] == [3, 4, 5, 6]


def test_weekly_prices_expand_to_daily_correctly(tmp_path):
    """Week 11201 (days 1-7) → 10.0; week 11202 (days 8-10) → 12.0."""
    data_dir = _build_fixture_dir(tmp_path)
    m5 = load_m5_slice(
        data_dir,
        item_ids=["ITEM_A"],
        store_ids=["STORE_1"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 10),
    )
    prices = m5.prices[("ITEM_A", "STORE_1")]
    assert prices[:7] == [10.0] * 7, "Week 1 price should be 10.0"
    assert prices[7:] == [12.0] * 3, "Week 2 price should be 12.0"


def test_weekly_price_boundary_across_weeks(tmp_path):
    """Price changes exactly at the Walmart week boundary (day 8 in this fixture)."""
    data_dir = _build_fixture_dir(tmp_path)
    m5 = load_m5_slice(
        data_dir,
        item_ids=["ITEM_A"],
        store_ids=["STORE_1"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 10),
    )
    prices = m5.prices[("ITEM_A", "STORE_1")]
    # Day 7 = last day of week 11201 → 10.0
    assert prices[6] == 10.0
    # Day 8 = first day of week 11202 → 12.0
    assert prices[7] == 12.0


def test_bfill_leading_gap(tmp_path):
    """STORE_2/ITEM_A has no price for week 11201 but price for week 11202.

    Days 1-7 (week 11201) must be back-filled with 8.0 (the first known price).
    """
    data_dir = _build_fixture_dir(tmp_path)
    m5 = load_m5_slice(
        data_dir,
        item_ids=["ITEM_A"],
        store_ids=["STORE_2"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 10),
    )
    prices = m5.prices[("ITEM_A", "STORE_2")]
    # All 10 days should be 8.0 (bfill from week 11202)
    assert prices == [8.0] * 10
    # bfill count == 7 (days in week 11201)
    assert m5.fill_counts[("ITEM_A", "STORE_2")]["bfill"] == 7
    assert m5.fill_counts[("ITEM_A", "STORE_2")]["ffill"] == 0


def test_ffill_trailing_gap(tmp_path):
    """STORE_1/ITEM_B only has price for week 11201.

    Days 8-10 (week 11202) must be forward-filled with 5.0.
    """
    data_dir = _build_fixture_dir(tmp_path)
    m5 = load_m5_slice(
        data_dir,
        item_ids=["ITEM_B"],
        store_ids=["STORE_1"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 10),
    )
    prices = m5.prices[("ITEM_B", "STORE_1")]
    assert prices[:7] == [5.0] * 7
    assert prices[7:] == [5.0] * 3  # ffilled
    assert m5.fill_counts[("ITEM_B", "STORE_1")]["ffill"] == 3
    assert m5.fill_counts[("ITEM_B", "STORE_1")]["bfill"] == 0


def test_no_price_series_produces_zeros(tmp_path):
    """STORE_2/ITEM_B has no sell_prices rows → prices are all 0.0."""
    data_dir = _build_fixture_dir(tmp_path)
    m5 = load_m5_slice(
        data_dir,
        item_ids=["ITEM_B"],
        store_ids=["STORE_2"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 10),
    )
    prices = m5.prices[("ITEM_B", "STORE_2")]
    assert prices == [0.0] * 10
    assert m5.fill_counts[("ITEM_B", "STORE_2")] == {"ffill": 0, "bfill": 0}


# ── quality report tests ──────────────────────────────────────────────────────


def test_quality_report_columns(tmp_path):
    """Report must contain exactly the required columns."""
    data_dir = _build_fixture_dir(tmp_path)
    m5 = load_m5_slice(
        data_dir,
        item_ids=["ITEM_A", "ITEM_B"],
        store_ids=["STORE_1", "STORE_2"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 10),
    )
    report = quality_report(m5)
    required_cols = {
        "item_id", "store_id", "price_coverage_pct", "ffill_count",
        "bfill_count", "zero_sale_day_share", "longest_zero_run", "launch_date",
    }
    assert required_cols <= set(report.columns)


def test_quality_report_row_count(tmp_path):
    """Report has one row per (item, store) combination."""
    data_dir = _build_fixture_dir(tmp_path)
    m5 = load_m5_slice(
        data_dir,
        item_ids=["ITEM_A", "ITEM_B"],
        store_ids=["STORE_1", "STORE_2"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 10),
    )
    report = quality_report(m5)
    assert len(report) == 4  # 2 items × 2 stores


def test_quality_report_price_coverage(tmp_path):
    """STORE_1/ITEM_A: price coverage should be 100% (prices for all 10 days)."""
    data_dir = _build_fixture_dir(tmp_path)
    m5 = load_m5_slice(
        data_dir,
        item_ids=["ITEM_A"],
        store_ids=["STORE_1"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 10),
    )
    report = quality_report(m5)
    row = report[report["store_id"] == "STORE_1"].iloc[0]
    assert row["price_coverage_pct"] == pytest.approx(100.0)


def test_quality_report_zero_sale_share_all_zeros(tmp_path):
    """STORE_1/ITEM_B: all-zero sales → zero_sale_day_share = 1.0."""
    data_dir = _build_fixture_dir(tmp_path)
    m5 = load_m5_slice(
        data_dir,
        item_ids=["ITEM_B"],
        store_ids=["STORE_1"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 10),
    )
    report = quality_report(m5)
    row = report.iloc[0]
    assert row["zero_sale_day_share"] == pytest.approx(1.0)


def test_quality_report_longest_zero_run(tmp_path):
    """STORE_2/ITEM_B: [5,0,0,0,0,5,0,0,0,5] → longest_zero_run = 4."""
    data_dir = _build_fixture_dir(tmp_path)
    m5 = load_m5_slice(
        data_dir,
        item_ids=["ITEM_B"],
        store_ids=["STORE_2"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 10),
    )
    report = quality_report(m5)
    row = report.iloc[0]
    assert row["longest_zero_run"] == 4


def test_quality_report_all_zeros_item_has_no_launch_date(tmp_path):
    """STORE_1/ITEM_B: all zeros → launch_date is None."""
    data_dir = _build_fixture_dir(tmp_path)
    m5 = load_m5_slice(
        data_dir,
        item_ids=["ITEM_B"],
        store_ids=["STORE_1"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 10),
    )
    report = quality_report(m5)
    row = report.iloc[0]
    assert row["launch_date"] is None or str(row["launch_date"]) == "None"


def test_quality_report_launch_date(tmp_path):
    """STORE_2/ITEM_A: first non-zero sale on 2024-01-04 (d_4 = day index 4, value 1).

    Sales: [0,0,0,1,2,3,4,5,6,7] → launch on 2024-01-04.
    """
    data_dir = _build_fixture_dir(tmp_path)
    m5 = load_m5_slice(
        data_dir,
        item_ids=["ITEM_A"],
        store_ids=["STORE_2"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 10),
    )
    report = quality_report(m5)
    row = report.iloc[0]
    assert row["launch_date"] == date(2024, 1, 4)


def test_fill_counts_surfaced_in_fill_counts(tmp_path):
    """fill_counts are present in the slice for every (item, store) pair."""
    data_dir = _build_fixture_dir(tmp_path)
    m5 = load_m5_slice(
        data_dir,
        item_ids=["ITEM_A", "ITEM_B"],
        store_ids=["STORE_1", "STORE_2"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 10),
    )
    for item_id in ["ITEM_A", "ITEM_B"]:
        for store_id in ["STORE_1", "STORE_2"]:
            key = (item_id, store_id)
            assert key in m5.fill_counts
            counts = m5.fill_counts[key]
            assert "ffill" in counts and "bfill" in counts
