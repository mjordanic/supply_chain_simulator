"""M5 dataset adapter (issue 05).

Sibling package — ``src/datasets/`` — that the core sim never imports
(ADR 0010 pattern).  Parses the three raw M5 Kaggle files, slices by
(items × stores × date window), expands weekly prices to daily, and
produces a quality report.

M5 file schema
--------------
``sales_train_*.csv``
    Columns: ``id``, ``item_id``, ``dept_id``, ``cat_id``, ``store_id``,
    ``state_id``, then one numeric column per day ``d_1`` … ``d_N``.

``sell_prices.csv``
    Columns: ``store_id``, ``item_id``, ``wm_yr_wk``, ``sell_price``.
    One row per (store, item, Walmart-week) when the item is on sale.
    Items have no rows for weeks before their launch date.

``calendar.csv``
    Columns include: ``date``, ``wm_yr_wk``, ``d`` (the day index as
    ``"d_1"`` … ``"d_N"``).

Public API
----------
``load_m5_slice(data_dir, item_ids, store_ids, start_date, end_date)``
    Return a ``M5Slice`` containing:
    - ``sales``: ``dict[(item_id, store_id), list[int]]`` — daily sales
      series, tick 0 = start_date.
    - ``prices``: ``dict[(item_id, store_id), list[float]]`` — daily
      selling prices after ffill / bfill.
    - ``fill_counts``: ``dict[(item_id, store_id), dict]`` — ffill/bfill
      counts per series.
    - ``start_date``, ``end_date``, ``item_ids``, ``store_ids``.

``quality_report(m5_slice)``
    Return a ``pandas.DataFrame`` with one row per (item, store) and
    columns: ``item_id``, ``store_id``, ``price_coverage_pct``,
    ``ffill_count``, ``bfill_count``, ``zero_sale_day_share``,
    ``longest_zero_run``, ``launch_date``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator


def _date_range(start: date, end: date) -> list[date]:
    """Return list of dates from start to end (inclusive)."""
    result = []
    d = start
    while d <= end:
        result.append(d)
        d += timedelta(days=1)
    return result


@dataclass
class M5Slice:
    """Result of ``load_m5_slice``.

    Attributes
    ----------
    sales:
        ``{(item_id, store_id): [qty, ...]}`` — daily sales, len = n_ticks.
    prices:
        ``{(item_id, store_id): [price, ...]}`` — daily prices after gap-fill,
        len = n_ticks.
    fill_counts:
        ``{(item_id, store_id): {"ffill": int, "bfill": int}}`` — fill stats.
    start_date:
        First date of the window (tick 0).
    end_date:
        Last date of the window (tick n_ticks - 1, inclusive).
    item_ids:
        Requested item IDs (in request order).
    store_ids:
        Requested store IDs (in request order).
    """

    sales: dict[tuple[str, str], list[int]] = field(default_factory=dict)
    prices: dict[tuple[str, str], list[float]] = field(default_factory=dict)
    fill_counts: dict[tuple[str, str], dict[str, int]] = field(default_factory=dict)
    start_date: date | None = None
    end_date: date | None = None
    item_ids: list[str] = field(default_factory=list)
    store_ids: list[str] = field(default_factory=list)

    @property
    def n_ticks(self) -> int:
        """Number of ticks (days) in the window."""
        if self.start_date is None or self.end_date is None:
            return 0
        return (self.end_date - self.start_date).days + 1


def _parse_calendar(
    calendar_path: str | Path,
    start_date: date,
    end_date: date,
) -> tuple[list[str], dict[str, str]]:
    """Parse calendar.csv and return:
    - ``day_ids``: list of d_<N> strings in [start_date, end_date] order.
    - ``day_to_wm_yr_wk``: ``{day_id: wm_yr_wk}`` for the same range.

    The ``wm_yr_wk`` column is the Walmart week identifier; used to join
    sell_prices.csv onto daily rows.
    """
    import csv

    day_ids: list[str] = []
    day_to_wm: dict[str, str] = {}

    with open(calendar_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # Parse the date column; M5 uses "YYYY-MM-DD" format.
            row_date = date.fromisoformat(row["date"])
            if row_date < start_date or row_date > end_date:
                continue
            day_id = row["d"]
            wm_yr_wk = row["wm_yr_wk"]
            day_ids.append(day_id)
            day_to_wm[day_id] = wm_yr_wk

    return day_ids, day_to_wm


def _parse_sales(
    sales_path: str | Path,
    item_ids: list[str],
    store_ids: list[str],
    day_ids: list[str],
) -> dict[tuple[str, str], list[int]]:
    """Parse the sales CSV and return per-(item, store) daily sales vectors.

    Rows not matching ``item_ids`` × ``store_ids`` are skipped.
    ``day_ids`` drives the column selection and preserves the window order.
    """
    import csv

    wanted_items = set(item_ids)
    wanted_stores = set(store_ids)
    # Column subset: only the day columns we need.
    wanted_days = set(day_ids)

    result: dict[tuple[str, str], list[int]] = {}

    with open(sales_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            item_id = row.get("item_id", "")
            store_id = row.get("store_id", "")
            if item_id not in wanted_items or store_id not in wanted_stores:
                continue
            key = (item_id, store_id)
            # Build the daily series in day_ids order (not CSV column order).
            series: list[int] = []
            for day in day_ids:
                val = row.get(day)
                series.append(int(val) if val is not None and val != "" else 0)
            result[key] = series

    return result


def _parse_sell_prices(
    prices_path: str | Path,
    item_ids: list[str],
    store_ids: list[str],
) -> dict[tuple[str, str, str], float]:
    """Parse sell_prices.csv into a lookup dict ``{(store_id, item_id, wm_yr_wk): price}``.

    Only rows for ``item_ids`` × ``store_ids`` are loaded.
    """
    import csv

    wanted_items = set(item_ids)
    wanted_stores = set(store_ids)
    result: dict[tuple[str, str, str], float] = {}

    with open(prices_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            store_id = row.get("store_id", "")
            item_id = row.get("item_id", "")
            if item_id not in wanted_items or store_id not in wanted_stores:
                continue
            wm_yr_wk = row["wm_yr_wk"]
            price_str = row.get("sell_price", "")
            if price_str == "" or price_str is None:
                continue
            result[(store_id, item_id, wm_yr_wk)] = float(price_str)

    return result


def _expand_prices(
    item_id: str,
    store_id: str,
    day_ids: list[str],
    day_to_wm: dict[str, str],
    price_lookup: dict[tuple[str, str, str], float],
) -> tuple[list[float | None], dict[str, int]]:
    """Expand weekly sell-prices to a daily series, then fill gaps.

    1. Map each day_id to its ``wm_yr_wk``.
    2. Look up the weekly price for that (store, item, week).
    3. Forward-fill interior gaps.
    4. Back-fill leading gaps (pre-launch).
    5. Any still-None values after both fills default to 0.0.

    Returns
    -------
    daily_prices : list[float]
        Length == len(day_ids).
    fill_counts : {"ffill": int, "bfill": int}
    """
    n = len(day_ids)
    raw: list[float | None] = []
    for day_id in day_ids:
        wm = day_to_wm.get(day_id)
        if wm is not None:
            price = price_lookup.get((store_id, item_id, wm))
        else:
            price = None
        raw.append(price)

    # Forward-fill: propagate last known price forward.
    filled: list[float | None] = list(raw)
    ffill_count = 0
    last_known: float | None = None
    for i in range(n):
        if filled[i] is not None:
            last_known = filled[i]
        elif last_known is not None:
            filled[i] = last_known
            ffill_count += 1

    # Back-fill: for leading None values (pre-launch), use first known price.
    bfill_count = 0
    first_known: float | None = None
    for v in filled:
        if v is not None:
            first_known = v
            break
    if first_known is not None:
        for i in range(n):
            if filled[i] is None:
                filled[i] = first_known
                bfill_count += 1
            else:
                break  # once we hit a non-None, leading gap is done

    # Replace any remaining None with 0.0 (no price data at all).
    final: list[float] = [v if v is not None else 0.0 for v in filled]
    return final, {"ffill": ffill_count, "bfill": bfill_count}


def load_m5_slice(
    data_dir: str | Path,
    item_ids: list[str],
    store_ids: list[str],
    start_date: date,
    end_date: date,
) -> M5Slice:
    """Parse M5 raw files and return a slice for the given items/stores/window.

    Expects ``data_dir`` to contain:
    - A CSV file whose name starts with ``sales_train`` (sales data).
    - ``sell_prices.csv``
    - ``calendar.csv``

    Parameters
    ----------
    data_dir:
        Directory containing the raw M5 CSV files.
    item_ids:
        Item IDs to include (e.g. ``["HOBBIES_1_001", "FOODS_3_090"]``).
    store_ids:
        Store IDs to include (e.g. ``["CA_1", "TX_2"]``).
    start_date:
        Start of the date window (tick 0).
    end_date:
        End of the date window (inclusive).

    Returns
    -------
    M5Slice
    """
    data_dir = Path(data_dir)

    # Locate sales CSV (name starts with "sales_train").
    sales_candidates = [
        f for f in data_dir.iterdir()
        if f.name.startswith("sales_train") and f.suffix == ".csv"
    ]
    if not sales_candidates:
        raise FileNotFoundError(
            f"No sales_train*.csv found in {data_dir}. "
            "Expected a file whose name starts with 'sales_train'."
        )
    sales_path = sales_candidates[0]
    calendar_path = data_dir / "calendar.csv"
    prices_path = data_dir / "sell_prices.csv"

    # 1. Parse calendar for the window.
    day_ids, day_to_wm = _parse_calendar(calendar_path, start_date, end_date)
    if not day_ids:
        raise ValueError(
            f"No calendar days found in [{start_date}, {end_date}]. "
            "Check that start_date/end_date are within the M5 date range."
        )

    # 2. Parse daily sales.
    sales = _parse_sales(sales_path, item_ids, store_ids, day_ids)

    # 3. Parse weekly sell prices.
    price_lookup = _parse_sell_prices(prices_path, item_ids, store_ids)

    # 4. Expand prices to daily and fill gaps.
    prices: dict[tuple[str, str], list[float]] = {}
    fill_counts: dict[tuple[str, str], dict[str, int]] = {}
    for item_id in item_ids:
        for store_id in store_ids:
            key = (item_id, store_id)
            daily, counts = _expand_prices(
                item_id, store_id, day_ids, day_to_wm, price_lookup
            )
            prices[key] = daily
            fill_counts[key] = counts

    return M5Slice(
        sales=sales,
        prices=prices,
        fill_counts=fill_counts,
        start_date=start_date,
        end_date=end_date,
        item_ids=list(item_ids),
        store_ids=list(store_ids),
    )


def quality_report(m5_slice: M5Slice) -> "Any":  # returns pd.DataFrame
    """Build a per-(item, store) quality report for the slice.

    Columns
    -------
    item_id, store_id
        Identifiers.
    price_coverage_pct
        Fraction of ticks with a non-zero price (%).
    ffill_count
        Number of days whose price was forward-filled from a prior week.
    bfill_count
        Number of days whose price was back-filled (pre-launch).
    zero_sale_day_share
        Fraction of ticks with zero sales.
    longest_zero_run
        Longest consecutive run of zero-sale days (suspected OOS indicator).
    launch_date
        First date with a non-zero sale (or None if always zero).

    Parameters
    ----------
    m5_slice:
        A ``M5Slice`` returned by ``load_m5_slice``.

    Returns
    -------
    pandas.DataFrame
        One row per (item_id, store_id); sorted by item_id, then store_id.
    """
    import pandas as pd

    rows: list[dict] = []
    n_ticks = m5_slice.n_ticks
    dates = (
        _date_range(m5_slice.start_date, m5_slice.end_date)
        if m5_slice.start_date and m5_slice.end_date
        else []
    )

    for item_id in m5_slice.item_ids:
        for store_id in m5_slice.store_ids:
            key = (item_id, store_id)
            sales_series = m5_slice.sales.get(key, [0] * n_ticks)
            price_series = m5_slice.prices.get(key, [0.0] * n_ticks)
            counts = m5_slice.fill_counts.get(key, {"ffill": 0, "bfill": 0})

            # Price coverage: fraction of days with a positive price.
            n_with_price = sum(1 for p in price_series if p > 0.0)
            price_coverage_pct = (n_with_price / n_ticks * 100.0) if n_ticks > 0 else 0.0

            # Zero-sale stats.
            n_zero = sum(1 for s in sales_series if s == 0)
            zero_sale_day_share = (n_zero / n_ticks) if n_ticks > 0 else 0.0

            # Longest consecutive zero run.
            longest = 0
            current = 0
            for s in sales_series:
                if s == 0:
                    current += 1
                    if current > longest:
                        longest = current
                else:
                    current = 0

            # Launch date: first day with a positive sale.
            launch_date = None
            for i, s in enumerate(sales_series):
                if s > 0:
                    if dates:
                        launch_date = dates[i]
                    else:
                        launch_date = i  # fallback: tick index
                    break

            rows.append({
                "item_id": item_id,
                "store_id": store_id,
                "price_coverage_pct": price_coverage_pct,
                "ffill_count": counts["ffill"],
                "bfill_count": counts["bfill"],
                "zero_sale_day_share": zero_sale_day_share,
                "longest_zero_run": longest,
                "launch_date": launch_date,
            })

    return pd.DataFrame(rows, columns=[
        "item_id", "store_id", "price_coverage_pct", "ffill_count", "bfill_count",
        "zero_sale_day_share", "longest_zero_run", "launch_date",
    ])


__all__ = ["M5Slice", "load_m5_slice", "quality_report"]
