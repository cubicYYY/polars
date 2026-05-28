"""Tests for dynamic (expression-based) interval support in datetime_ranges.

GitHub issue: https://github.com/pola-rs/polars/issues/12987
"""

import datetime as dt

import polars as pl


def test_datetime_ranges_dynamic_interval() -> None:
    """Test datetime_ranges with dynamic interval (allows sub-day intervals)."""
    df = pl.DataFrame(
        {
            "start": [dt.datetime(2023, 1, 1), dt.datetime(2023, 1, 1)],
            "end": [dt.datetime(2023, 1, 1, 6, 0), dt.datetime(2023, 1, 1, 6, 0)],
            "interval": ["2h", "3h"],
        }
    )
    result = df.with_columns(
        pl.datetime_ranges("start", "end", pl.col("interval")).alias("ranges")
    )
    ranges = result["ranges"].to_list()
    # 2h interval: 00:00, 02:00, 04:00, 06:00 = 4 items
    assert len(ranges[0]) == 4
    # 3h interval: 00:00, 03:00, 06:00 = 3 items
    assert len(ranges[1]) == 3


def test_datetime_ranges_dynamic_interval_column_name() -> None:
    """Test datetime_ranges with bare column name string as interval."""
    df = pl.DataFrame(
        {
            "start": [dt.datetime(2023, 1, 1), dt.datetime(2023, 1, 1)],
            "end": [dt.datetime(2023, 1, 2), dt.datetime(2023, 1, 2)],
            "interval": ["6h", "12h"],
        }
    )
    result = df.with_columns(
        pl.datetime_ranges("start", "end", "interval").alias("ranges")
    )
    ranges = result["ranges"].to_list()
    # 6h over 24h: 5 items (00, 06, 12, 18, 24)
    assert len(ranges[0]) == 5
    # 12h over 24h: 3 items (00, 12, 24)
    assert len(ranges[1]) == 3


def test_datetime_ranges_dynamic_interval_with_null() -> None:
    """Null interval in datetime_ranges produces null output."""
    df = pl.DataFrame(
        {
            "start": [dt.datetime(2023, 1, 1), dt.datetime(2023, 1, 1)],
            "end": [dt.datetime(2023, 1, 1, 6, 0), dt.datetime(2023, 1, 1, 6, 0)],
            "interval": ["2h", None],
        }
    )
    result = df.with_columns(
        pl.datetime_ranges("start", "end", pl.col("interval")).alias("ranges")
    )
    assert result["ranges"][0] is not None
    assert result["ranges"][1] is None


def test_datetime_ranges_static_interval_unchanged() -> None:
    """Existing literal interval behavior for datetime_ranges is preserved."""
    df = pl.DataFrame(
        {
            "start": [dt.datetime(2023, 1, 1), dt.datetime(2023, 1, 1)],
            "end": [dt.datetime(2023, 1, 1, 6, 0), dt.datetime(2023, 1, 1, 6, 0)],
        }
    )
    result = df.with_columns(
        pl.datetime_ranges("start", "end", "2h").alias("ranges")
    )
    ranges = result["ranges"].to_list()
    assert len(ranges[0]) == 4
    assert len(ranges[1]) == 4
