"""Tests for dynamic (expression-based) interval support in date_ranges.

GitHub issue: https://github.com/pola-rs/polars/issues/12987
"""

import datetime as dt

import polars as pl


def test_date_ranges_dynamic_interval_basic() -> None:
    """Basic test: different intervals per row."""
    df = pl.DataFrame(
        {
            "start": [dt.date(2023, 1, 1), dt.date(2023, 1, 1)],
            "end": [dt.date(2023, 4, 1), dt.date(2023, 4, 1)],
            "interval": ["1mo", "3mo"],
        }
    )
    result = df.with_columns(
        pl.date_ranges("start", "end", pl.col("interval")).alias("ranges")
    )
    ranges = result["ranges"].to_list()
    # "1mo" from Jan 1 to Apr 1: [Jan 1, Feb 1, Mar 1, Apr 1]
    assert ranges[0] == [
        dt.date(2023, 1, 1),
        dt.date(2023, 2, 1),
        dt.date(2023, 3, 1),
        dt.date(2023, 4, 1),
    ]
    # "3mo" from Jan 1 to Apr 1: [Jan 1, Apr 1]
    assert ranges[1] == [dt.date(2023, 1, 1), dt.date(2023, 4, 1)]


def test_date_ranges_static_interval_unchanged() -> None:
    """Existing behavior with literal interval string is unchanged."""
    df = pl.DataFrame(
        {
            "start": [dt.date(2023, 1, 1), dt.date(2023, 1, 1)],
            "end": [dt.date(2023, 3, 1), dt.date(2023, 3, 1)],
        }
    )
    result = df.with_columns(
        pl.date_ranges("start", "end", "1mo").alias("ranges")
    )
    ranges = result["ranges"].to_list()
    assert ranges[0] == [dt.date(2023, 1, 1), dt.date(2023, 2, 1), dt.date(2023, 3, 1)]
    assert ranges[1] == [dt.date(2023, 1, 1), dt.date(2023, 2, 1), dt.date(2023, 3, 1)]


def test_date_ranges_dynamic_interval_insurance_use_case() -> None:
    """The original use case from issue #12987: insurance payment schedules."""
    df = pl.DataFrame(
        {
            "policy_start_date": [dt.date(2023, 1, 1), dt.date(2023, 1, 1)],
            "policy_end_date": [dt.date(2024, 1, 1), dt.date(2024, 1, 1)],
            "installments": [12.0, 4.0],
        }
    )

    # Build interval string column: 12 installments = "1mo", 4 installments = "3mo"
    months_between = (pl.lit(12) / pl.col("installments")).cast(pl.Int32).cast(pl.String) + "mo"

    result = df.with_columns(
        pl.date_ranges(
            "policy_start_date", "policy_end_date", months_between
        ).alias("payment_dates")
    )

    payment_dates = result["payment_dates"].to_list()
    # Monthly (12 installments): 13 dates from Jan 2023 to Jan 2024
    assert len(payment_dates[0]) == 13
    assert payment_dates[0][0] == dt.date(2023, 1, 1)
    assert payment_dates[0][-1] == dt.date(2024, 1, 1)
    # Quarterly (4 installments): 5 dates
    assert len(payment_dates[1]) == 5
    assert payment_dates[1][0] == dt.date(2023, 1, 1)
    assert payment_dates[1][-1] == dt.date(2024, 1, 1)


def test_date_ranges_dynamic_interval_with_null() -> None:
    """Null interval should produce null output."""
    df = pl.DataFrame(
        {
            "start": [dt.date(2023, 1, 1), dt.date(2023, 1, 1)],
            "end": [dt.date(2023, 3, 1), dt.date(2023, 3, 1)],
            "interval": ["1mo", None],
        }
    )
    result = df.with_columns(
        pl.date_ranges("start", "end", pl.col("interval")).alias("ranges")
    )
    assert result["ranges"][0] is not None
    assert result["ranges"][1] is None


def test_date_ranges_dynamic_interval_scalar_broadcast() -> None:
    """Scalar start/end with multi-row interval."""
    df = pl.DataFrame({"interval": ["1mo", "2mo", "3mo"]})
    result = df.with_columns(
        pl.date_ranges(
            pl.lit(dt.date(2023, 1, 1)),
            pl.lit(dt.date(2023, 7, 1)),
            pl.col("interval"),
        ).alias("ranges")
    )
    ranges = result["ranges"].to_list()
    # 1mo: Jan, Feb, Mar, Apr, May, Jun, Jul = 7 dates
    assert len(ranges[0]) == 7
    # 2mo: Jan, Mar, May, Jul = 4 dates
    assert len(ranges[1]) == 4
    # 3mo: Jan, Apr, Jul = 3 dates
    assert len(ranges[2]) == 3
