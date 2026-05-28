import pytest

import polars as pl
from polars.testing import assert_frame_equal


def test_rolling_argmax_basic() -> None:
    df = pl.DataFrame({"a": [1, 5, 3, 4, 2]})
    result = df.select(pl.col("a").rolling_argmax(window_size=3))
    expected = pl.DataFrame({"a": [None, None, 1, 0, 1]}, schema={"a": pl.UInt32})
    assert_frame_equal(result, expected)


def test_rolling_argmin_basic() -> None:
    df = pl.DataFrame({"a": [1, 5, 3, 4, 2]})
    result = df.select(pl.col("a").rolling_argmin(window_size=3))
    expected = pl.DataFrame({"a": [None, None, 0, 1, 2]}, schema={"a": pl.UInt32})
    assert_frame_equal(result, expected)


def test_rolling_argmax_min_samples() -> None:
    df = pl.DataFrame({"a": [1, 5, 3, 4, 2]})
    result = df.select(pl.col("a").rolling_argmax(window_size=3, min_samples=1))
    expected = pl.DataFrame({"a": [0, 1, 1, 0, 1]}, schema={"a": pl.UInt32})
    assert_frame_equal(result, expected)


def test_rolling_argmin_min_samples() -> None:
    df = pl.DataFrame({"a": [1, 5, 3, 4, 2]})
    result = df.select(pl.col("a").rolling_argmin(window_size=3, min_samples=1))
    expected = pl.DataFrame({"a": [0, 0, 0, 1, 2]}, schema={"a": pl.UInt32})
    assert_frame_equal(result, expected)


def test_rolling_argmax_centered() -> None:
    df = pl.DataFrame({"a": [1.0, 5.0, 3.0, 4.0, 2.0]})
    result = df.select(pl.col("a").rolling_argmax(window_size=3, min_samples=1, center=True))
    expected = pl.DataFrame({"a": [1, 1, 0, 1, 0]}, schema={"a": pl.UInt32})
    assert_frame_equal(result, expected)


def test_rolling_argmax_with_nulls() -> None:
    df = pl.DataFrame({"a": [1.0, None, 3.0, 2.0, 5.0]})
    result = df.select(pl.col("a").rolling_argmax(window_size=3, min_samples=2))
    expected = pl.DataFrame({"a": [None, None, 2, 1, 2]}, schema={"a": pl.UInt32})
    assert_frame_equal(result, expected)


def test_rolling_argmin_with_nulls() -> None:
    df = pl.DataFrame({"a": [1.0, None, 3.0, 2.0, 5.0]})
    result = df.select(pl.col("a").rolling_argmin(window_size=3, min_samples=2))
    expected = pl.DataFrame({"a": [None, None, 0, 2, 1]}, schema={"a": pl.UInt32})
    assert_frame_equal(result, expected)


def test_rolling_argmax_consistency_with_rolling_max() -> None:
    """Verify that values[start + argmax] == rolling_max for each row."""
    values = [2.0, 7.0, 1.0, 8.0, 3.0, 6.0, 4.0, 9.0, 5.0, 0.0]
    df = pl.DataFrame({"a": values})
    result = df.select(
        argmax=pl.col("a").rolling_argmax(window_size=4),
        max_val=pl.col("a").rolling_max(window_size=4),
    )
    for i in range(len(values)):
        argmax = result["argmax"][i]
        max_val = result["max_val"][i]
        if argmax is not None and max_val is not None:
            window_start = max(0, i - 3)
            actual = values[window_start + argmax]
            assert actual == max_val, f"at i={i}: values[{window_start}+{argmax}]={actual} != {max_val}"


def test_rolling_argmax_output_dtype() -> None:
    """Output should always be UInt32 (IDX_DTYPE) regardless of input type."""
    for dtype in [pl.Int32, pl.Int64, pl.Float32, pl.Float64]:
        df = pl.DataFrame({"a": pl.Series([1, 2, 3, 4, 5], dtype=dtype)})
        result = df.select(pl.col("a").rolling_argmax(window_size=3))
        assert result["a"].dtype == pl.UInt32


def test_rolling_argmax_all_equal() -> None:
    """When all values are equal, argmax should return 0 (first occurrence)."""
    df = pl.DataFrame({"a": [3.0, 3.0, 3.0, 3.0, 3.0]})
    result = df.select(pl.col("a").rolling_argmax(window_size=3))
    expected = pl.DataFrame({"a": [None, None, 0, 0, 0]}, schema={"a": pl.UInt32})
    assert_frame_equal(result, expected)


def test_rolling_argmax_window_1() -> None:
    """Window size 1: argmax is always 0."""
    df = pl.DataFrame({"a": [3, 1, 4, 1, 5]})
    result = df.select(pl.col("a").rolling_argmax(window_size=1))
    expected = pl.DataFrame({"a": [0, 0, 0, 0, 0]}, schema={"a": pl.UInt32})
    assert_frame_equal(result, expected)


def test_rolling_argmax_empty() -> None:
    """Empty series should return empty."""
    df = pl.DataFrame({"a": pl.Series([], dtype=pl.Float64)})
    result = df.select(pl.col("a").rolling_argmax(window_size=3))
    assert result.height == 0
    assert result["a"].dtype == pl.UInt32


def test_rolling_argmax_by_basic() -> None:
    """Test rolling_argmax_by with a time-based window."""
    df = pl.DataFrame({
        "date": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 5), eager=True),
        "a": [1.0, 5.0, 3.0, 4.0, 2.0],
    })
    result = df.select(
        pl.col("a").rolling_argmax_by("date", window_size="3d")
    )
    # 3-day window: each row sees itself and 2 days prior
    # row 0 (Jan 1): [1.0] -> argmax=0
    # row 1 (Jan 2): [1.0, 5.0] -> argmax=1
    # row 2 (Jan 3): [1.0, 5.0, 3.0] -> argmax=1
    # row 3 (Jan 4): [5.0, 3.0, 4.0] -> argmax=0
    # row 4 (Jan 5): [3.0, 4.0, 2.0] -> argmax=1
    expected = pl.DataFrame({"a": [0, 1, 1, 0, 1]}, schema={"a": pl.UInt32})
    assert_frame_equal(result, expected)


def test_rolling_argmin_by_basic() -> None:
    """Test rolling_argmin_by with a time-based window."""
    df = pl.DataFrame({
        "date": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 5), eager=True),
        "a": [1.0, 5.0, 3.0, 4.0, 2.0],
    })
    result = df.select(
        pl.col("a").rolling_argmin_by("date", window_size="3d")
    )
    # row 0 (Jan 1): [1.0] -> argmin=0
    # row 1 (Jan 2): [1.0, 5.0] -> argmin=0
    # row 2 (Jan 3): [1.0, 5.0, 3.0] -> argmin=0
    # row 3 (Jan 4): [5.0, 3.0, 4.0] -> argmin=1
    # row 4 (Jan 5): [3.0, 4.0, 2.0] -> argmin=2
    expected = pl.DataFrame({"a": [0, 0, 0, 1, 2]}, schema={"a": pl.UInt32})
    assert_frame_equal(result, expected)


def test_rolling_argmax_lazy_schema() -> None:
    """Schema inference should work correctly in lazy mode."""
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    lf = df.lazy().with_columns(
        pl.col("a").rolling_argmax(window_size=3).alias("argmax")
    )
    schema = lf.collect_schema()
    assert schema["argmax"] == pl.UInt32
    result = lf.collect()
    assert result["argmax"].dtype == pl.UInt32
