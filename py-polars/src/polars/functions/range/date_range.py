from __future__ import annotations

import contextlib
from datetime import timedelta
from typing import TYPE_CHECKING, Any, overload

from polars import functions as F
from polars._utils.parse import parse_into_expression
from polars._utils.wrap import wrap_expr
from polars.functions.range._utils import parse_interval_argument

with contextlib.suppress(ImportError):  # Module not available when building docs
    import polars._plr as plr

if TYPE_CHECKING:
    from datetime import date, datetime
    from typing import Literal

    from polars import Expr, Series
    from polars._typing import ClosedInterval, IntoExpr, IntoExprColumn


def _parse_interval_or_expr(interval: str | timedelta | IntoExprColumn) -> Any:
    """Parse the interval argument as either a duration string or a PyExpr.

    Returns either a `str` (for literal intervals) or a `PyExpr` (for expression-based intervals).
    The Rust binding accepts both via `&Bound<'_, PyAny>`.

    For strings: if it looks like a duration (starts with a digit or '-'), treat as literal.
    Otherwise treat as a column name.
    """
    if isinstance(interval, timedelta):
        return parse_interval_argument(interval)
    if isinstance(interval, str):
        if interval and (interval[0].isdigit() or interval[0] == "-"):
            return parse_interval_argument(interval)
        # Doesn't look like a duration — treat as column name
        return parse_into_expression(interval)
    # Expr or Series
    return parse_into_expression(interval)


@overload
def date_range(
    start: date | datetime | IntoExprColumn,
    end: date | datetime | IntoExprColumn,
    interval: str | timedelta = ...,
    *,
    closed: ClosedInterval = ...,
    eager: Literal[False] = ...,
) -> Expr: ...


@overload
def date_range(
    start: date | datetime | IntoExprColumn,
    end: date | datetime | IntoExprColumn,
    interval: str | timedelta = ...,
    *,
    closed: ClosedInterval = ...,
    eager: Literal[True],
) -> Series: ...


@overload
def date_range(
    start: date | datetime | IntoExprColumn,
    end: date | datetime | IntoExprColumn,
    interval: str | timedelta = ...,
    *,
    closed: ClosedInterval = ...,
    eager: bool,
) -> Series | Expr: ...


def date_range(
    start: date | datetime | IntoExprColumn,
    end: date | datetime | IntoExprColumn,
    interval: str | timedelta = "1d",
    *,
    closed: ClosedInterval = "both",
    eager: bool = False,
) -> Series | Expr:
    """
    Generate a date range.

    Parameters
    ----------
    start
        Lower bound of the date range.
    end
        Upper bound of the date range.
    interval
        Interval of the range periods, specified as a Python `timedelta` object
        or using the Polars duration string language (see "Notes" section below).
        Must consist of full days.
    closed : {'both', 'left', 'right', 'none'}
        Define which sides of the range are closed (inclusive).
    eager
        Evaluate immediately and return a `Series`.
        If set to `False` (default), return an expression instead.

    Returns
    -------
    Expr or Series
        Column of data type :class:`Date`.

    See Also
    --------
    date_ranges
    datetime_range

    Notes
    -----
    `interval` is created according to the following string language:

    - 1d    (1 calendar day)
    - 1w    (1 calendar week)
    - 1mo   (1 calendar month)
    - 1q    (1 calendar quarter)
    - 1y    (1 calendar year)

    Or combine them:
    "1w2d" # 1 week, 2 days

    By "calendar day", we mean the corresponding time on the next day (which may
    not be 24 hours, due to daylight savings). Similarly for "calendar week",
    "calendar month", "calendar quarter", and "calendar year".

    Examples
    --------
    Using Polars duration string to specify the interval:

    >>> from datetime import date
    >>> pl.date_range(date(2022, 1, 1), date(2022, 3, 1), "1mo", eager=True).alias(
    ...     "date"
    ... )
    shape: (3,)
    Series: 'date' [date]
    [
        2022-01-01
        2022-02-01
        2022-03-01
    ]

    Using `timedelta` object to specify the interval:

    >>> from datetime import timedelta
    >>> pl.date_range(
    ...     date(1985, 1, 1),
    ...     date(1985, 1, 10),
    ...     timedelta(days=2),
    ...     eager=True,
    ... ).alias("date")
    shape: (5,)
    Series: 'date' [date]
    [
        1985-01-01
        1985-01-03
        1985-01-05
        1985-01-07
        1985-01-09
    ]

    Omit `eager=True` if you want to use `date_range` as an expression:

    >>> df = pl.DataFrame(
    ...     {
    ...         "date": [
    ...             date(2024, 1, 1),
    ...             date(2024, 1, 2),
    ...             date(2024, 1, 1),
    ...             date(2024, 1, 3),
    ...         ],
    ...         "key": ["one", "one", "two", "two"],
    ...     }
    ... )
    >>> result = (
    ...     df.group_by("key")
    ...     .agg(pl.date_range(pl.col("date").min(), pl.col("date").max()))
    ...     .sort("key")
    ... )
    >>> with pl.Config(fmt_str_lengths=50):
    ...     print(result)
    shape: (2, 2)
    ┌─────┬──────────────────────────────────────┐
    │ key ┆ date                                 │
    │ --- ┆ ---                                  │
    │ str ┆ list[date]                           │
    ╞═════╪══════════════════════════════════════╡
    │ one ┆ [2024-01-01, 2024-01-02]             │
    │ two ┆ [2024-01-01, 2024-01-02, 2024-01-03] │
    └─────┴──────────────────────────────────────┘
    """
    interval = parse_interval_argument(interval)

    start_pyexpr = parse_into_expression(start)
    end_pyexpr = parse_into_expression(end)
    result = wrap_expr(plr.date_range(start_pyexpr, end_pyexpr, interval, closed))

    if eager:
        return F.select(result).to_series()

    return result


@overload
def date_ranges(
    start: date | datetime | IntoExprColumn,
    end: date | datetime | IntoExprColumn,
    interval: str | timedelta | IntoExprColumn = ...,
    *,
    closed: ClosedInterval = ...,
    eager: Literal[False] = ...,
) -> Expr: ...


@overload
def date_ranges(
    start: date | datetime | IntoExprColumn,
    end: date | datetime | IntoExprColumn,
    interval: str | timedelta | IntoExprColumn = ...,
    *,
    closed: ClosedInterval = ...,
    eager: Literal[True],
) -> Series: ...


@overload
def date_ranges(
    start: date | datetime | IntoExprColumn,
    end: date | datetime | IntoExprColumn,
    interval: str | timedelta | IntoExprColumn = ...,
    *,
    closed: ClosedInterval = ...,
    eager: bool,
) -> Series | Expr: ...


def date_ranges(
    start: date | datetime | IntoExprColumn,
    end: date | datetime | IntoExprColumn,
    interval: str | timedelta | IntoExprColumn = "1d",
    *,
    closed: ClosedInterval = "both",
    eager: bool = False,
) -> Series | Expr:
    """
    Create a column of date ranges.

    Parameters
    ----------
    start
        Lower bound of the date range.
    end
        Upper bound of the date range.
    interval
        Interval of the range periods, specified as a Python `timedelta` object,
        using the Polars duration string language (see "Notes" section below),
        or as an expression/column name referencing a String column of duration
        strings. Must consist of full days.
    closed : {'both', 'left', 'right', 'none'}
        Define which sides of the range are closed (inclusive).
    eager
        Evaluate immediately and return a `Series`.
        If set to `False` (default), return an expression instead.

    Returns
    -------
    Expr or Series
        Column of data type `List(Date)`.

    See Also
    --------
    date_range
    datetime_ranges

    Notes
    -----
    `interval` is created according to the following string language:

    - 1d    (1 calendar day)
    - 1w    (1 calendar week)
    - 1mo   (1 calendar month)
    - 1q    (1 calendar quarter)
    - 1y    (1 calendar year)

    Or combine them:
    "1w2d" # 1 week, 2 days

    By "calendar day", we mean the corresponding time on the next day (which may
    not be 24 hours, due to daylight savings). Similarly for "calendar week",
    "calendar month", "calendar quarter", and "calendar year".

    When `interval` is a column expression, the duration string is parsed per row.
    If your data has only a few distinct interval values, consider using
    `partition_by` with a static interval for better performance.

    Examples
    --------
    >>> from datetime import date
    >>> df = pl.DataFrame(
    ...     {
    ...         "start": [date(2022, 1, 1), date(2022, 1, 2)],
    ...         "end": date(2022, 1, 3),
    ...     }
    ... )
    >>> with pl.Config(fmt_str_lengths=50):
    ...     df.with_columns(date_range=pl.date_ranges("start", "end"))
    shape: (2, 3)
    ┌────────────┬────────────┬──────────────────────────────────────┐
    │ start      ┆ end        ┆ date_range                           │
    │ ---        ┆ ---        ┆ ---                                  │
    │ date       ┆ date       ┆ list[date]                           │
    ╞════════════╪════════════╪══════════════════════════════════════╡
    │ 2022-01-01 ┆ 2022-01-03 ┆ [2022-01-01, 2022-01-02, 2022-01-03] │
    │ 2022-01-02 ┆ 2022-01-03 ┆ [2022-01-02, 2022-01-03]             │
    └────────────┴────────────┴──────────────────────────────────────┘
    """
    interval_parsed = _parse_interval_or_expr(interval)
    start_pyexpr = parse_into_expression(start)
    end_pyexpr = parse_into_expression(end)

    result = wrap_expr(plr.date_ranges(start_pyexpr, end_pyexpr, interval_parsed, closed))

    if eager:
        return F.select(result).to_series()

    return result
