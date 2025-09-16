# noinspection SpellCheckingInspection
"""
Time Conversion Module

This module provides utilities for handling time representations in GNSS applications.
It includes functions for converting between different time systems and formats:
- GPS time (weeks and seconds of week)
- UTC time
- Handling of leap seconds
- Conversion between different datetime representations

The module supports precise time calculations necessary for observable positioning
and navigation applications.
"""

import datetime
import functools
from typing import Union, Tuple

import numpy as np


@functools.singledispatch
def datetime_gps_to_week_and_seconds(dt):
    """
    Extract GPS week and seconds of week from a datetime object that's already in GPS time.
    No UTC to GPS conversion is performed.

    Args:
        dt: Datetime object (Python, numpy, or array) in GPS timescale

    Returns:
        (np.int32, np.float64): GPS week number and seconds of week

    Raises:
        TypeError: If input type is not supported
    """
    raise TypeError(f"Unsupported input type: {type(dt)}")


@datetime_gps_to_week_and_seconds.register(np.ndarray)
def _(dt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Handle numpy arrays for vectorized operation"""
    if not np.issubdtype(dt.dtype, np.datetime64):
        raise TypeError("Array must contain datetime64 values")

    # GPS epoch (January 6, 1980)
    gps_epoch = np.datetime64("1980-01-06T00:00:00")

    # Check if any time is before GPS epoch
    if np.any(dt < gps_epoch):
        raise ValueError("GPS time cannot be earlier than GPS epoch (January 6, 1980)")

    # Calculate the difference in seconds (vectorized)
    total_seconds = (dt - gps_epoch) / np.timedelta64(1, 's')

    # Compute GPS week and time of week using numpy divmod (vectorized)
    gps_week, seconds_of_week = np.divmod(total_seconds, 7 * 24 * 60 * 60)

    return gps_week.astype(np.int32), seconds_of_week.astype(np.float64)


@datetime_gps_to_week_and_seconds.register(np.datetime64)
def _(dt: np.datetime64) -> Tuple[np.int32, np.float64]:
    """Handle numpy datetime64 input by using the ndarray handler"""
    weeks, seconds = datetime_gps_to_week_and_seconds(np.array([dt]))
    return np.int32(weeks[0]), np.float64(seconds[0])


@datetime_gps_to_week_and_seconds.register(datetime.datetime)
@datetime_gps_to_week_and_seconds.register(str)
def _(dt: Union[datetime.datetime, str]) -> Tuple[np.int32, np.float64]:
    """Handle Python's datetime.datetime or string by converting to numpy.datetime64"""
    return datetime_gps_to_week_and_seconds(np.datetime64(dt))


def datetime_utc_to_datetime_gps(dt: Union[np.datetime64, datetime.datetime, str]) -> np.datetime64:
    """
    Convert UTC time to GPS time.

    GPS time started at 00:00:00 January 6, 1980, UTC with 0 leap seconds.
    GPS time does not include leap seconds, so the difference between
    UTC and GPS time increases with each leap second addition.

    Args:
        dt: UTC time as a datetime object, numpy datetime64 object,
                  or as an ISO format string (e.g., "2023-05-24T12:34:56Z")

    Returns:
        np.datetime64: GPS datetime
    """

    # Leap seconds table (UTC - GPS) as of April 2025
    # This needs to be updated when new leap seconds are added!
    leap_seconds = [
        (np.datetime64("1980-01-06T00:00:00"), np.int32(0)),
        (np.datetime64("1981-07-01T00:00:00"), np.int32(1)),
        (np.datetime64("1982-07-01T00:00:00"), np.int32(2)),
        (np.datetime64("1983-07-01T00:00:00"), np.int32(3)),
        (np.datetime64("1985-07-01T00:00:00"), np.int32(4)),
        (np.datetime64("1988-01-01T00:00:00"), np.int32(5)),
        (np.datetime64("1990-01-01T00:00:00"), np.int32(6)),
        (np.datetime64("1991-01-01T00:00:00"), np.int32(7)),
        (np.datetime64("1992-07-01T00:00:00"), np.int32(8)),
        (np.datetime64("1993-07-01T00:00:00"), np.int32(9)),
        (np.datetime64("1994-07-01T00:00:00"), np.int32(10)),
        (np.datetime64("1996-01-01T00:00:00"), np.int32(11)),
        (np.datetime64("1997-07-01T00:00:00"), np.int32(12)),
        (np.datetime64("1999-01-01T00:00:00"), np.int32(13)),
        (np.datetime64("2006-01-01T00:00:00"), np.int32(14)),
        (np.datetime64("2009-01-01T00:00:00"), np.int32(15)),
        (np.datetime64("2012-07-01T00:00:00"), np.int32(16)),
        (np.datetime64("2015-07-01T00:00:00"), np.int32(17)),
        (np.datetime64("2017-01-01T00:00:00"), np.int32(18)),
    ]

    # Convert input to numpy.datetime64 if it's not already
    if isinstance(dt, (datetime.datetime, str)):
        dt = np.datetime64(dt)

    # GPS time epoch
    gps_epoch = np.datetime64("1980-01-06T00:00:00")

    # Check if time is before GPS epoch
    if dt < gps_epoch:
        raise ValueError("UTC time cannot be earlier than GPS epoch (January 6, 1980)")

    # Find applicable leap seconds
    leap_second_offset = np.int32(0)
    for leap_date, offset in leap_seconds:
        if dt >= leap_date:
            leap_second_offset = offset

    # Create a GPS time by adding the leap seconds to UTC
    gps_time = dt + np.timedelta64(leap_second_offset, 's')

    return gps_time


def datetime_utc_to_week_and_seconds(dt: Union[np.datetime64, datetime.datetime, str]) -> Tuple[np.int32, np.float64]:
    """
    Convert UTC time to GPS week and seconds of week.

    GPS time started at 00:00:00 January 6, 1980, UTC with 0 leap seconds.
    GPS time does not include leap seconds, so the difference between
    UTC and GPS time increases with each leap second addition.

    Args:
        dt: UTC time as a datetime object, numpy datetime64 object,
                  or as an ISO format string (e.g., "2023-05-24T12:34:56Z")

    Returns:
        Tuple[np.int32, np.float64]: GPS week number and seconds of week
    """
    # First convert UTC to GPS time
    gps_time = datetime_utc_to_datetime_gps(dt)

    # Then extract GPS week and seconds
    return datetime_gps_to_week_and_seconds(gps_time)


# Function that given a np.datetime64 return the day of year
def datetime_to_day_of_year(dt: Union[np.datetime64, datetime.datetime]) -> int:
    """
    Convert a datetime object to the day of the year.

    Args:
        dt: Datetime object (Python or numpy)

    Returns:
        int: Day of the year (1-366)
    """
    # Convert input to numpy.datetime64 if it's not already
    if isinstance(dt, (datetime.datetime, str)):
        dt = np.datetime64(dt)

    # Get the start of the year
    start_of_year = np.datetime64(str(dt)[:4] + "-01-01T00:00:00")

    # Calculate and return the day of the year
    return int((dt - start_of_year) / np.timedelta64(1, 'D')) + 1
