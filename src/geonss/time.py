# noinspection SpellCheckingInspection
"""
Time Conversion Module

This module provides utilities for handling time representations in GNSS applications.
It includes functions for converting between different time systems and formats:
- GPS time (weeks and seconds of week)
- UTC time
- Handling of leap seconds
- Conversion between different datetime representations

The module supports precise time calculations necessary for satellite positioning
and navigation applications.
"""

import datetime
import numpy as np
from typing import Union, Tuple


def datetime_gps_to_week_and_seconds(dt: Union[np.datetime64, datetime.datetime, str]) -> Tuple[int, np.float64]:
    """
    Extract GPS week and seconds of week from a datetime object that's already in GPS time.
    No UTC to GPS conversion is performed.

    Args:
        dt: Datetime object (Python or numpy) in GPS timescale

    Returns:
        Tuple[int, float]: GPS week number and seconds of week
    """

    # Convert input to numpy.datetime64 if it's not already
    if isinstance(dt, str) or isinstance(dt, datetime.datetime):
        dt = np.datetime64(dt)

    # GPS epoch (January 6, 1980)
    gps_epoch = np.datetime64("1980-01-06T00:00:00")

    # Check if time is before GPS epoch
    if dt < gps_epoch:
        raise ValueError("GPS time cannot be earlier than GPS epoch (January 6, 1980)")

    # Calculate and return the difference in seconds
    total_seconds = (dt - gps_epoch) / np.timedelta64(1, 's')

    # Compute GPS week and time of week using numpy divmod
    gps_week, seconds_of_week = np.divmod(total_seconds, 7 * 24 * 60 * 60)

    return int(gps_week), seconds_of_week

def datetime_utc_to_week_and_seconds(dt: Union[np.datetime64, datetime.datetime, str]) -> Tuple[int, np.float64]:
    """
    Convert UTC time to GPS time.

    GPS time started at 00:00:00 January 6, 1980, UTC with 0 leap seconds.
    GPS time does not include leap seconds, so the difference between
    UTC and GPS time increases with each leap second addition.

    Args:
        dt: UTC time as a datetime object, numpy datetime64 object,
                  or as an ISO format string (e.g., "2023-05-24T12:34:56Z")

    Returns:
        Tuple[int, np.float64]: GPS week number and seconds of week
    """

    # Leap seconds table (UTC - GPS) as of April 2025
    # This needs to be updated when new leap seconds are added!
    leap_seconds = [
        (np.datetime64("1980-01-06T00:00:00"), 0),
        (np.datetime64("1981-07-01T00:00:00"), 1),
        (np.datetime64("1982-07-01T00:00:00"), 2),
        (np.datetime64("1983-07-01T00:00:00"), 3),
        (np.datetime64("1985-07-01T00:00:00"), 4),
        (np.datetime64("1988-01-01T00:00:00"), 5),
        (np.datetime64("1990-01-01T00:00:00"), 6),
        (np.datetime64("1991-01-01T00:00:00"), 7),
        (np.datetime64("1992-07-01T00:00:00"), 8),
        (np.datetime64("1993-07-01T00:00:00"), 9),
        (np.datetime64("1994-07-01T00:00:00"), 10),
        (np.datetime64("1996-01-01T00:00:00"), 11),
        (np.datetime64("1997-07-01T00:00:00"), 12),
        (np.datetime64("1999-01-01T00:00:00"), 13),
        (np.datetime64("2006-01-01T00:00:00"), 14),
        (np.datetime64("2009-01-01T00:00:00"), 15),
        (np.datetime64("2012-07-01T00:00:00"), 16),
        (np.datetime64("2015-07-01T00:00:00"), 17),
        (np.datetime64("2017-01-01T00:00:00"), 18)
    ]

    # Convert input to numpy.datetime64 if it's not already
    if isinstance(dt, str) or isinstance(dt, datetime.datetime):
        dt = np.datetime64(dt)

    # GPS time epoch
    gps_epoch = np.datetime64("1980-01-06T00:00:00")

    # Check if time is before GPS epoch
    if dt < gps_epoch:
        raise ValueError("UTC time cannot be earlier than GPS epoch (January 6, 1980)")

    # Find applicable leap seconds
    leap_second_offset = 0
    for leap_date, offset in leap_seconds:
        if dt >= leap_date:
            leap_second_offset = offset

    # Create a GPS time by adding the leap seconds to UTC
    gps_time = dt + np.timedelta64(leap_second_offset, 's')

    # Use the helper function to get GPS week and seconds of week
    return datetime_gps_to_week_and_seconds(gps_time)
