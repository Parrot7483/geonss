# pylint: disable=missing-module-docstring
import logging

import numpy as np

logger = logging.getLogger(__name__)


def split_time_range(start_time: np.datetime64, end_time: np.datetime64, n_parts):
    """
    Split a time range into n equal parts.

    Parameters:
        start_time (numpy.datetime64): Start time
        end_time (numpy.datetime64): End time
        n_parts (int): Number of parts to split the range into

    Returns:
        list: List of numpy.datetime64 timestamps including start and end times
    """
    # Convert to datetime64[ns] for consistency
    start = np.datetime64(start_time, 'ns')
    end = np.datetime64(end_time, 'ns')

    # Calculate the time delta in nanoseconds
    delta_ns = (end - start).astype(np.int64)

    # Calculate the step size in nanoseconds
    step_ns = delta_ns // n_parts

    # Generate the timestamps
    timestamps = [start + np.timedelta64(i * step_ns, 'ns') for i in range(n_parts)]
    # Add the end time
    timestamps.append(end)

    return timestamps
