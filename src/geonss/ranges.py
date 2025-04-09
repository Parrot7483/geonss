"""
GNSS Pseudo-Range Processing Module

This module provides functionality for calculating and correcting pseudo-ranges from
GNSS observations. It includes:
- Dual-frequency ionosphere-free combination calculations
- SNR-based weighting of measurements
- Tropospheric delay corrections
- Consolidated pseudo-range calculation from raw observations
"""
import numpy as np
from typing import Tuple

from geonss.rinexmanager.util import *

logger = logging.getLogger(__name__)


def apply_ionospheric_correction(
        c1c: np.float64,
        c5q: np.float64,
        s1c: np.float64,
        s5q: np.float64
) -> Tuple[np.float64, np.float64]:
    """
    Apply ionospheric correction using dual-frequency measurements.

    Uses the ionosphere-free combination to eliminate first-order ionospheric effects
    and weights the measurements based on SNR.

    Args:
        c1c: L1 C/A code pseudo range in meters
        c5q: L5 Q code pseudo range in meters
        s1c: L1 C/A signal-to-noise ratio in dB-Hz
        s5q: L5 Q signal-to-noise ratio in dB-Hz

    Returns:
        Tuple containing:
            - ionosphere-corrected pseudo range in meters
            - signal weight based on combined SNR values
    """
    # Skip if any measurements are missing
    if np.isnan(c1c) or np.isnan(c5q) or np.isnan(s1c) or np.isnan(s5q):
        return np.float64(np.nan), np.float64(np.nan)

    # L1 and L5 carrier frequencies in MHz
    f1 = 1575.42
    f5 = 1176.45

    # Calculate ionospheric-free combination
    alpha = f1 ** 2 / (f1 ** 2 - f5 ** 2)
    beta = -f5 ** 2 / (f1 ** 2 - f5 ** 2)
    iono_free = alpha * c1c + beta * c5q

    # Apply SNR weighting
    w1 = 10 ** (s1c / 10)
    w5 = 10 ** (s5q / 10)
    weighted_range = (w1 * c1c + w5 * c5q) / (w1 + w5)
    corrected_range = (iono_free + weighted_range) / 2
    medium_weight = (w1 + w5) / 2

    # Return ionospheric-free pseudo range and medium weight
    return corrected_range, medium_weight


def calculate_tropospheric_delay(
    elevation_angle_rad: np.float64
) -> np.float64:
    """
    Calculate tropospheric delay based on elevation angle using
    the Niell mapping function for the hydrostatic component.

    This is a simplified model that uses only the elevation angle and standard
    atmospheric conditions.

    Args:
        elevation_angle_rad: Satellite elevation angle in radians

    Returns:
        Tropospheric delay in meters
    """
    # Ensure minimum elevation to avoid division by zero
    if elevation_angle_rad <= np.float64(0.05):  # About 3 degrees
        elevation_angle_rad = np.float64(0.05)

    # Calculate sine of elevation
    sin_elev = np.sin(elevation_angle_rad)

    # Parameters for simplified Niell hydrostatic mapping function
    a = np.float64(1.2769934e-3)
    b = np.float64(2.9153695e-3)
    c = np.float64(62.610505e-3)

    # Hydrostatic mapping function
    m_h = np.float64(1.0) / (sin_elev + (a / (sin_elev + b / (sin_elev + c))))

    # Approximate zenith hydrostatic delay (ZHD) at standard conditions
    # Using average value of approximately 2.3 meters at sea level
    zhd = np.float64(2.3)

    # Calculate tropospheric delay
    tropospheric_delay = zhd * m_h

    return tropospheric_delay


# TODO: This can be parallelized
def calculate_pseudo_ranges(
        obs_data: xr.Dataset
) -> xr.Dataset:
    """
    Calculate pseudo ranges and weights from GNSS observations.

    Args:
        obs_data: Dataset containing GNSS observations with C1C and C5Q measurements

    Returns:
        Dataset containing pseudo pseudo_ranges and SNR weights for each time and satellite
    """
    ranges = xr.Dataset(
        coords={
            'time': obs_data.time,
            'sv': obs_data.sv
        }
    )

    ranges['pseudo_range'] = xr.DataArray(
        dims=['time', 'sv'],
        coords={'time': obs_data.time, 'sv': obs_data.sv},
        attrs={'long_name': 'Corrected Pseudo-Range', 'units': 'meter'}
    )

    ranges['weight'] = xr.DataArray(
        dims=['time', 'sv'],
        coords={'time': obs_data.time, 'sv': obs_data.sv},
        attrs={'long_name': 'Signal Weight', 'units': ''}
    )

    # TODO: Right now we drop any Satellite that does not support dual frequency measurements
    # Maybe we should use single frequency measurements as well and weight
    # them accordingly
    for timestamp in obs_data.time.values:
        for satellite in obs_data.sv.values:
            data_slice = obs_data.sel(time=timestamp, sv=satellite)

            # Extract measurements
            c1c = data_slice.C1C.item()
            c5q = data_slice.C5Q.item()
            s1c = data_slice.S1C.item()
            s5q = data_slice.S5Q.item()

            pseudo_range, weight = apply_ionospheric_correction(c1c, c5q, s1c, s5q)

            # TODO: Maybe also use single frequency
            if np.isnan(pseudo_range):
                continue

            ranges['pseudo_range'].loc[dict(
                time=timestamp, sv=satellite)] = pseudo_range
            ranges['weight'].loc[dict(time=timestamp, sv=satellite)] = weight

    return ranges