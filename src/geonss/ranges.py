"""
GNSS Pseudo-Range Processing Module

This module provides functionality for calculating and correcting pseudo-ranges from
GNSS observations. It includes:
- Dual-frequency ionosphere-free combination calculations
- SNR-based weighting of measurements
- Tropospheric delay corrections using Niell Mapping Function
- Consolidated pseudo-range calculation from raw observations

References:
- Niell Mapping Function: https://gssc.esa.int/navipedia/index.php/Mapping_of_Niell
- Tropospheric Model: https://gssc.esa.int/navipedia/index.php?title=Galileo_Tropospheric_Correction_Model
"""

import numpy as np
from functools import partial
from typing import Tuple
from geonss.time import datetime_to_day_of_year
from geonss.rinexmanager.util import *

logger = logging.getLogger(__name__)

# Niell Mapping Function coefficient tables
# Reference latitude values in degrees
latitudes = np.array([15, 30, 45, 60, 75])
# Hydrostatic average coefficients
a_hydrostatic_average = np.array([1.2769934e-3, 1.2683230e-3, 1.2465397e-3, 1.2196049e-3, 1.2045996e-3])
b_hydrostatic_average = np.array([2.9153695e-3, 2.9152299e-3, 2.9288445e-3, 2.9022565e-3, 2.9024912e-3])
c_hydrostatic_average = np.array([62.610505e-3, 62.837393e-3, 63.721774e-3, 63.824265e-3, 64.258455e-3])

# Hydrostatic amplitude coefficients
a_hydrostatic_amplitude = np.array([0.0, 1.2709626e-5, 2.6523662e-5, 3.4000452e-5, 4.1202191e-5])
b_hydrostatic_amplitude = np.array([0.0, 2.1414979e-5, 3.0160779e-5, 7.2562722e-5, 11.723375e-5])
c_hydrostatic_amplitude = np.array([0.0, 9.0128400e-5, 4.3497037e-5, 84.795348e-5, 170.37206e-5])

# Wet coefficients
a_wet = np.array([5.8021897e-4, 5.6794847e-4, 5.8118019e-4, 5.9727542e-4, 6.1641693e-4])
b_wet = np.array([1.4275268e-3, 1.5138625e-3, 1.4572752e-3, 1.5007428e-3, 1.7599082e-3])
c_wet = np.array([4.347296e-2, 4.672951e-2, 4.390893e-2, 4.462698e-2, 5.473603e-2])

# Height correction constants
a_ht, b_ht, c_ht = 2.53e-5, 5.49e-3, 1.14e-3

# Zenith Hydrostatic Delay (ZHD) and Zenith Wet Delay (ZWD) constants
zhd = np.float64(2.3)
zwd = np.float64(0.2)

# Create partial functions for each coefficient interpolation
interpolate_hydrostatic_avg_a = partial(np.interp, xp=latitudes, fp=a_hydrostatic_average)
interpolate_hydrostatic_avg_b = partial(np.interp, xp=latitudes, fp=b_hydrostatic_average)
interpolate_hydrostatic_avg_c = partial(np.interp, xp=latitudes, fp=c_hydrostatic_average)
interpolate_hydrostatic_amp_a = partial(np.interp, xp=latitudes, fp=a_hydrostatic_amplitude)
interpolate_hydrostatic_amp_b = partial(np.interp, xp=latitudes, fp=b_hydrostatic_amplitude)
interpolate_hydrostatic_amp_c = partial(np.interp, xp=latitudes, fp=c_hydrostatic_amplitude)
interpolate_wet_a = partial(np.interp, xp=latitudes, fp=a_wet)
interpolate_wet_b = partial(np.interp, xp=latitudes, fp=b_wet)
interpolate_wet_c = partial(np.interp, xp=latitudes, fp=c_wet)

def niell_mapping(
        time: np.datetime64,
        elevation: np.ndarray | np.float64,
        height: np.ndarray | np.float64,
        latitude: np.ndarray | np.float64
) -> (np.ndarray, np.ndarray):
    """
    Niell Mapping Function using NumPy interpolation (vectorized).

    Parameters:
        time: Time in datetime64 format
        elevation: Elevation angle(s) in radians
        height: Receiver height(s) in meters
        latitude: Receiver latitude(s) in radian

    Returns:
        Tuple: (hydrostatic mapping, wet mapping)
    """

    def interpolate_hydrostatic_coefficient(d, l, avg_fn, amp_fn):
        return avg_fn(l) - amp_fn(l) * np.cos(2 * np.pi * ((d - 28.0) / 365.25))

    def mapping(e, a, b, c):
        sin_e = np.sin(e)
        return (1 + (a / (1 + (b / (1 + c))))) / (sin_e + (a / (sin_e + (b / (sin_e + c)))))

    def delta_mapping(e, h):
        return ((1.0 / np.sin(e)) - mapping(e, a_ht, b_ht, c_ht)) * h

    day_of_year = datetime_to_day_of_year(time)
    elevation = np.asarray(elevation, dtype=np.float64)
    height = np.asarray(height, dtype=np.float64)
    latitude = np.asarray(latitude, dtype=np.float64)

    min_elevation = np.float64(0.05)  # About 3 degrees
    elevation = np.clip(elevation, min_elevation, None)

    height_km = height / 1000.0
    a_d = interpolate_hydrostatic_coefficient(day_of_year, latitude, interpolate_hydrostatic_avg_a, interpolate_hydrostatic_amp_a)
    b_d = interpolate_hydrostatic_coefficient(day_of_year, latitude, interpolate_hydrostatic_avg_b, interpolate_hydrostatic_amp_b)
    c_d = interpolate_hydrostatic_coefficient(day_of_year, latitude, interpolate_hydrostatic_avg_c, interpolate_hydrostatic_amp_c)

    m_dry = mapping(elevation, a_d, b_d, c_d) + delta_mapping(elevation, height_km)

    a_w = interpolate_wet_a(latitude)
    b_w = interpolate_wet_b(latitude)
    c_w = interpolate_wet_c(latitude)

    m_wet = mapping(elevation, a_w, b_w, c_w)

    return m_dry, m_wet

def tropospheric_delay(time, elevation, height, latitude):
    """
    Calculate tropospheric delay using Niell mapping function.

    Args:
        time: Time in datetime64 format
        elevation: Elevation angle(s) in radians
        height: Receiver height(s) in meters
        latitude: Receiver latitude(s) in radians

    Returns:
        Tropospheric delay in meters
    """
    # Calculate Niell mappings
    m_dry, m_wet = niell_mapping(time, elevation, height, latitude)

    # Calculate tropospheric delay
    return m_dry * zhd + m_wet * zwd



def apply_ionospheric_correction(
        c1c: np.ndarray,
        c5q: np.ndarray,
        s1c: np.ndarray,
        s5q: np.ndarray,
        f1: np.float64 = np.float64(1575.42),  # L1 carrier frequency in MHz
        f2: np.float64 = np.float64(1176.45)   # L5 carrier frequency in MHz
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply ionospheric correction using dual-frequency measurements.

    Uses the ionosphere-free combination to eliminate first-order ionospheric effects
    and weights the measurements based on SNR.

    This function is fully vectorized for numpy arrays and supports arbitrary frequencies,
    with defaults set to GPS L1 C/A and L5 frequencies.

    Args:
        c1c: Primary frequency code pseudo range(s) in meters
        c5q: Secondary frequency code pseudo range(s) in meters
        s1c: Primary frequency signal-to-noise ratio(s) in dB-Hz
        s5q: Secondary frequency signal-to-noise ratio(s) in dB-Hz
        f1: Primary carrier frequency in MHz (default: 1575.42 for GPS L1)
        f2: Secondary carrier frequency in MHz (default: 1176.45 for GPS L5)

    Returns:
        Tuple containing:
            - ionosphere-corrected pseudo range(s) in meters
            - signal weight(s) based on combined SNR values
    """
    # Create a mask for valid measurements
    valid_mask = ~(np.isnan(c1c) | np.isnan(c5q) | np.isnan(s1c) | np.isnan(s5q))

    # Initialize output arrays with NaN values
    corrected_range = np.full_like(c1c, np.nan)
    medium_weight = np.full_like(c1c, np.nan)

    # Calculate ionospheric-free combination parameters
    alpha = f1**2 / (f1**2 - f2**2)
    beta = -f2**2 / (f1**2 - f2**2)

    # Calculate ionospheric-free combination
    iono_free = alpha * c1c[valid_mask] + beta * c5q[valid_mask]

    # Apply SNR weighting
    w1 = 10**(s1c[valid_mask] / 10)
    w2 = 10**(s5q[valid_mask] / 10)
    weighted_range = (w1 * c1c[valid_mask] + w2 * c5q[valid_mask]) / (w1 + w2)

    # Compute corrected range and weights
    corrected_range[valid_mask] = (iono_free + weighted_range) / 2
    medium_weight[valid_mask] = (w1 + w2) / 2

    # Return ionosphere-corrected pseudo range and medium weight
    return corrected_range, medium_weight


def calculate_pseudo_ranges(
        obs_data: xr.Dataset
) -> xr.Dataset:
    """
    Calculate pseudo ranges and weights from GNSS observations.

    Args:
        obs_data: Dataset containing GNSS observations with C1C and C5Q measurements

    Returns:
        Dataset containing pseudo pseudo_ranges and SNR weights for each time and observable
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

    c1c = obs_data.C1C.values
    c5q = obs_data.C5Q.values
    s1c = obs_data.S1C.values
    s5q = obs_data.S5Q.values

    # Apply ionospheric correction
    pseudo_range, weight = apply_ionospheric_correction(c1c, c5q, s1c, s5q)

    # Store results in the dataset
    ranges['pseudo_range'].values = pseudo_range
    ranges['weight'].values = weight

    return ranges