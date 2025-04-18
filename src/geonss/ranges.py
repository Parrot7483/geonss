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

def ionospheric_correction(
        c1: np.float64 | np.ndarray | xr.DataArray,
        c5: np.float64 | np.ndarray | xr.DataArray,
        s1: np.float64 | np.ndarray | xr.DataArray,
        s5: np.float64 | np.ndarray | xr.DataArray,
        f1: np.float64 | np.ndarray | xr.DataArray = np.float64(1575.42), # L1 / E1 frequency in MHz
        f5: np.float64 | np.ndarray | xr.DataArray = np.float64(1176.45), # L5 / E5a frequency in MHz
) -> (np.float64 | np.ndarray | xr.DataArray, np.float64 | np.ndarray | xr.DataArray):
    """
    Ionospheric correction using the dual-frequency ionosphere-free combination.

    Calculates the ionosphere-free pseudo-range and determines its weight based
    on the SNR of the input measurements, assuming variance is inversely
    proportional to linear SNR. Handles multidimensional NumPy arrays or
    xarray DataArrays. Preserves coordinates and attributes for xarray inputs.

    Args:
        c1: Primary frequency code pseudo range(s) in meters
        c5: Secondary frequency code pseudo range(s) in meters
        s1: Primary frequency signal-to-noise ratio(s) in dB-Hz
        s5: Secondary frequency signal-to-noise ratio(s) in dB-Hz
        f1: Primary carrier frequency in MHz (default: 1575.42 for GPS L1).
            Can be scalar, NumPy array, or xarray DataArray.
        f5: Secondary carrier frequency in MHz (default: 1176.45 for GPS L5).
            Can be scalar, NumPy array, or xarray DataArray.

    Returns:
        Tuple containing:
            - ionosphere-corrected pseudo range(s) (ionosphere-free combination)
              in meters (same type as input c1).
            - signal weight(s) reflecting the estimated precision of the
              corrected range (same type as input c1).
    """

    # calculate ionospheric-free combination parameters
    f1_sq = f1**2
    f2_sq = f5**2
    denominator = f1_sq - f2_sq

    alpha = f1_sq / denominator
    beta = -f2_sq / denominator

    # Calculate ionosphere-free combination
    iono_free = alpha * c1 + beta * c5

    # Calculate linear SNR for weighting
    w1 = 10**(s1 / 10.0)
    w2 = 10**(s5 / 10.0)

    # Calculate the variance term for the weight
    epsilon = 1e-12 # Small value to avoid division by exactly zero SNR
    variance_term = (alpha**2 / xr.ufuncs.maximum(w1, epsilon)) + \
                    (beta**2 / xr.ufuncs.maximum(w2, epsilon))

    # Calculate weight = 1 / variance
    weight = xr.where(variance_term > 0, 1.0 / variance_term, np.nan)

    # Create a mask for valid inputs (non-NaN in original data)
    valid_input_mask = ~(xr.ufuncs.isnan(c1) | xr.ufuncs.isnan(c5) |
                         xr.ufuncs.isnan(s1) | xr.ufuncs.isnan(s5))

    # Create a mask for valid calculated weights (finite and positive)
    valid_weight_mask = xr.ufuncs.isfinite(weight) & (weight > 0) & xr.ufuncs.isfinite(variance_term) & (variance_term > 0)

    # Also ensure the iono_free calculation itself is finite
    valid_iono_mask = xr.ufuncs.isfinite(iono_free)

    # Final results are valid only where inputs AND calculations are valid
    final_valid_mask = valid_input_mask & valid_weight_mask & valid_iono_mask

    # Assign the calculated values to the output arrays only where the final mask is True
    corrected_range_final = xr.where(final_valid_mask, iono_free, np.nan)
    combined_weight_final = xr.where(final_valid_mask, weight, np.nan)

    # Add metadata if using xarray
    if isinstance(c1, xr.DataArray):
        corrected_range_final = corrected_range_final.rename("iono_free_range")
        corrected_range_final.attrs['units'] = 'm'
        corrected_range_final.attrs['long_name'] = 'Ionosphere-free pseudo-range combination'
        corrected_range_final.attrs['formula'] = '(f1^2*C1 - f5^2*C5) / (f1^2 - f5^2)'
        corrected_range_final.attrs['input_vars'] = f"c1={c1.name}, c5={c5.name}, s1={s1.name}, s5={s5.name}"

        combined_weight_final = combined_weight_final.rename("iono_free_weight")
        combined_weight_final.attrs['units'] = 'unitless'
        combined_weight_final.attrs['long_name'] = 'Weight of ionosphere-free pseudo-range combination'
        combined_weight_final.attrs['formula'] = '1 / ( (alpha^2/SNR1_lin) + (beta^2/SNR5_lin) )'
        combined_weight_final.attrs['input_vars'] = f"c1={c1.name}, c5={c5.name}, s1={s1.name}, s5={s5.name}"

    # Return ionosphere-corrected pseudo range and its weight
    return corrected_range_final, combined_weight_final


def calculate_pseudo_ranges(obs_data: xr.Dataset) -> xr.Dataset:
    """
    Calculate ionosphere-free pseudo ranges and weights for GPS and Galileo.

    Args:
        obs_data: Dataset containing GNSS observations with 'sv' coordinate
                  and relevant RINEX observation code variables.

    Returns:
        Dataset containing iono-free 'pseudo_range' and 'weight',
        with NaNs for non-G/E/C constellations or where correction failed.
    """
    c1 = obs_data.C1C
    s2 = obs_data.S1C
    c5 = obs_data.C5Q
    s5 = obs_data.S5Q

    ranges, weights = ionospheric_correction(c1, c5, s2, s5)

    # Create a new dataset to store the results
    result = xr.Dataset(
        {
            'pseudo_range': ranges,
            'weight': weights
        },
        coords={
            'time': ranges.time,
            'sv': ranges.sv
        }
    )

    return result
