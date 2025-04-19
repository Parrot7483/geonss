# noinspection SpellCheckingInspection
"""
GNSS Navigation and Satellite Positioning Module

This module provides utilities for handling ephemeris data and calculating observable positions.
It includes functions for:
- Processing GNSS navigation/ephemeris messages
- Computing observable positions in Earth-Centered Earth-Fixed (ECEF) coordinates
- Applying observable clock corrections
- Calculating orbital parameters from broadcast ephemeris
- Handling relativistic effects in observable positioning

The module implements algorithms for satellite position calculation
based on the WGS-84 reference frame and GNSS constellation parameters.
"""

from typing import Tuple
import logging

import numpy as np
import scipy as sp
import xarray as xr

from geonss.constants import *
from geonss.time import datetime_gps_to_week_and_seconds

logger = logging.getLogger(__name__)


def eccentric_anomaly(m: np.ndarray | float, e: np.ndarray | float) -> np.ndarray | float:
    """
    Calculate eccentric anomaly using Newton's method for Kepler's equation.

    Iteratively solves Kepler's equation E - e*sin(E) = M, where E is the eccentric anomaly,
    e is the orbital eccentricity, and M is the mean anomaly. Uses Newton-Raphson iteration
    with the mean anomaly as initial guess.

    Parameters:
        m (array_like): Mean anomaly in radians
        e (float or array_like): Orbital eccentricity

    Returns:
        array_like: Eccentric anomaly in radians
    """
    def kepler_equation(e_anom, mean_anom, ecc):
        """Kepler's equation: E - e*sin(E) - M = 0"""
        return e_anom - ecc * np.sin(e_anom) - mean_anom

    def kepler_equation_prime(e_anom, _, ecc):
        """Derivative of Kepler's equation: 1 - e*cos(E)"""
        return 1.0 - ecc * np.cos(e_anom)

    return sp.optimize.newton(
        kepler_equation,
        x0=m,
        fprime=kepler_equation_prime,
        args=(m, e),
        tol=1e-10,
        maxiter=100
    )

def satellite_position_velocity_clock_correction(
        ephemeris: xr.Dataset,
        dt: np.datetime64
) -> Tuple[np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64]:
    """
    Function to compute satellite position (ECEF), velocity (ECEF),
    and clock correction using individual ephemeris parameters.

    Parameters:
        ephemeris (xarray.Dataset): Dataset containing broadcast ephemeris parameters
        dt (numpy.datetime64): Time for which to compute the satellite position

    Returns:
        Tuple[np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64]:
        Satellite ECEF coordinates (x, y, z) in meters,
        Satellite velocity components (vx, vy, vz) in meters/second,
        Clock correction (delta_t_sv_m) in meters
    """
    t_oe = ephemeris['Toe'].item()
    sqrt_a = ephemeris['sqrtA'].item()
    delta_n = ephemeris['DeltaN'].item()
    m0 = ephemeris['M0'].item()
    e = ephemeris['Eccentricity'].item()
    omega = ephemeris['omega'].item()
    i0 = ephemeris['Io'].item()
    i_dot = ephemeris['IDOT'].item()
    omega0 = ephemeris['Omega0'].item()
    omega_dot = ephemeris['OmegaDot'].item()

    # TODO: Add check that dt is in the same week as t_oe
    _, gps_seconds = datetime_gps_to_week_and_seconds(dt)

    t_k = gps_seconds - t_oe

    # Handle week crossover
    t_k = np.where(t_k > np.float64(302400), t_k - np.float64(604800),
        np.where(t_k < np.float64(-302400), t_k + np.float64(604800), t_k))

    # Convert sqrt_a to a
    a = np.square(sqrt_a)

    # Mean motion
    n0 = np.sqrt(GM / np.power(a, 3))
    n = n0 + delta_n

    # Mean anomaly
    m = m0 + n * t_k

    # Eccentric anomaly
    e_anom = eccentric_anomaly(m, e)

    # True anomaly v
    sin_e = np.sin(e_anom)
    cos_e = np.cos(e_anom)
    v = np.arctan2(
        np.sqrt(1 - np.square(e)) * sin_e,
        cos_e - e
    )

    # Argument of latitude (phi) - the angle from the ascending node to the observable position
    phi = v + omega

    # Calculate sine and cosine of 2 * phi for harmonic correction terms
    sin_2phi = np.sin(2 * phi)
    cos_2phi = np.cos(2 * phi)

    # Extract correction coefficients for argument of latitude (u), radius (r), and inclination (i)
    c_uc = ephemeris['Cuc'].item()
    c_us = ephemeris['Cus'].item()
    c_rc = ephemeris['Crc'].item()
    c_rs = ephemeris['Crs'].item()
    c_ic = ephemeris['Cic'].item()
    c_is = ephemeris['Cis'].item()

    # Calculate correction terms using second harmonic perturbations
    delta_u = c_us * sin_2phi + c_uc * cos_2phi
    delta_r = c_rs * sin_2phi + c_rc * cos_2phi
    delta_i = c_is * sin_2phi + c_ic * cos_2phi

    # Apply corrections to obtain the final orbital parameters
    u = phi + delta_u
    r = a * (1.0 - e * cos_e) + delta_r
    i = i0 + i_dot * t_k + delta_i

    # Positions in orbital plane
    x_prime = r * np.cos(u)
    y_prime = r * np.sin(u)

    # Corrected longitude of ascending node
    omega_k = omega0 + (omega_dot - OMEGA_E) * t_k - OMEGA_E * t_oe

    # ECEF coordinates
    cos_omega_k = np.cos(omega_k)
    sin_omega_k = np.sin(omega_k)
    cos_i = np.cos(i)
    sin_i = np.sin(i)

    x = x_prime * cos_omega_k - y_prime * cos_i * sin_omega_k
    y = x_prime * sin_omega_k + y_prime * cos_i * cos_omega_k
    z = y_prime * sin_i

    # Time derivative of the eccentric anomaly
    e_anom_dot = n / (1.0 - e * cos_e)

    # Time derivative of the argument of latitude
    phi_dot = np.sqrt(1.0 - np.square(e)) / (1.0 - e * cos_e) * e_anom_dot

    # Time derivatives of the correction terms
    delta_u_dot = 2.0 * (c_us * cos_2phi - c_uc * sin_2phi) * phi_dot
    delta_r_dot = 2.0 * (c_rs * cos_2phi - c_rc * sin_2phi) * phi_dot

    # Time derivatives of radius and argument of latitude
    u_dot = phi_dot + delta_u_dot
    r_dot = a * e * sin_e * e_anom_dot + delta_r_dot

    # Time derivatives in the orbital plane
    x_prime_dot = r_dot * np.cos(u) - r * np.sin(u) * u_dot
    y_prime_dot = r_dot * np.sin(u) + r * np.cos(u) * u_dot

    # Account for the earth's rotation and time derivative of the orbit inclination
    omega_k_dot = omega_dot - OMEGA_E

    # ECEF velocity components
    vx = x_prime_dot * cos_omega_k - y_prime_dot * cos_i * sin_omega_k - \
         (x_prime * sin_omega_k + y_prime * cos_i * cos_omega_k) * omega_k_dot + \
         y_prime_dot * sin_i * sin_omega_k * i_dot

    vy = x_prime_dot * sin_omega_k + y_prime_dot * cos_i * cos_omega_k + \
         (x_prime * cos_omega_k - y_prime * cos_i * sin_omega_k) * omega_k_dot - \
         y_prime_dot * sin_i * cos_omega_k * i_dot

    vz = y_prime_dot * sin_i + y_prime * cos_i * i_dot

    # Extract observable clock parameters from ephemeris data
    t_oc = ephemeris.time.values
    a0 = ephemeris['SVclockBias'].item()
    a1 = ephemeris['SVclockDrift'].item()
    a2 = ephemeris['SVclockDriftRate'].item()

    # Convert the ephemeris reference time to GPS week and seconds of week
    _, seconds_of_clock = datetime_gps_to_week_and_seconds(t_oc)

    # Relativistic clock correction
    delta_tr = REL_CONST * np.power(e, sqrt_a) * sin_e

    # Satellite clock correction in meter
    delta_t_oc = gps_seconds - seconds_of_clock
    delta_t_sv = a0 + a1 * delta_t_oc + a2 * np.square(delta_t_oc) + delta_tr
    delta_t_sv_m = delta_t_sv * SPEED_OF_LIGHT

    return x, y, z, vx, vy, vz, delta_t_sv_m


def calculate_satellite_positions(
        nav_data: xr.Dataset,
        ranges: xr.Dataset
) -> xr.Dataset:
    """
    Calculate observable positions and clock biases for each observation.

    This function determines observable positions at signal reception time.

    Args:
        nav_data: Navigation messages dataset containing ephemeris data
        ranges: Dataset with pseudo ranges and time/observable coordinates

    Returns:
        Dataset containing observable positions (x, y, z) coordinates and
        clock bias for each time and observable
    """
    logger.info(
        f"Starting to compute {len(ranges.sv.values)} observable positions for {len(ranges.time.values)} time steps")

    # Initialize result dataset with time and sv coordinates from pseudo ranges
    result = xr.Dataset(
        coords={
            'time': ranges.time,
            'sv': ranges.sv
        }
    )

    # Initialize data variables
    result['x'] = xr.DataArray(
        dims=['time', 'sv'],
        coords={'time': ranges.time, 'sv': ranges.sv},
        attrs={'long_name': 'X Position', 'units': 'meter'}
    )

    result['y'] = xr.DataArray(
        dims=['time', 'sv'],
        coords={'time': ranges.time, 'sv': ranges.sv},
        attrs={'long_name': 'Y Position', 'units': 'meter'}
    )

    result['z'] = xr.DataArray(
        dims=['time', 'sv'],
        coords={'time': ranges.time, 'sv': ranges.sv},
        attrs={'long_name': 'Z Position', 'units': 'meter'}
    )

    result['vx'] = xr.DataArray(
        dims=['time', 'sv'],
        coords={'time': ranges.time, 'sv': ranges.sv},
        attrs={'long_name': 'X Velocity', 'units': 'meter / second'}
    )

    result['vy'] = xr.DataArray(
        dims=['time', 'sv'],
        coords={'time': ranges.time, 'sv': ranges.sv},
        attrs={'long_name': 'Y Velocity', 'units': 'meter / second'}
    )

    result['vz'] = xr.DataArray(
        dims=['time', 'sv'],
        coords={'time': ranges.time, 'sv': ranges.sv},
        attrs={'long_name': 'Z Velocity', 'units': 'meter / second'}
    )

    result['clock_bias'] = xr.DataArray(
        dims=['time', 'sv'],
        coords={'time': ranges.time, 'sv': ranges.sv},
        attrs={'long_name': 'Clock Bias', 'units': 'second'}
    )

    for satellite in ranges.sv.values:
        nav_data_sat = nav_data.sel(sv=satellite).dropna(dim='time', how='all')
        ranges_sv = ranges.sel(sv=satellite).dropna(dim='time', how='all')

        for dt in ranges_sv.time.values:
            ephemeris = nav_data_sat.sel(time=dt, method='nearest')

            # Calculate observable position and clock bias at transmission time
            x, y, z, vx, vy, vz, clock_bias = satellite_position_velocity_clock_correction(
                ephemeris, dt)

            # Store results in dataset
            result['x'].loc[dt, satellite] = x
            result['y'].loc[dt, satellite] = y
            result['z'].loc[dt, satellite] = z
            result['vx'].loc[dt, satellite] = vx
            result['vy'].loc[dt, satellite] = vy
            result['vz'].loc[dt, satellite] = vz
            result['clock_bias'].loc[dt, satellite] = clock_bias

        logger.info(f"Finished computing positions for {satellite}")

    return result