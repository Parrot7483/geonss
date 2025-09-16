"""
Module for GNSS position estimation.

This module provides functions for estimating positions using GNSS data. It includes
functions for processing raw GNSS measurements, applying corrections, and
calculating positions using various algorithms.
"""
import logging

import xarray as xr
import numpy as np

from geonss.constants import SPEED_OF_LIGHT, OMEGA_E
from geonss.algorithms import iterative_reweighted_least_squares, huber_weight
from geonss.coordinates import ECEFPosition
from geonss.interpolation import interpolate_orbit_positions, interpolate_orbit_positions_with_antex_correction
from geonss.navigation import calculate_satellite_positions
from geonss.ranges import tropospheric_delay, calculate_pseudo_ranges

logger = logging.getLogger(__name__)


# TODO: Think about a better convergence criterion
def check_gnss_convergence(
        parameter_vector_update: np.ndarray
) -> bool:
    """
    Check if GNSS position solution has converged.

    Args:
        parameter_vector_update: Update to parameter_vector vector

    Returns:
        True if solution has converged, False otherwise
    """
    return np.linalg.norm(parameter_vector_update[:3]) < 0.01


def build_positioning_model(
        parameters: np.ndarray,
        ranges: xr.Dataset,
        satellites: xr.Dataset,
        enable_signal_travel_time_correction: bool = True,
        enable_earth_rotation_correction: bool = True,
        enable_tropospheric_correction: bool = True,
        enable_elevation_weighting: bool = True,
        enable_snr_weighting: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
    # pylint: disable=too-many-locals
    """
    Build geometry matrix, residuals vector and weights for GNSS positioning.

    Args:
        parameters: Current state vector (position, clock bias, [ISBs...])
        ranges: Dataset with pseudo ranges and weights indexed by 'sv'
        satellites: Dataset with observable positions and clock biases indexed by 'sv'
        enable_signal_travel_time_correction: Whether to apply signal travel time correction
        enable_earth_rotation_correction: Whether to apply Earth rotation correction
        enable_tropospheric_correction: Whether to apply tropospheric correction
        enable_elevation_weighting: Whether to apply elevation-based weighting
        enable_snr_weighting: Whether to apply SNR-based weighting
    Returns:
        Tuple containing geometry matrix, residuals vector, weights vector
    """
    # Initial Setup & Parameter Extraction
    time = ranges.time.values
    receiver_pos = ECEFPosition.from_array(parameters[:3])
    receiver_pos_lla = receiver_pos.to_lla()
    clock_bias = parameters[3]

    # Filter to valid satellites and align datasets
    ranges = ranges.dropna(dim="sv", subset=["pseudo_range"])
    ranges, satellites = xr.align(
        ranges, satellites, join="inner", exclude=["time"])

    # Extract data for valid satellites
    satellite_ranges = ranges.pseudo_range.values

    # Get satellite positions, velocities, and clock biases
    satellite_velocities = satellites['velocity'].values
    satellite_positions = satellites['position'].values
    satellite_clock_biases = satellites['clock'].values

    satellite_clock_biases_m = satellite_clock_biases * 1e-6 * SPEED_OF_LIGHT

    num_sats = len(ranges.sv)

    # Apply signal travel time correction if enabled
    if enable_signal_travel_time_correction:
        signal_travel_times = satellite_ranges / SPEED_OF_LIGHT
        satellite_positions -= (satellite_velocities *
                                signal_travel_times[:, np.newaxis])

    # Apply Earth rotation correction if enabled
    if enable_earth_rotation_correction:
        signal_travel_times = satellite_ranges / SPEED_OF_LIGHT
        thetas = OMEGA_E * signal_travel_times  # Rotation during travel time
        # Rotation matrix for Z-axis rotation
        cos_thetas = np.cos(thetas)
        sin_thetas = np.sin(thetas)
        # Applying rotation to each satellite position vector
        # [ cos(theta), sin(theta), 0]
        # [-sin(theta), cos(theta), 0]
        # [          0,          0, 1]
        rotated_x = cos_thetas * \
            satellite_positions[:, 0] + sin_thetas * satellite_positions[:, 1]
        rotated_y = -sin_thetas * \
            satellite_positions[:, 0] + cos_thetas * satellite_positions[:, 1]
        satellite_positions = np.column_stack(
            [rotated_x, rotated_y, satellite_positions[:, 2]])

    # Calculate elevation angles (needed for troposphere and weighting)
    elevation_angles = receiver_pos.elevation_angle(satellite_positions)

    # Apply tropospheric correction if enabled (applied to pseudo-range measurement)
    tropospheric_corrections = np.zeros(num_sats)
    if enable_tropospheric_correction:
        tropospheric_corrections = tropospheric_delay(
            time,
            elevation_angles,
            receiver_pos_lla.altitude,
            receiver_pos_lla.latitude
        )

    # Geometric Calculations
    displacements = satellite_positions - receiver_pos.array
    geometric_ranges = np.linalg.norm(displacements, axis=1)
    unit_vectors = displacements / geometric_ranges[:, np.newaxis]

    # Build Geometry Matrix
    # Base part: partial derivatives w.r.t. x, y, z, clock_bias
    geometry_matrix = np.column_stack([
        -unit_vectors,  # d(range)/dx, d(range)/dy, d(range)/dz
        np.ones(num_sats)  # d(range)/d(clock_bias)
    ])

    # Calculate Residuals
    # Residual = Observed Pseudo Range - Modeled Pseudo Range
    # Modeled Pseudo Range = Geometric Range + Receiver Clock Bias - Satellite Clock Bias + Tropo Delay + ISB
    residuals = (
        satellite_ranges
        - geometric_ranges
        - clock_bias
        + satellite_clock_biases_m
        - tropospheric_corrections
    )

    # Initialize Weights
    weights = np.ones(num_sats)

    if enable_elevation_weighting:
        # Calculate elevation-based weight
        sin_elevation_sq = np.sin(elevation_angles) ** 2

        weights *= sin_elevation_sq

    if enable_snr_weighting:
        # Get linear SNR values from the input dataset
        snr_linear = ranges.weight.values

        # Add a small floor to SNR to prevent zero weights if SNR is ever zero/negative
        snr_linear_safe = np.maximum(snr_linear, 1e-6)

        weights *= snr_linear_safe

    assert geometry_matrix.shape[1] == 4, \
        f"Geometry matrix columns mismatch. Expected 4, got {geometry_matrix.shape[1]}"
    assert geometry_matrix.shape[0] == num_sats, \
        f"Geometry matrix rows mismatch. Expected {num_sats}, got {geometry_matrix.shape[0]}"
    assert len(residuals) == num_sats, \
        f"Residuals length mismatch. Expected {num_sats}, got {len(residuals)}"
    assert len(weights) == num_sats, \
        f"Weights length mismatch. Expected {num_sats}, got {len(weights)}"

    return geometry_matrix, residuals, weights


def spp(
        observation: xr.Dataset,
        navigation: xr.Dataset | None = None,
        sp3: xr.Dataset | None = None,
        antex: xr.Dataset | None = None,
        a_priori_position: ECEFPosition = ECEFPosition(),
        a_priori_clock_bias: np.float64 = np.float64(0),
        enable_signal_travel_time_correction: bool = True,
        enable_earth_rotation_correction: bool = True,
        enable_tropospheric_correction: bool = True,
        enable_elevation_weighting: bool = True,
        enable_snr_weighting: bool = True,
) -> xr.Dataset:
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
    # pylint: disable=too-many-locals
    """
    Compute receiver positions for each time step in the observation data
    using iterative least squares method

    This function:
    1. Prepares the input data (selects common satellites)
    2. Calculates pseudo ranges and observable positions
    3. Iteratively solves for receiver position at each time step

    Args:
        observation: Dataset containing GNSS observations
        navigation: Dataset containing navigation messages (optional)
        sp3: Dataset containing precise orbit data (optional)
        antex: Dataset containing antenna phase center corrections (optional)
        a_priori_position: Initial position guess (optional)
        a_priori_clock_bias: Initial clock bias guess in meters (optional)
        enable_signal_travel_time_correction: Whether to apply signal travel time correction
        enable_earth_rotation_correction: Whether to apply Earth rotation correction
        enable_tropospheric_correction: Whether to apply tropospheric correction
        enable_elevation_weighting: Whether to apply elevation-based weighting
        enable_snr_weighting: Whether to apply SNR-based weighting

    Returns:
        xarray Dataset with receiver positions (ECEF coordinates) and clock bias for each time step
    """

    # Calculate pseudo ranges
    if sp3 and antex:
        logger.info(
            "Selecting common satellites between observation and sp3 data")
        observation, sp3 = xr.align(observation, sp3, exclude='time')

        logger.info("Calculating pseudo ranges")
        ranges = calculate_pseudo_ranges(observation)

        logger.info("Interpolating satellite positions")
        sat_pos = interpolate_orbit_positions_with_antex_correction(
            sp3, ranges, antex)

    elif navigation:
        logger.info(
            "Selecting common satellites between observation and navigation data")
        observation, navigation = xr.align(
            observation, navigation, exclude='time')

        logger.info("Calculating pseudo ranges")
        ranges = calculate_pseudo_ranges(observation)

        logger.info("Calculating satellite positions")
        sat_pos = calculate_satellite_positions(navigation, ranges)

    else:
        raise ValueError(
            "Either navigation or sp3 + antex data must be provided")

    # Compute position for each time step
    logger.info(
        "Computing receiver positions for %d time steps", len(observation.time.values)
    )

    # Create initial parameter vector (position and clock bias)
    initial_state = np.array(
        [*a_priori_position.array, a_priori_clock_bias], dtype=np.float64)

    # Prepare output arrays
    times = observation.time.values
    n_times = len(times)
    positions = np.zeros((n_times, 3), dtype=np.float64)
    clock_biases = np.zeros(n_times, dtype=np.float64)

    successful_computations = 0

    for i, t in enumerate(times):
        # Select data for this time step
        ranges_t = ranges.sel(time=t)
        sat_pos_t = sat_pos.sel(time=t)

        # Solve approximate solution with Iterative Reweighted Least Squares (IRLS)
        intermediate_state = iterative_reweighted_least_squares(
            initial_state=initial_state,
            model_fn=build_positioning_model,
            loss_fn=huber_weight,
            convergence_fn=check_gnss_convergence,
            max_iterations=3,
            ranges=ranges_t,
            satellites=sat_pos_t,
            enable_tropospheric_correction=False,
            enable_elevation_weighting=False,
            enable_snr_weighting=False,
        )

        # Solve solution with Iterative Reweighted Least Squares (IRLS)
        final_state = iterative_reweighted_least_squares(
            initial_state=intermediate_state,
            model_fn=build_positioning_model,
            loss_fn=huber_weight,
            convergence_fn=check_gnss_convergence,
            max_iterations=5,
            ranges=ranges_t,
            satellites=sat_pos_t,
            enable_signal_travel_time_correction=enable_signal_travel_time_correction,
            enable_earth_rotation_correction=enable_earth_rotation_correction,
            enable_tropospheric_correction=enable_tropospheric_correction,
            enable_elevation_weighting=enable_elevation_weighting,
            enable_snr_weighting=enable_snr_weighting,
        )

        # Store results directly in output arrays
        positions[i, :] = final_state[:3]
        clock_biases[i] = np.float64(final_state[3])
        successful_computations += 1
        logger.debug("Successfully computed position for time %s", t)

    logger.info(
        "Successfully computed %d positions out of %d time steps", successful_computations, n_times
    )

    # Convert positions from meter to km
    positions /= 1000.0

    # I want to add the sv dimension to the positions and clock_biases
    # It is a single sv, so we can just add a dimension of size 1
    # positions = positions[:, np.newaxis, :]
    # clock_biases = clock_biases[:, np.newaxis]

    # Create output xarray Dataset
    positions_ds = xr.Dataset(
        data_vars={
            "position": (["time", "sv", "ECEF"], positions[:, np.newaxis, :], {
                "units": "km",
                "long_name": "ECEF coordinates",
            }),
            "clock": (["time", "sv"], clock_biases[:, np.newaxis], {
                "units": "microseconds",
                "long_name": "Receiver clock bias",
            })
        },
        coords={
            "sv": ["UNK"],
            "time": times,
            "ECEF": ["x", "y", "z"]
        },
        attrs={
            "description": "GNSS single point positioning solution",
            "method": "Iterative Reweighted Least Squares (IRLS)",
            "created_by": "geonss.position.single_point_position",
            "data_source": "GNSS observations"
        }
    )

    # Interpolate the velocity
    result_ds = interpolate_orbit_positions(
        positions_ds,
        positions_ds,
        8,
    )

    return result_ds
