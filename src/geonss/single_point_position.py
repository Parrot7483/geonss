import xarray

from geonss.constellation import get_constellation
from geonss.algorithms import *
from geonss.coordinates import *
from geonss.navigation import *
from geonss.ranges import *
from geonss.rinexmanager.util import *

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


def build_gnss_model(
        parameters: np.ndarray,
        ranges: xr.Dataset,
        satellites: xr.Dataset,
        enable_signal_travel_time_correction: bool = True,
        enable_earth_rotation_correction: bool = True,
        enable_tropospheric_correction: bool = True,
        enable_iter_system_bias_correction: bool = True,
        enable_elevation_weighting: bool = True,
        enable_snr_weighting: bool = True,
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Build geometry matrix, residuals vector and weights for GNSS positioning.

    Args:
        parameters: Current state vector (position, clock bias, [ISBs...])
        ranges: Dataset with pseudo ranges and weights indexed by 'sv'
        satellites: Dataset with observable positions and clock biases indexed by 'sv'
        enable_signal_travel_time_correction: Whether to apply signal travel time correction
        enable_earth_rotation_correction: Whether to apply Earth rotation correction
        enable_tropospheric_correction: Whether to apply tropospheric correction
        enable_iter_system_bias_correction: Whether to apply inter-system bias correction
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
    ranges, satellites = xarray.align(ranges, satellites, join="inner", exclude=["time"])

    # Extract data for valid satellites
    satellite_ranges = ranges.pseudo_range.values
    sv_ids = ranges.sv.values

    # Get satellite positions, velocities, and clock biases
    satellite_velocities = satellites[["vx", "vy", "vz"]].to_array(dim="coord").transpose("sv", "coord").values
    satellite_positions = satellites[["x", "y", "z"]].to_array(dim="coord").transpose("sv", "coord").values
    satellite_clock_biases = satellites.clock_bias.values

    # Inter-System Bias (ISB) Setup (if enabled)
    sv_constellations = np.array([get_constellation(sv) for sv in sv_ids])
    unique_constellations = np.unique(sv_constellations)
    num_unique_constellations = len(unique_constellations)

    num_sats = len(ranges.sv)
    isb_corrections = np.zeros(num_sats) # Initialize ISB corrections with zeros
    isb_geometry = np.empty((num_sats, 0)) # Initialize as empty

    if enable_iter_system_bias_correction:
        assert len(parameters) == 4 + num_unique_constellations - 1, \
            f"Invalid parameter vector length. Expected {num_unique_constellations + 3} (3 pos + 1 clock + {num_unique_constellations - 1} ISBs), got {len(parameters)}"
    else:
        assert len(parameters) == 4, \
            f"Invalid parameter vector length. Expected 4 parameters (pos + clock) when ISB disabled, got {len(parameters)}"

    if enable_iter_system_bias_correction and num_unique_constellations > 1:
        non_ref_constellations = unique_constellations[1:]
        num_isb_params = len(non_ref_constellations)

        # Map non-reference constellations to their parameter index (starting from 4)
        isb_param_indices = {const: i + 4 for i, const in enumerate(non_ref_constellations)}
        # Map non-reference constellations to their column index in the ISB part of the geometry matrix
        isb_col_indices = {const: i for i, const in enumerate(non_ref_constellations)}

        # Calculate ISB correction values for residuals
        for const in non_ref_constellations:
            mask = (sv_constellations == const)
            isb_value = parameters[isb_param_indices[const]]
            isb_corrections[mask] = isb_value # ISB is subtracted from residual later

        # Build the ISB part of the geometry matrix
        isb_geometry = np.zeros((num_sats, num_isb_params))
        for const in non_ref_constellations:
            mask = (sv_constellations == const)
            col_idx = isb_col_indices[const]
            isb_geometry[mask, col_idx] = 1.0 # Coefficient for ISB parameter is 1

    # Calculate elevation angles (needed for troposphere and weighting)
    # Use initial satellite positions before time-of-flight corrections for elevation
    elevation_angles = receiver_pos.elevation_angle(
        satellites[["x", "y", "z"]].to_array(dim="coord").transpose("sv", "coord").values
    )

    assert 0 <= elevation_angles <= np.pi / 2, \
        f"Elevation angles out of range. Expected [0, pi/2], got {elevation_angles}"

    # Apply signal travel time correction if enabled
    if enable_signal_travel_time_correction:
        signal_travel_times = satellite_ranges / SPEED_OF_LIGHT
        satellite_positions -= (satellite_velocities * signal_travel_times[:, np.newaxis])

    # Apply Earth rotation correction if enabled
    if enable_earth_rotation_correction:
        signal_travel_times = satellite_ranges / SPEED_OF_LIGHT
        thetas = OMEGA_E * signal_travel_times # Rotation during travel time
        # Rotation matrix for Z-axis rotation
        cos_thetas = np.cos(thetas)
        sin_thetas = np.sin(thetas)
        # Applying rotation to each satellite position vector
        # [ cos(theta), sin(theta), 0]
        # [-sin(theta), cos(theta), 0]
        # [          0,          0, 1]
        rotated_x = cos_thetas * satellite_positions[:, 0] + sin_thetas * satellite_positions[:, 1]
        rotated_y = -sin_thetas * satellite_positions[:, 0] + cos_thetas * satellite_positions[:, 1]
        satellite_positions = np.column_stack([rotated_x, rotated_y, satellite_positions[:, 2]])

    # Apply tropospheric correction if enabled (applied to pseudo-range measurement)
    tropospheric_corrections = np.zeros(num_sats)
    if enable_tropospheric_correction:
        tropospheric_corrections = tropospheric_delay(
            time, # Assuming time is compatible or single value
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
    base_geometry_matrix = np.column_stack([
        -unit_vectors,                 # d(range)/dx, d(range)/dy, d(range)/dz
        np.ones(num_sats)              # d(range)/d(clock_bias)
    ])

    # Combine base geometry with ISB geometry (if enabled)
    geometry_matrix = np.hstack((base_geometry_matrix, isb_geometry))

    # Calculate Residuals
    # Residual = Observed Pseudo Range - Modeled Pseudo Range
    # Modeled Pseudo Range = Geometric Range + Receiver Clock Bias - Satellite Clock Bias + Tropo Delay + ISB
    residuals = (
        satellite_ranges
        - geometric_ranges
        - clock_bias
        + satellite_clock_biases
        - tropospheric_corrections
        - isb_corrections
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

    # Final Assertions
    if enable_iter_system_bias_correction:
        assert geometry_matrix.shape[1] == 3 + len(unique_constellations), \
            f"Geometry matrix columns mismatch. Expected {3 + len(unique_constellations)}, got {geometry_matrix.shape[1]}"
    else:
        assert geometry_matrix.shape[1] == 4, \
            f"Geometry matrix columns mismatch. Expected 4, got {geometry_matrix.shape[1]}"
    assert geometry_matrix.shape[0] == num_sats, \
        f"Geometry matrix rows mismatch. Expected {num_sats}, got {geometry_matrix.shape[0]}"
    assert len(residuals) == num_sats, \
        f"Residuals length mismatch. Expected {num_sats}, got {len(residuals)}"
    assert len(weights) == num_sats, \
        f"Weights length mismatch. Expected {num_sats}, got {len(weights)}"

    return geometry_matrix, residuals, weights


def prepare_ranges_satellite_positions(observation: xr.Dataset, navigation: xr.Dataset) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Prepare data by selecting common satellites and calculating ranges and positions.

    Args:
        observation: Dataset containing GNSS observations
        navigation: Dataset containing navigation messages

    Returns:
        Tuple of (ranges, observable positions)
    """
    # Select common satellites
    logger.info("Selecting common satellites between observation and navigation data")
    observation, navigation = xarray.align(observation, navigation, exclude='time')

    # Calculate pseudo ranges
    logger.info("Calculating pseudo ranges")
    ranges = calculate_pseudo_ranges(observation)

    # Calculate observable positions
    logger.info("Calculating observable positions")
    sat_pos = calculate_satellite_positions(navigation, ranges)

    return ranges, sat_pos


def single_point_position(
        observation: xr.Dataset,
        navigation: xr.Dataset,
        a_priori_position: ECEFPosition = ECEFPosition(),
        a_priori_clock_bias: Optional[np.float64] = np.float64(0)
) -> List[Tuple[np.datetime64, ECEFPosition, np.float64]]:
    """
    Compute receiver positions for each time step in the observation data
    using iterative least squares method

    This function:
    1. Prepares the input data (selects common satellites)
    2. Calculates pseudo ranges and observable positions
    3. Iteratively solves for receiver position at each time step

    Args:
        observation: Dataset containing GNSS observations
        navigation: Dataset containing navigation messages
        a_priori_position: Initial position guess (optional)
        a_priori_clock_bias: Initial clock bias guess in meters (optional)

    Returns:
        List of tuples (timestamp, position, clock_bias) for each successful time step
    """
    ranges, sat_pos = prepare_ranges_satellite_positions(observation, navigation)

    # Compute position for each time step
    logger.info(
        f"Computing receiver positions for {len(observation.time.values)} time steps"
    )

    # Create initial parameter vector (position and clock bias)
    initial_state = np.array([*a_priori_position.array, a_priori_clock_bias], dtype=np.float64)

    positions = []

    for t in observation.time.values:
        try:
            # Select data for this time step
            ranges_t = ranges.sel(time=t)
            sat_pos_t = sat_pos.sel(time=t)

            # Find approximate position using Iterative Least Squares (ILS) without corrections
            intermediate_state = iterative_least_squares(
                initial_state=initial_state,
                model_fn=build_gnss_model,
                iterations=5,
                ranges=ranges_t,
                satellites=sat_pos_t,
                enable_tropospheric_correction = False,
                enable_iter_system_bias_correction = False,
            )

            # Add a zero to intermedia state for inter system bias
            intermediate_state = np.append(intermediate_state, 0)

            huber_weight_fn = huber_weight
            huber_weight_fn = partial(huber_weight, k=0.7)

            # Refine solution with Iterative Reweighted Least Squares (IRLS)
            final_state = iterative_reweighted_least_squares(
                initial_state=intermediate_state,
                model_fn=build_gnss_model,
                loss_fn=huber_weight_fn,
                convergence_fn=check_gnss_convergence,
                damping_factor=np.float64(0.01),
                max_iterations=5,
                ranges=ranges_t,
                satellites=sat_pos_t,
            )

            # Convert final parameter_vector back to expected return format
            position = ECEFPosition.from_array(final_state[:3])
            clock_bias = np.float64(final_state[3])

            positions.append((t, position, clock_bias))
            logger.debug(f"Successfully computed position for time {t}")

        except Exception as e:
            logger.warning(f"Error computing position at {t}: {e}")

    logger.info(
        f"Successfully computed "
        f"{len(positions)} positions out of "
        f"{len(observation.time.values)} time steps"
    )

    return positions
