import numpy as np
import scipy as sp
# from scipy.spatial.transform import Rotation as R
# from scipy.spacial.transform import Rotation
from scipy.spatial.transform import Rotation
from functools import partial

from geonss.algorithms import *
from geonss.constellation import *
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
    sat_pos: xr.Dataset,
    enable_tropospheric_correction: bool = True,
    enable_earth_rotation_correction: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build geometry matrix, residuals vector and weights for GNSS positioning.

    Args:
        parameters: Current state vector (position and clock bias)
        ranges: Dataset with pseudo ranges and weights
        sat_pos: Dataset with observable positions and clock biases
        enable_tropospheric_correction: Whether to apply tropospheric correction
        enable_earth_rotation_correction: Whether to apply Earth rotation correction
    Returns:
        Tuple containing geometry matrix, residuals vector, weights vector
    """
    # Extract position and clock bias from parameters
    receiver_pos = ECEFPosition.from_array(parameters[:3])
    clock_bias = parameters[3]

    # Filter to valid satellites and handle empty case
    valid_ranges = ranges.dropna(dim="sv", subset=["pseudo_range"])
    if len(valid_ranges.sv) == 0:
        return (
            np.empty((0, 4), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64)
        )

    # Extract data for valid satellites
    satellite_ranges = valid_ranges.pseudo_range.values
    weights = valid_ranges.weight.values
    valid_sat_pos = sat_pos.sel(sv=valid_ranges.sv)
    sat_clock_biases = valid_sat_pos.clock_bias.values

    # Get satellite positions and velocities
    sat_positions = valid_sat_pos[["x", "y", "z"]].to_array(dim="coord").transpose("sv", "coord").values
    sat_velocities = valid_sat_pos[["vx", "vy", "vz"]].to_array(dim="coord").transpose("sv", "coord").values

    # Calculate positions at transmission time
    signal_travel_times = satellite_ranges / SPEED_OF_LIGHT
    corrected_positions = sat_positions - (sat_velocities * signal_travel_times[:, np.newaxis])

    # Apply Earth rotation correction if enabled
    if enable_earth_rotation_correction:
        thetas = -OMEGA_E * signal_travel_times
        rotation_vectors = np.outer(thetas, np.array([0, 0, 1]))
        rotations = Rotation.from_rotvec(rotation_vectors)
        satellite_positions = rotations.apply(corrected_positions)
    else:
        satellite_positions = corrected_positions

    # Apply tropospheric correction if enabled
    if enable_tropospheric_correction:
        elevation_angles = receiver_pos.elevation_angle(satellite_positions)
        tropospheric_corrections = calculate_tropospheric_delay(elevation_angles)
    else:
        tropospheric_corrections = 0

    # Calculate geometric ranges and unit vectors
    displacements = satellite_positions - receiver_pos.array
    geometric_ranges = np.linalg.norm(displacements, axis=1)
    unit_vectors = displacements / geometric_ranges[:, np.newaxis]

    # Calculate residuals with all corrections applied
    residuals = (
        satellite_ranges
        - tropospheric_corrections
        - clock_bias
        + sat_clock_biases
        - geometric_ranges
    )

    # Create geometry matrix
    geometry_matrix = np.column_stack([
        -unit_vectors,
        np.ones(len(valid_ranges.sv))
    ])

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
    common_satellites = get_common_satellites(observation, navigation)
    observation = select_satellites(observation, common_satellites)
    navigation = select_satellites(navigation, common_satellites)

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

            # Ensure we have enough satellites with valid pseudo-ranges
            if len([r for r in ranges_t.pseudo_range.values if not np.isnan(r)]) < 4:
                raise ValueError("Not enough pseudo-ranges to compute position")

            # Find approximate position using Iterative Least Squares (ILS) without corrections
            intermediate_state = iterative_least_squares(
                initial_state=initial_state,
                model_fn=build_gnss_model,
                iterations=3,
                ranges=ranges_t,
                sat_pos=sat_pos_t,
                enable_tropospheric_correction = False,
                enable_earth_rotation_correction = False,
            )

            # Refine solution with Iterative Reweighted Least Squares (IRLS)
            final_state = iterative_reweighted_least_squares(
                initial_state=intermediate_state,
                model_fn=build_gnss_model,
                loss_fn=huber_weight,
                convergence_fn=check_gnss_convergence,
                max_iterations=3,
                damping_factor=np.float64(0.01),
                ranges=ranges_t,
                sat_pos=sat_pos_t,
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
