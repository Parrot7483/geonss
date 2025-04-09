import numpy as np

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
    return np.linalg.norm(parameter_vector_update) < 0.1


def build_gnss_model(
        parameter_vector: np.ndarray,
        ranges: xr.Dataset,
        sat_pos: xr.Dataset
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build geometry matrix, residuals vector and weights for GNSS positioning.

    Args:
        parameter_vector: Current state vector (position and clock bias)
        ranges: Dataset with pseudo ranges and weights
        sat_pos: Dataset with satellite positions and clock biases

    Returns:
        Tuple containing geometry matrix, residuals vector, weights vector
    """
    # Extract position and clock bias from parameter_vector
    receiver_pos_array = parameter_vector[:3]
    clock_bias = parameter_vector[3]

    # Create ECEFPosition for elevation calculation only
    receiver_pos = ECEFPosition.from_array(receiver_pos_array)

    geometry_matrix = []
    residuals = []
    weights = []

    for satellite in ranges.sv.values:
        satellite_range = ranges.pseudo_range.sel(sv=satellite).item()
        snr_weight = ranges.weight.sel(sv=satellite).item()
        pvc = sat_pos.sel(sv=satellite)
        sat_clock_bias = pvc['clock_bias'].item()

        if np.isnan(satellite_range):
            continue

        sat_coords = ECEFPosition(
            pvc['x'].item(),
            pvc['y'].item(),
            pvc['z'].item()
        )

        # Calculate signal travel time
        signal_travel_time = satellite_range / SPEED_OF_LIGHT

        # Move the satellite back (from receive time position to transmission time position)
        sat_coords.x -= pvc['vx'].item() * signal_travel_time
        sat_coords.y -= pvc['vy'].item() * signal_travel_time
        sat_coords.z -= pvc['vz'].item() * signal_travel_time

        # Apply Earth rotation correction to account for Earth's rotation during signal travel
        theta = -OMEGA_E * signal_travel_time
        sat_coords.rotate_z(theta)

        # Calculate elevation angle
        elevation_angle = receiver_pos.calculate_elevation_angle(sat_coords)

        # Apply tropospheric correction
        satellite_range -= calculate_tropospheric_delay(elevation_angle)

        # Calculate geometric range and line-of-sight vector using numpy arrays
        predicted_range = np.linalg.norm(sat_coords.array - receiver_pos.array)
        line_of_sight = (sat_coords.array - receiver_pos.array) / predicted_range

        # Calculate elevation-based weight (sin²(elevation))
        elevation_weight = np.square(np.sin(elevation_angle))

        # Combine SNR and elevation weights
        combined_weight = snr_weight * elevation_weight

        # Update geometry matrix and residuals
        residuals.append(satellite_range - (predicted_range + clock_bias - sat_clock_bias))
        geometry_matrix.append([-line_of_sight[0], -line_of_sight[1], -line_of_sight[2], 1])
        weights.append(combined_weight)

    return (
        np.array(geometry_matrix, dtype=np.float64),
        np.array(residuals, dtype=np.float64),
        np.array(weights, dtype=np.float64)
    )


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
    2. Calculates pseudo ranges and satellite positions
    3. Iteratively solves for receiver position at each time step

    Args:
        observation: Dataset containing GNSS observations
        navigation: Dataset containing navigation messages
        a_priori_position: Initial position guess (optional)
        a_priori_clock_bias: Initial clock bias guess in meters (optional)

    Returns:
        List of tuples (timestamp, position, clock_bias) for each successful time step
    """
    # Select common satellites
    logger.info(
        "Selecting common satellites between observation and navigation data")
    common_satellites = get_common_satellites(observation, navigation)
    observation = select_satellites(observation, common_satellites)
    navigation = select_satellites(navigation, common_satellites)

    # Calculate pseudo ranges
    logger.info("Calculating pseudo ranges")
    ranges = calculate_pseudo_ranges(observation)

    # Calculate satellite positions
    logger.info("Calculating satellite positions")
    sat_pos = calculate_satellite_positions(navigation, ranges)

    # Compute position for each time step
    logger.info(
        f"Computing receiver positions for {len(observation.time.values)} time steps"
    )
    positions = []

    for t in observation.time.values:
        try:
            # Select data for this time step
            ranges_t = ranges.sel(time=t)
            sat_pos_t = sat_pos.sel(time=t)

            # Ensure we have enough satellites with valid pseudo-ranges
            if len([r for r in ranges_t.pseudo_range.values if not np.isnan(r)]) < 4:
                raise ValueError("Not enough pseudo-ranges to compute position")

            # Create initial parameter vector (position and clock bias)
            initial_state = np.array([*a_priori_position.array, a_priori_clock_bias], dtype=np.float64)

            # Solve using sequential least squares
            final_state = iterative_weighted_least_squares(
                initial_state=initial_state,
                build_model_fn=build_gnss_model,
                check_convergence_fn=check_gnss_convergence,
                max_iterations=10,
                damping_factor=np.float64(0.01),
                ranges=ranges_t,
                sat_pos=sat_pos_t
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
