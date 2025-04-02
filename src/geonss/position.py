# noinspection SpellCheckingInspection
"""
GNSS Position Determination Module

This module provides functionality for calculating precise receiver positions from GNSS observations.
It includes:
- Satellite position computation based on ephemeris data
- Earth rotation corrections for signal travel time
- Weighted least squares solver for position estimation
- Iterative positioning algorithms

The module implements standard GNSS positioning techniques including ionosphere-free combinations,
signal weighting based on SNR, and satellite position interpolation to handle transmission time effects.
"""
import numpy as np

from geonss.algorithms import *
from geonss.constellation import *
from geonss.coordinates import *
from geonss.navigation import *
from geonss.ranges import *
from geonss.rinexmanager.util import *

logger = logging.getLogger(__name__)


# TODO: Think about a better convergence criterion
def check_gnss_convergence(parameter_vector_update: np.ndarray) -> bool:
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

        if np.isnan(satellite_range):
            continue

        sat_coords = ECEFPosition(
            sat_pos.sel(sv=satellite)['x'].item(),
            sat_pos.sel(sv=satellite)['y'].item(),
            sat_pos.sel(sv=satellite)['z'].item()
        )

        sat_clock_bias = sat_pos.sel(sv=satellite)['clock_bias'].item() * SPEED_OF_LIGHT

        # Calculate signal travel time and apply Earth rotation correction
        tau = (satellite_range - clock_bias + sat_clock_bias) / SPEED_OF_LIGHT
        theta = -OMEGA_E * tau
        sat_coords.rotate_z(theta)

        # Calculate elevation angle (requires ECEFPosition)
        elevation_angle = receiver_pos.calculate_elevation_angle(sat_coords)

        # Apply tropospheric correction
        corrected_range = apply_tropospheric_correction(satellite_range, elevation_angle)

        # Calculate geometric range and line-of-sight vector using numpy arrays
        predicted_range = np.linalg.norm(sat_coords.array - receiver_pos_array)
        line_of_sight = (sat_coords.array - receiver_pos_array) / predicted_range

        # Calculate elevation-based weight (sin²(elevation))
        elevation_weight = np.square(np.sin(elevation_angle))

        # Combine SNR and elevation weights
        combined_weight = snr_weight * elevation_weight

        # Update geometry matrix and residuals
        residuals.append(corrected_range - (predicted_range + clock_bias - sat_clock_bias))
        geometry_matrix.append([-line_of_sight[0], -line_of_sight[1], -line_of_sight[2], 1])
        weights.append(combined_weight)

    return (
        np.array(geometry_matrix, dtype=np.float64),
        np.array(residuals, dtype=np.float64),
        np.array(weights, dtype=np.float64)
    )


def solve_position_solution(
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

            # Create initial parameter_vector vector (position and clock bias)
            initial_state = np.zeros(4, dtype=np.float64)
            initial_state[:3] = a_priori_position.array
            initial_state[3] = a_priori_clock_bias

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


def main():
    # Set logging detail
    logging.basicConfig(level=logging.INFO)

    # Load data
    base = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__))))

    observation = load_cached_rinex(base + "/tests/data/GEOP057V.25o")
    navigation = load_cached_navigation_message(datetime(2025, 2, 26), "WTZR00DEU")

    # observation = load_cached_rinex(base + "/tests/data/GEOP085R.25o")
    # navigation = load_cached_navigation_message(datetime(2025, 3, 26), "WTZR00DEU")

    # Select constellations
    observation = select_constellations(observation, galileo=True, gps=False)
    navigation = select_constellations(navigation, galileo=True, gps=False)

    # Compute positions
    position_results = solve_position_solution(observation, navigation)

    # Extract results
    computed_positions = [pos for _, pos, _ in position_results]

    # Get true position from observation data
    true_position = ECEFPosition.from_array(observation.position)

    # Calculate statistics
    mean_position = ECEFPosition.from_positions_list_mean(computed_positions)
    mean_distance = mean_position.distance_to(true_position)
    horizontal_dist, altitude_diff = mean_position.horizontal_and_altitude_distance_to(true_position)

    # Convert to LLA for visualization
    mean_position_lla = mean_position.to_lla()
    true_position_lla = true_position.to_lla()
    computed_positions_lla = [p.to_lla() for p in computed_positions]

    # Print results
    print(f"Mean distance: {mean_distance:.3f} meters")
    print(f"Horizontal distance: {horizontal_dist:.3f} meters, Altitude difference: {altitude_diff:.3f} meters")
    print(f"Computed: {mean_position_lla}; {mean_position_lla.google_maps_link()}")
    print(f"Real: {true_position_lla}; {true_position_lla.google_maps_link()}")

    # Optional: Plot results
    from subprocess import Popen
    from geonss.plotting import plot_coordinates_on_map
    logging.getLogger().setLevel(logging.INFO)
    _ = plot_coordinates_on_map(
        true_position_lla,
        mean_position_lla,
        computed_positions_lla)


if __name__ == "__main__":
    main()
