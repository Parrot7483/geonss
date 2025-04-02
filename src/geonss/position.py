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

from geonss.constellation import *
from geonss.coordinates import *
from geonss.navigation import *
from geonss.ranges import *
from geonss.rinexmanager.util import *

logger = logging.getLogger(__name__)


def build_geometry_and_residuals(
        ranges: xr.Dataset,
        sat_pos: xr.Dataset,
        receiver_pos: ECEFPosition,
        clock_bias: np.float64
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build geometry matrix, residuals vector and weights for positioning.

    Args:
        ranges: Dataset with pseudo ranges and weights
        sat_pos: Dataset with satellite positions and clock biases
        receiver_pos: Current receiver position estimate as numpy array
        clock_bias: Current receiver clock bias estimate in meters

    Returns:
        Tuple containing:
            - geometry matrix
            - residuals vector
            - weights vector
    """
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

        sat_clock_bias = np.float64(
            sat_pos.sel(
                sv=satellite)['clock_bias'].item() *
            SPEED_OF_LIGHT)

        # Calculate signal travel time and apply Earth rotation correction
        tau = np.float64((satellite_range - clock_bias + sat_clock_bias) / SPEED_OF_LIGHT)
        theta = np.float64(- OMEGA_E * tau)
        sat_coords.rotate_z(theta)

        # Calculate elevation angle
        elevation_angle = receiver_pos.calculate_elevation_angle(sat_coords)

        # Apply tropospheric correction
        corrected_range = apply_tropospheric_correction(satellite_range, elevation_angle)
        # corrected_range = satellite_range

        # Calculate geometric range and line-of-sight vector
        predicted_range = np.linalg.norm(sat_coords.array - receiver_pos.array)
        line_of_sight = (sat_coords.array - receiver_pos.array) / predicted_range

        # Calculate elevation-based weight (sin²(elevation))
        elevation_weight = np.square(np.sin(elevation_angle))

        # Combine SNR and elevation weights
        # TODO: Find a good trade-off between SNR and elevation weights
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


def solve_weighted_least_squares(
        geometry_matrix: np.ndarray,
        residuals: np.ndarray,
        weights: np.ndarray
) -> Tuple[ECEFPosition, np.float64, np.float64, np.float64]:
    """
    Solve the weighted least squares problem for position and clock bias.

    Args:
        geometry_matrix: Matrix relating position and clock bias to measurements
        residuals: Differences between measured and predicted measurements
        weights: Measurement weights

    Returns:
        Tuple containing:
            - position update vector [dx, dy, dz]
            - clock bias update (scalar)
            - magnitude of position change
            - magnitude of clock bias change
    """
    # Create weight matrix (diagonal matrix with weights)
    w = np.diag(weights)

    # Weighted least squares solution: (G^T w G)^(-1) G^T w r
    weighted_normal = np.linalg.inv(geometry_matrix.T @ w @ geometry_matrix)
    position_update = weighted_normal @ geometry_matrix.T @ w @ residuals

    # Separate position and clock bias updates
    dx = ECEFPosition.from_array(position_update[:3])
    db = np.float64(position_update[3])

    # Calculate magnitudes for convergence check
    position_change = np.linalg.norm(dx.array)
    clock_change = np.abs(db)

    return dx, db, position_change, clock_change


# TODO: implementing a phase-smoothed pseudorange algorithm
def solve_position_solution(
        ranges: xr.Dataset,
        sat_pos: xr.Dataset,
        initial_pos: ECEFPosition = ECEFPosition(),
        initial_clock_bias: np.float64 = np.float64(0.0)
) -> Tuple[ECEFPosition, np.float64]:
    """
    Solve for receiver position using iterative least squares method.

    Args:
        ranges: Dataset with pseudo ranges and weights for each satellite
        sat_pos: Dataset with satellite positions and clock bias
        initial_pos: Initial position guess as ECEFPosition
        initial_clock_bias: Initial clock bias guess in meters

    Returns:
        Tuple containing:
            - receiver position as ECEFPosition
            - receiver clock bias in meters
    """
    # Ensure we have enough satellites with valid pseudo-pseudo_ranges
    if len([r for r in ranges.pseudo_range.values if not np.isnan(r)]) < 4:
        raise ValueError("Not enough pseudo-pseudo_ranges to compute position")

    # Initialize position using initial_pos or zeros
    xu = initial_pos.copy()

    # Initialize clock bias
    clock_bias = initial_clock_bias.copy()

    # Iterative least squares solution
    for i in np.arange(10):
        # Build geometry matrix and calculate residuals for each satellite
        geometry_matrix, residuals, weights = build_geometry_and_residuals(
            ranges, sat_pos, xu, clock_bias)

        # Solve weighted least squares to get position and clock bias updates
        dx, db, position_change, clock_change = solve_weighted_least_squares(
            geometry_matrix, residuals, weights)

        # Check for convergence
        # TODO: Find good convergence criteria that are a trade-off between speed and accuracy
        if clock_change * SPEED_OF_LIGHT < 0.5 and position_change < 0.05:
            logger.debug(
                f"Converged after {i + 1} iterations with"
                f"dx={position_change:.6f}m, " 
                f"db={(clock_change * SPEED_OF_LIGHT):.6f}m"
            )
            return xu, clock_bias

        # Update position and clock bias
        xu += dx
        clock_bias += db

    logger.debug(f"Converged after 10 iterations")
    return xu, clock_bias


def solve_position_solutions(
        observation: xr.Dataset,
        navigation: xr.Dataset,
        initial_pos: ECEFPosition = ECEFPosition(),
) -> List[Tuple[np.datetime64, ECEFPosition, np.float64]]:
    """
    Compute receiver positions for each time step in the observation data.

    This function:
    1. Prepares the input data (selects common satellites)
    2. Calculates pseudo ranges and satellite positions
    3. Iteratively solves for receiver position at each time step

    Args:
        observation: Dataset containing GNSS observations
        navigation: Dataset containing navigation messages
        initial_pos: Initial position guess (optional)

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
    current_pos = initial_pos

    for t in observation.time.values:
        try:
            # Select data for this time step
            ranges_t = ranges.sel(time=t)
            sat_pos_t = sat_pos.sel(time=t)

            position, clock_bias = solve_position_solution(ranges_t, sat_pos_t, current_pos)

            # Save for next iteration
            current_pos = position
            current_clock_bias = clock_bias

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

    # observation = load_cached_rinex(base + "/tests/data/GEOP057V.25o")
    # navigation = load_cached_navigation_message(datetime(2025, 2, 26), "WTZR00DEU")

    observation = load_cached_rinex(base + "/tests/data/GEOP085R.25o")
    navigation = load_cached_navigation_message(datetime(2025, 3, 26), "WTZR00DEU")

    # Select constellations
    observation = select_constellations(observation, galileo=True, gps=False)
    navigation = select_constellations(navigation, galileo=True, gps=False)

    # Compute positions
    position_results = solve_position_solutions(observation, navigation)

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
