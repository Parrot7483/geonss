# noinspection SpellCheckingInspection
"""
GNSS Position Determination Module

This module provides functionality for calculating precise receiver positions from GNSS observations.
It includes:
- Pseudo-range calculation from dual-frequency observations
- Satellite position computation based on ephemeris data
- Earth rotation corrections for signal travel time
- Weighted least squares solver for position estimation
- Iterative positioning algorithms

The module implements standard GNSS positioning techniques including ionosphere-free combinations,
signal weighting based on SNR, and satellite position interpolation to handle transmission time effects.
"""
from math import isnan

from geonss.constellation import *
from geonss.coordinates import *
from geonss.navigation import *
from geonss.rinexmanager.util import *

logger = logging.getLogger(__name__)


def calculate_pseudo_ranges(obs_data: xr.Dataset) -> xr.Dataset:
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
    def calculate_dual_frequency_range(
            obs_slice: xr.Dataset) -> Tuple[np.float64, np.float64]:
        # L1 and L5 carrier frequencies in MHz
        f1 = 1575.42
        f5 = 1176.45

        # Get pseudo pseudo_ranges and SNR values
        c1c = obs_slice.C1C.item()
        c5q = obs_slice.C5Q.item()
        s1c = obs_slice.S1C.item()
        s5q = obs_slice.S5Q.item()

        # Skip this satellite if any pseudo-range measurement or SNR value is
        # missing
        if isnan(c1c) or isnan(c5q) or isnan(s1c) or isnan(s5q):
            return np.float64(np.nan), np.float64(np.nan)

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

    for timestamp in obs_data.time.values:
        for satellite in obs_data.sv.values:
            data_slice = obs_data.sel(time=timestamp, sv=satellite)
            pseudo_range, weight = calculate_dual_frequency_range(data_slice)

            # TODO: Maybe also use single frequency
            if isnan(pseudo_range):
                continue

            ranges['pseudo_range'].loc[dict(
                time=timestamp, sv=satellite)] = pseudo_range
            ranges['weight'].loc[dict(time=timestamp, sv=satellite)] = weight

    return ranges


def calculate_satellite_positions(
        nav_data: xr.Dataset,
        ranges: xr.Dataset
) -> xr.Dataset:
    """
    Calculate satellite positions and clock biases for each observation.

    This function determines satellite positions at signal transmission time
    by using navigation messages and correcting for signal travel time.
    For each time and satellite, it:
    1. Estimates signal transmission time based on pseudo range
    2. Computes satellite position and clock bias at transmission time
    3. Stores results in an organized dataset

    Args:
        nav_data: Navigation messages dataset containing ephemeris data
        ranges: Dataset with pseudo ranges and time/satellite coordinates

    Returns:
        Dataset containing satellite positions (x, y, z) coordinates and
        clock bias for each time and satellite
    """
    logger.info(
        f"Starting to compute {len(ranges.sv.values)} satellite positions for {len(ranges.time.values)} time steps")

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

    result['clock_bias'] = xr.DataArray(
        dims=['time', 'sv'],
        coords={'time': ranges.time, 'sv': ranges.sv},
        attrs={'long_name': 'Clock Bias', 'units': 'second'}
    )

    # Get the last valid navigation message for the first observation time
    last_nav_data = get_last_nav_messages(nav_data, ranges.time.values[0])

    for satellite in ranges.sv.values:
        # Get ephemeris data for this satellite
        ephemeris = last_nav_data.sel(sv=satellite)

        for dt in ranges.time.values:
            # Get reception time (time when signal arrived at receiver)
            gps_week, reception_time = datetime_gps_to_week_and_seconds(dt)

            # Get pseudo-range for this time and satellite
            pseudo_range = ranges.pseudo_range.sel(
                time=dt, sv=satellite).item()

            # Calculate signal travel time
            # This is physically correct: travel_time = distance/speed
            travel_time = pseudo_range / SPEED_OF_LIGHT

            # Calculate transmission time
            # This is the time when satellite sent the signal (reception_time -
            # travel_time)
            transmission_time = reception_time - travel_time

            # Calculate satellite position and clock bias at transmission time
            position, clock_bias = satellite_position_clock_correction(
                ephemeris, transmission_time)

            # Store results in dataset
            result['x'].loc[dict(time=dt, sv=satellite)] = position.x
            result['y'].loc[dict(time=dt, sv=satellite)] = position.y
            result['z'].loc[dict(time=dt, sv=satellite)] = position.z
            result['clock_bias'].loc[dict(time=dt, sv=satellite)] = clock_bias

        logger.info(f"Finished computing positions for satellite {satellite}")

    return result


def apply_earth_rotation_correction(
        sat_coords: np.ndarray, travel_time: np.float64) -> np.ndarray:
    """
    Apply Earth rotation correction to satellite coordinates.

    Args:
        sat_coords: Satellite ECEF coordinates [x, y, z]
        travel_time: Signal travel time in seconds

    Returns:
        Corrected satellite coordinates
    """
    theta = np.float64(OMEGA_E * travel_time)

    # Create rotation matrix
    rot_matrix = np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ], dtype=np.float64)

    return rot_matrix @ sat_coords


def build_geometry_and_residuals(
        ranges: xr.Dataset,
        sat_pos: xr.Dataset,
        receiver_pos: np.ndarray,
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
        weight = ranges.weight.sel(sv=satellite).item()

        if isnan(satellite_range):
            continue

        # Get satellite position
        sat_coords = np.array([
            sat_pos.sel(sv=satellite)['x'].item(),
            sat_pos.sel(sv=satellite)['y'].item(),
            sat_pos.sel(sv=satellite)['z'].item()
        ], dtype=np.float64)

        sat_clock_bias = np.float64(
            sat_pos.sel(
                sv=satellite)['clock_bias'].item() *
            SPEED_OF_LIGHT)

        # Calculate signal travel time and apply Earth rotation correction
        tau = np.float64(
            (satellite_range -
             clock_bias +
             sat_clock_bias) /
            SPEED_OF_LIGHT)
        corrected_sat_coords = apply_earth_rotation_correction(sat_coords, tau)

        # Calculate geometric range and line-of-sight vector
        predicted_range = np.linalg.norm(corrected_sat_coords - receiver_pos)
        line_of_sight = (corrected_sat_coords - receiver_pos) / predicted_range

        # Update geometry matrix and residuals
        residuals.append(satellite_range -
                         (predicted_range + clock_bias - sat_clock_bias))
        geometry_matrix.append(
            [-line_of_sight[0], -line_of_sight[1], -line_of_sight[2], 1])
        weights.append(weight)

    return (
        np.array(geometry_matrix, dtype=np.float64),
        np.array(residuals, dtype=np.float64),
        np.array(weights, dtype=np.float64)
    )


def solve_weighted_least_squares(
        geometry_matrix: np.ndarray,
        residuals: np.ndarray,
        weights: np.ndarray
) -> Tuple[np.ndarray, np.float64, np.float64, np.float64]:
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
    W = np.diag(weights)

    # Weighted least squares solution: (G^T W G)^(-1) G^T W r
    weighted_normal = np.linalg.inv(geometry_matrix.T @ W @ geometry_matrix)
    position_update = weighted_normal @ geometry_matrix.T @ W @ residuals

    # Separate position and clock bias updates
    dx = position_update[:3]
    db = position_update[3]

    # Calculate magnitudes for convergence check
    position_change = np.linalg.norm(dx)
    clock_change = abs(db)

    return dx, db, position_change, clock_change


def solve_position_solution(
        ranges: xr.Dataset,
        sat_pos: xr.Dataset,
        initial_pos: Optional[ECEFPosition] = None,
        initial_clock_bias: Optional[np.float64] = None
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
    if len([r for r in ranges.pseudo_range.values if not isnan(r)]) < 4:
        raise ValueError("Not enough pseudo-pseudo_ranges to compute position")

    # Initialize position using initial_pos or zeros
    if initial_pos is None:
        xu = np.zeros(3, dtype=np.float64)
    else:
        xu = initial_pos.array  # Use the array method to get numpy array

    # Initialize clock bias
    if initial_clock_bias is None:
        clock_bias = np.float64(0.0)
    else:
        clock_bias = initial_clock_bias

    # Iterative least squares solution
    for _ in range(10):
        # Build geometry matrix and calculate residuals for each satellite
        geometry_matrix, residuals, weights = build_geometry_and_residuals(
            ranges, sat_pos, xu, clock_bias)

        # Solve weighted least squares to get position and clock bias updates
        dx, db, position_change, clock_change = solve_weighted_least_squares(
            geometry_matrix, residuals, weights)

        # Check for convergence
        if clock_change < 0.00003 and position_change < 0.05:
            logger.debug(
                f"Converged after iteration with dx={
                    position_change:.6f}m, db={
                    clock_change:.6f}m")
            break

        # Update position and clock bias
        xu += dx
        clock_bias += db

    return ECEFPosition.from_array(xu), clock_bias


def solve_position_solutions(
        observation: xr.Dataset,
        navigation: xr.Dataset,
        initial_pos: Optional[ECEFPosition] = None,
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
        f"Computing receiver positions for {len(observation.time.values)} time steps")
    positions = []
    current_pos = initial_pos
    current_clock_bias = None

    for t in observation.time.values:
        try:
            # Select data for this time step
            ranges_t = ranges.sel(time=t)
            sat_pos_t = sat_pos.sel(time=t)

            position, clock_bias = solve_position_solution(
                ranges_t, sat_pos_t, current_pos, current_clock_bias)

            # Save for next iteration
            current_pos = position
            current_clock_bias = clock_bias

            positions.append((t, position, clock_bias))
            logger.debug(f"Successfully computed position for time {t}")

        except Exception as e:
            logger.warning(f"Error computing position at {t}: {e}")

    logger.info(
        f"Successfully computed {
            len(positions)} positions out of {
            len(
                observation.time.values)} time steps")

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
    navigation = load_cached_navigation_message(
        datetime(2025, 3, 26), "WTZR00DEU")

    # Select constellations
    observation = select_constellations(observation, galileo=True, gps=True)
    navigation = select_constellations(navigation, galileo=True, gps=True)

    # Compute positions
    position_results = solve_position_solutions(observation, navigation)

    # Extract results
    computed_positions = [pos for _, pos, _ in position_results]

    # for t, _, clock_bias in position_results:
    #     print(f"Time: {t}, Clock Bias: {clock_bias:.6f} meter")

    # Get true position from observation data
    true_position = ECEFPosition.from_array(observation.position)

    # Calculate statistics
    mean_position = ECEFPosition.from_positions_list_mean(computed_positions)
    mean_distance = mean_position.distance_to(true_position)
    horizontal_dist, altitude_diff = mean_position.horizontal_and_altitude_distance_to(
        true_position)

    # Convert to LLA for visualization
    mean_position_lla = mean_position.to_lla()
    true_position_lla = true_position.to_lla()
    computed_positions_lla = [p.to_lla() for p in computed_positions]

    # Print results
    print(f"Mean distance: {mean_distance:.3f} meters")
    print(
        f"Horizontal distance: {
            horizontal_dist:.3f} meters, Altitude difference: {
            altitude_diff:.3f} meters")
    print(
        f"Computed: {mean_position_lla}; {
            mean_position_lla.google_maps_link()}")
    print(f"Real: {true_position_lla}; {true_position_lla.google_maps_link()}")

    # Optional: Plot results
    from subprocess import Popen
    from geonss.plotting import plot_coordinates_on_map
    path = plot_coordinates_on_map(
        true_position_lla,
        mean_position_lla,
        computed_positions_lla)
    Popen(['xdg-open', path], start_new_session=True)


if __name__ == "__main__":
    main()
