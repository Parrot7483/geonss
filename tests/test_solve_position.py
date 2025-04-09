from geonss.rinexmanager.util import load_cached_rinex
from geonss.coordinates import ECEFPosition
from geonss.constellation import select_constellations
from geonss import single_point_position

import os


def test_solve_position_solution_1():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    observation = load_cached_rinex(base + "/tests/data/GEOP057V.25o")
    navigation = load_cached_rinex(base + "/tests/data/WTZR00DEU_20250226_navigation.rnx")

    # Select constellations
    observation = select_constellations(observation, galileo=True, gps=False)
    navigation = select_constellations(navigation, galileo=True, gps=False)

    # Compute positions
    position_results = single_point_position(observation, navigation)

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

    # Print results
    print("")
    print(f"Mean distance: {mean_distance:.3f} meters")
    print(f"Horizontal distance: {horizontal_dist:.3f} meters, Altitude difference: {altitude_diff:.3f} meters")
    print(f"Computed: {mean_position_lla}; {mean_position_lla.google_maps_link()}")
    print(f"Real: {true_position_lla}; {true_position_lla.google_maps_link()}")

    assert mean_distance < 10.0 # meters


def test_solve_position_solution_2():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    observation = load_cached_rinex(base + "/tests/data/GEOP085R.25o")
    navigation = load_cached_rinex(base + "/tests/data/WTZR00DEU_20250326_navigation.rnx")

    # Select constellations
    observation = select_constellations(observation, galileo=True, gps=False)
    navigation = select_constellations(navigation, galileo=True, gps=False)

    # Compute positions
    position_results = single_point_position(observation, navigation)

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

    # Print results
    print("")
    print(f"Mean distance: {mean_distance:.3f} meters")
    print(f"Horizontal distance: {horizontal_dist:.3f} meters, Altitude difference: {altitude_diff:.3f} meters")
    print(f"Computed: {mean_position_lla}; {mean_position_lla.google_maps_link()}")
    print(f"Real: {true_position_lla}; {true_position_lla.google_maps_link()}")

    assert mean_distance < 10.0 # meters


