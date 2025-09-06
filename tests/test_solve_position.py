from geonss.parsing import load_cached
from geonss.coordinates import ECEFPosition
from geonss.constellation import select_constellations
from geonss import spp
from tests.util import path_test_file

import numpy as np


def test_solve_position_solution_1():
    observation = load_cached(path_test_file("GEOP057V.25o"))
    navigation = load_cached(path_test_file("WTZR00DEU_R_20250570000_01D_MN.rnx"))

    # Only use a subset of the data for testing
    random_indices = np.sort(np.random.choice(len(observation['time']), size=25, replace=False))
    observation = observation.isel({'time': random_indices})

    # Select constellations
    navigation = select_constellations(navigation, galileo=True)

    # Compute positions
    result = spp(observation, navigation)

    # Extract results
    computed_positions = [ECEFPosition.from_array(pos[0]) for pos in result.position * 1000.0]

    # Get true position from observation data
    true_position = ECEFPosition.from_array(np.array(observation.position))

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

    assert mean_distance < 5.0 # meters
