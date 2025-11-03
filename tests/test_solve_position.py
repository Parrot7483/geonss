from geonss.parsing import load_cached
from geonss.coordinates import ecef_distance, ecef_to_lla, lla_distance
from geonss.constellation import select_constellations
from geonss import spp
from geonss.util import lla_to_google_maps, lla_to_str, print_distance_information
from tests.util import path_test_file
import xarray as xr
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
    computed_positions = result.position * 1000.0

    # Get true position from observation data
    true_position = xr.DataArray(
        observation.position,
        dims=['ECEF'],
        coords={'ECEF': ['x', 'y', 'z']}
    )

    # Calculate statistics
    mean_position = computed_positions.mean(dim=['sv', 'time'])
    mean_distance = ecef_distance(mean_position, true_position, coord_dim='ECEF')

    # Convert to LLA for visualization
    mean_position_lla = ecef_to_lla(mean_position, coord_dim='ECEF')
    true_position_lla = ecef_to_lla(true_position, coord_dim='ECEF')

    # Calculated distances
    horizontal_dist = lla_distance(mean_position_lla, true_position_lla, coord_dim='ECEF')
    altitude_diff = mean_position_lla.loc['altitude'] - true_position_lla.loc['altitude']

    # Print results
    print("")
    print(f"Mean distance: {mean_distance:.3f} meters")
    print(f"Horizontal distance: {horizontal_dist:.3f} meters, Altitude difference: {altitude_diff:.3f} meters")
    print(f"Computed: {lla_to_str(mean_position_lla)}; {lla_to_google_maps(mean_position_lla)}")
    print(f"Real: {lla_to_str(true_position_lla)}; {lla_to_google_maps(true_position_lla)}")

    assert mean_distance < 5.0 # meters
