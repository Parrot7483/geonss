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
signal weighting based on SNR, and observable position interpolation to handle transmission time effects.
"""
import logging
import os

import numpy as np

from geonss.constellation import select_constellations
from geonss.coordinates import ECEFPosition
from geonss.parsing import load_cached, load_cached_antex
from geonss.plotting import plot_positions_in_latlon, plot_positions_in_ecef, plot_altitude_differences
from geonss.single_point_position import single_point_position

logger = logging.getLogger(__name__)


def analyze_and_print_positions(positions, true_position):
    """
    Analyze and print positioning results compared to true position.

    Args:
        positions: List of ECEF positions
        true_position: The actual ECEF position to compare against
    """
    # Get Mean position
    mean_position = ECEFPosition.from_positions_list_mean(positions)

    # Calculate statistics
    std_dev = np.std([pos.distance_to(true_position) for pos in positions])
    distance = true_position.distance_to(mean_position)
    horizontal_dist, altitude_diff = true_position.horizontal_and_altitude_distance_to(mean_position)

    # Convert to LLA for visualization
    computed_position_lla = mean_position.to_lla()
    true_position_lla = true_position.to_lla()

    # Print results
    print(f"Mean distance: {distance:.3f}m")
    print(f"Mean horizontal distance: {horizontal_dist:.3f}m, Mean altitude difference: {altitude_diff:.3f}m")
    print(f"1σ (68%): {std_dev:.3f}m, 2σ (95%): {std_dev * 2:.3f}m, 3σ (99.7%): {std_dev * 3:.3f}m")
    print(f"Mean: {computed_position_lla}; {computed_position_lla.google_maps_link()}")
    print(f"Real: {true_position_lla}; {true_position_lla.google_maps_link()}")


def main():
    # Set logging detail
    logging.basicConfig(level=logging.INFO)

    # Get path to the project root directory (3 levels up from current file)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))

    # Load data
    # observation = load_cached_rinex(os.path.join(project_root, "code/tests/data/GEOP057V.25o"))
    # navigation = load_cached_rinex(os.path.join(project_root, "code/tests/data/BRDC00IGS_R_20250570000_01D_MN.rnx.gz"))

    # observation = load_cached_rinex(os.path.join(project_root, "code/tests/data/GEOP085R.25o"))
    # navigation = load_cached_navigation_message(datetime(2025, 3, 26), "WTZR00DEU")

    observation = load_cached(os.path.join(project_root, "code/tests/data/WTZR00DEU_R_20250980000_01D_30S_MO.crx"))
    navigation = load_cached(os.path.join(project_root, "code/tests/data/BRDC00IGS_R_20250980000_01D_MN.rnx"))
    sp3 = load_cached(os.path.join(project_root, "code/tests/data/COD0OPSRAP_20250980000_01D_05M_ORB.SP3"))
    antex = load_cached_antex(os.path.join(project_root, "code/tests/data/igs20.atx"))

    navigation = select_constellations(navigation, gps=False, galileo=True)
    sp3 = select_constellations(sp3, gps=False, galileo=True)

    # Only use a subset of the data for testing
    # observation = observation.isel(time=slice(40, -40))
    observation = observation.isel(time=np.sort(np.random.choice(len(observation.time), size=50, replace=False)))
    # observation = observation.isel(time=slice(40, 140))
    # observation = observation.isel(time=slice(130, 140))

    # Compute positions
    position_results = single_point_position(observation, navigation=navigation, sp3=sp3, antex=antex)
    # position_results = single_point_position(observation, navigation=navigation)

    # Get true position from observation data
    true_position = ECEFPosition.wrap_array(observation.position)

    computed_positions = [ECEFPosition.wrap_array(pos) for pos in position_results.position.values]

    # Analyze the results
    analyze_and_print_positions(computed_positions, true_position)

    # Plot results
    logging.getLogger().setLevel(logging.INFO)

    path1 = plot_positions_in_ecef(
        true_position,
        computed_positions,
        margin=2.5,
    )
    print(f"ECEF plot: file://{path1}")

    path2 = plot_positions_in_latlon(
        true_position.to_lla(),
        [p.to_lla() for p in computed_positions],
        margin=0.00002,
    )
    print(f"LLA plot: file://{path2}")

    path3 = plot_altitude_differences(
        true_position.to_lla(),
        [p.to_lla() for p in computed_positions],
    )
    print(f"Altitude plot: file://{path3}")


if __name__ == "__main__":
    main()
