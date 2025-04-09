from geonss.time import datetime_utc_to_week_and_seconds
from geonss.position import *
from geonss.navigation import satellite_position_velocity_clock_correction, get_last_nav_messages

import numpy as np

def test_solve_satellite_position_1():
    # Load data
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    navigation = load_cached_rinex(base + "/tests/data/WTZR00DEU_20250226_navigation.rnx")

    navigation = select_satellites(navigation, ['G01'])

    time = np.datetime64('2025-02-26T21:42:00')

    ephemeris = get_last_nav_messages(navigation, time).sel(sv='G01')
    week, seconds = datetime_utc_to_week_and_seconds(time)

    x, y, z, _, _, _, _, = satellite_position_velocity_clock_correction(ephemeris, seconds)
    computed_position = ECEFPosition(x, y, z)

    # TODO: Get correct values
    real_lla = LLAPosition(24.456, 32.392, 20196620.0)
    real_position = real_lla.to_ecef()

    distance = computed_position.distance_to(real_position)

    # print()
    # print('G01', time, week, seconds)
    # print(f"Computed: {computed_position}")
    # print(f"Expected: {real_position}")
    # print(f"Distance: {distance}")

    assert distance < 3872.0 * 3 # m/s * 3s

# def test_solve_satellite_position_2():
#     base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#
#     navigation = load_cached_rinex(base + "/tests/data/WTZR00DEU_20250226_navigation.rnx")
#
#     navigation = select_satellites(navigation, ['E10'])
#
#     time = np.datetime64('2025-02-26T21:42:00')
#     ephemeris = get_last_nav_messages(navigation, time).sel(sv='E10')
#     week, seconds = utc_to_gps_time(time)
#
#     x, y, z, _, _, _, _ = satellite_position_clock_correction(ephemeris, seconds)
#     computed_position = ECEFPosition(x, y, z)
#
#     # TODO: Get correct values
#     real_lla = LLAPosition(32.7, 28.0, 23234000.0)
#     real_position = real_lla.to_ecef()
#
#     distance = computed_position.distance_to(real_position)
#
#     print()
#     print('E10', time, week, seconds)
#     print(f"Computed: {computed_position}")
#     print(f"Expected: {real_position}")
#     print(f"Distance: {distance}")
#
#     # assert distance < 3872.0 # m/s * 1s
#     assert True