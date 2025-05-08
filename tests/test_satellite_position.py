from geonss.parsing import load_cached
from geonss.time import datetime_utc_to_datetime_gps
from geonss.coordinates import LLAPosition, ECEFPosition
from geonss.navigation import satellite_position_velocity_clock_correction
from tests.util import path_test_file

import numpy as np

def test_solve_satellite_position_1():
    # Load data
    navigation = load_cached(path_test_file("WTZR00DEU_R_20250570000_01D_MN.rnx"))

    time = datetime_utc_to_datetime_gps(np.datetime64('2025-02-26T21:42:00'))

    ephemeris = navigation.sel(sv='G01').dropna(dim='time', how='all').sel(time=time, method='pad')

    x, y, z, _, _, _, _, = satellite_position_velocity_clock_correction(ephemeris, time)
    computed_position = ECEFPosition(x, y, z)

    # TODO: Get correct values
    real_lla = LLAPosition(24.456, 32.392, 20196620.0)
    real_position = real_lla.to_ecef()

    distance = computed_position.distance_to(real_position)

    assert distance < 3872.0 * 3 # m/s * 3s
