import xarray as xr

from geonss.constellation import select_constellations
from geonss.coordinates import ecef_distance, lla_distance, ecef_to_lla
from geonss.parsing import load_cached
from geonss.position import spp
from tests.util import path_test_file

def test_sentinel():
    # Load data
    navigation = load_cached(path_test_file("BRDC00IGS_R_20240010000_01D_MN.rnx"))
    observation = load_cached(path_test_file("s6an0010.24o"), meas=["C1C", "S1C", "C5Q", "S5Q", "C2W", "S2W", "C2L", "S2L"])
    sat_pos_true = load_cached(path_test_file("S6ATUM_22951.sp3"))

    # Select Data
    observation_sel = select_constellations(observation, galileo=True)

    sat_pos_true, observation = xr.align(sat_pos_true.squeeze().drop_vars("sv"), observation_sel, join="inner")

    # selection = slice(600, 800)
    # observation_sel = observation_align.isel(time=selection)
    # sat_pos_true = sat_pos_true_align.isel(time=selection)

    # Calculation
    sat_pos_calc = spp(
        observation,
        navigation,
        enable_tropospheric_correction=False,
        enable_elevation_weighting=False,
        nadir_correction=-1.117,
        track_correction=-0.94,
    ).squeeze()

    # Comparison
    sat_pos_true['position'] *= 1000.0
    sat_pos_calc['position'] *= 1000.0

    sat_pos_calc_lla = ecef_to_lla(sat_pos_calc.position, coord_dim='ECEF')
    sat_pos_true_lla = ecef_to_lla(sat_pos_true.position, coord_dim='ECEF')
    lla_d = lla_distance(sat_pos_true_lla, sat_pos_calc_lla, coord_dim='ECEF')
    alt_d = sat_pos_true_lla.sel({'ECEF': 'altitude'}) - sat_pos_calc_lla.sel({'ECEF': 'altitude'})

    assert lla_d.mean().item() < 0.5 # Below 50 cm
    assert lla_d.std().item() < 0.66 # Below 66 cm
    assert alt_d.mean().item() < 0.5 # Below 50 cm
    assert alt_d.std().item() < 2.0 # Below 200 cm
