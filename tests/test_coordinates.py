import pytest
import numpy as np

from geonss.coordinates import ECEFPosition, LLAPosition

TEST_POSITIONS = [
    (6378137.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # equator at prime meridian
    (4510023.92, 4510023.92, 0.0, 0.0, 45.0, 0),  # point on the equator at 45 degree
    (4167590.8574, 860036.0231, 4735797.0342, 48.2495956, 11.6600437, 529.66),  # point in Garching b. Munich
    (318977.27, 5635056.79, 2979456.01, 27.9881201, 86.7601802, 8764.80),  # point on Mount Everest
    (1542852.54, -4630972.86, -4092557.07, -40.1670188, -71.5740163, 593.00), # point in Patagonia
]

@pytest.mark.parametrize("x,y,z,lat,lon,alt", TEST_POSITIONS)
def test_ecef_to_lla_coordinates(x, y, z, lat, lon, alt):
    lat_calc, lon_calc, alt_calc = ECEFPosition(x, y, z).to_lla().to_tuple()

    assert np.abs(lat_calc - lat) < np.float64(1e-6)
    assert np.abs(lon_calc - lon) < np.float64(1e-6)
    assert np.abs(alt_calc - alt) < np.float64(1e-2)

@pytest.mark.parametrize("x,y,z,lat,lon,alt", TEST_POSITIONS)
def test_lla_to_ecef_coordinates(x, y, z, lat, lon, alt):
    x_calc, y_calc, z_calc = LLAPosition(lat, lon, alt).to_ecef().to_tuple()

    assert np.abs(x_calc - x) < np.float64(1e-2)
    assert np.abs(y_calc - y) < np.float64(1e-2)
    assert np.abs(z_calc - z) < np.float64(1e-2)