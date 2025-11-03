import pytest
import xarray as xr
import numpy as np

from geonss.coordinates import ECEFPosition, LLAPosition, lla_to_ecef, ecef_to_lla, ecef_to_elevation, ecef_distance

# x, y, z, lat, lon, alt
TEST_POSITIONS = [
    (6378137.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # equator at prime meridian
    (4510023.92, 4510023.92, 0.0, 0.0, 45.0, 0),  # point on the equator at 45 degree
    (4167590.8574, 860036.0231, 4735797.0342, 48.2495956, 11.6600437, 529.66),  # point in Garching b. Munich
    (318977.27, 5635056.79, 2979456.01, 27.9881201, 86.7601802, 8764.80),  # point on Mount Everest
    (1542852.54, -4630972.86, -4092557.07, -40.1670188, -71.5740163, 593.00), # point in Patagonia
]

LLA_DA = xr.DataArray(
    np.array([[pos[3], pos[4], pos[5]] for pos in TEST_POSITIONS]),
    dims=['point', 'coordinate'],
    coords={
        'point': range(len(TEST_POSITIONS)),
        'coordinate': ['latitude', 'longitude', 'altitude']
    },
    attrs={
        'units': 'degrees/meters',
        'long_name': 'LLA coordinates'
    }
)

ECEF_DA = xr.DataArray(
    np.array([[pos[0], pos[1], pos[2]] for pos in TEST_POSITIONS]),
    dims=['point', 'coordinate'],
    coords={
        'point': range(len(TEST_POSITIONS)),
        'coordinate': ['x', 'y', 'z']
    },
    attrs={
        'units': 'meters',
        'long_name': 'ECEF coordinates'
    }
)

RECEIVER_SATELLITE_ELEVATION_AZIMUTH_RANGE = [
    (+1169256.86,-4363726.00,+4487419.12,+1187558.13,-4432027.27,+4558129.80,+90.000000,+90.000000,+100000.00),
    (+1169256.86,-4363726.00,+4487419.12,+5744574.41,-21439043.55,+22165088.65,+90.000000,+90.000000,+25000000.00),
    (+1169256.86,-4363726.00,+4487419.12,+1182197.81,-4412022.30,+4400816.58,-15.000000,+180.000000,+100000.00),
    (+1169256.86,-4363726.00,+4487419.12,+4404494.92,-16437798.83,-17163215.98,-15.000000,+180.000000,+25000000.00),
    (+1169256.86,-4363726.00,+4487419.12,+1189083.55,-4437720.21,+4423140.36,+5.000000,+180.000000,+100000.00),
    (+1169256.86,-4363726.00,+4487419.12,+1194256.86,-4457027.27,+4513301.02,+60.000000,+180.000000,+100000.00),
    (+1169256.86,-4363726.00,+4487419.12,+7419256.86,-27689043.55,+10957895.25,+60.000000,+180.000000,+25000000.00),
    (+1169256.86,-4363726.00,+4487419.12,+1187558.13,-4432027.27,+4558129.80,+90.000000,+180.000000,+100000.00),
    (+1169256.86,-4363726.00,+4487419.12,+5744574.41,-21439043.55,+22165088.65,+90.000000,+180.000000,+25000000.00),
    (+1169256.86,-4363726.00,+4487419.12,+1071218.87,-4371048.33,+4469117.85,-15.000000,+270.000000,+100000.00),
    (+1169256.86,-4363726.00,+4487419.12,-23340240.01,-6194308.62,-87898.43,-15.000000,+270.000000,+25000000.00),
    (+1169256.86,-4363726.00,+4487419.12,+1074626.90,-4395462.27,+4493581.96,+5.000000,+270.000000,+100000.00),
    (+1169256.86,-4363726.00,+4487419.12,+1136809.93,-4435817.59,+4548656.36,+60.000000,+270.000000,+100000.00),
    (+1169256.86,-4363726.00,+4487419.12,-6942474.74,-22386622.84,+19796730.01,+60.000000,+270.000000,+25000000.00),
    (+1169256.86,-4363726.00,+4487419.12,+1187558.13,-4432027.27,+4558129.80,+90.000000,+270.000000,+100000.00),
    (+1008631.16,+470332.44,+6269391.04,+918345.26,+428231.43,+6260675.46,-15.000000,+0.000000,+100000.00),
    (+1008631.16,+470332.44,+6269391.04,+977633.61,+455878.04,+6363360.30,+60.000000,+0.000000,+100000.00),
    (+1008631.16,+470332.44,+6269391.04,-6740756.82,-3143266.53,+29761706.56,+60.000000,+0.000000,+25000000.00),
    (+1008631.16,+470332.44,+6269391.04,+1024369.03,+477671.12,+6367871.81,+90.000000,+0.000000,+100000.00),
    (+1008631.16,+470332.44,+6269391.04,+963736.11,+555975.65,+6243902.34,-15.000000,+90.000000,+100000.00),
    (+1008631.16,+470332.44,+6269391.04,-10215131.27,+21881136.76,-102784.02,-15.000000,+90.000000,+25000000.00),
    (+1008631.16,+470332.44,+6269391.04,+967901.80,+561257.95,+6277974.20,+5.000000,+90.000000,+100000.00),
    (+1008631.16,+470332.44,+6269391.04,+1001129.64,+522003.32,+6354677.89,+60.000000,+90.000000,+100000.00),
    (+1008631.16,+470332.44,+6269391.04,-866748.40,+13388052.57,+27591104.34,+60.000000,+90.000000,+25000000.00),
    (+1008631.16,+470332.44,+6269391.04,+1024369.03,+477671.12,+6367871.81,+90.000000,+90.000000,+100000.00),
    (+1008631.16,+470332.44,+6269391.04,+4943098.55,+2305004.71,+30889584.86,+90.000000,+90.000000,+25000000.00),
    (+1008631.16,+470332.44,+6269391.04,+1090770.54,+508634.66,+6227129.21,-15.000000,+180.000000,+100000.00),
    (+1008631.16,+470332.44,+6269391.04,+21543476.28,+10045887.97,-4296065.50,-15.000000,+180.000000,+25000000.00),
    (+1008631.16,+470332.44,+6269391.04,+1098917.06,+512433.44,+6260675.46,+5.000000,+180.000000,+100000.00),
    (+1008631.16,+470332.44,+6269391.04,+23580106.47,+10995584.23,+4090497.47,+5.000000,+180.000000,+25000000.00),
    (+1008631.16,+470332.44,+6269391.04,+15572716.56,+7261676.99,+25420502.12,+60.000000,+180.000000,+25000000.00),
    (+1008631.16,+470332.44,+6269391.04,+1024369.03,+477671.12,+6367871.81,+90.000000,+180.000000,+100000.00),
    (+1008631.16,+470332.44,+6269391.04,+4943098.55,+2305004.71,+30889584.86,+90.000000,+180.000000,+25000000.00),
    (+1008631.16,+470332.44,+6269391.04,+1045379.69,+380890.43,+6243902.34,-15.000000,+270.000000,+100000.00),
    (+1008631.16,+470332.44,+6269391.04,+10195763.41,-21890168.14,-102784.02,-15.000000,+270.000000,+25000000.00),
    (+1008631.16,+470332.44,+6269391.04,+1052103.81,+380686.14,+6277974.20,+5.000000,+270.000000,+100000.00),
    (+1008631.16,+470332.44,+6269391.04,+11876794.38,-21941240.65,+8415182.32,+5.000000,+270.000000,+25000000.00),
    (+1008631.16,+470332.44,+6269391.04,+1043391.47,+431372.54,+6354677.89,+60.000000,+270.000000,+100000.00),
    (+1008631.16,+470332.44,+6269391.04,+9698708.14,-9269642.11,+27591104.34,+60.000000,+270.000000,+25000000.00),
    (+1008631.16,+470332.44,+6269391.04,+1024369.03,+477671.12,+6367871.81,+90.000000,+270.000000,+100000.00),
    (+1008631.16,+470332.44,+6269391.04,+4943098.55,+2305004.71,+30889584.86,+90.000000,+270.000000,+25000000.00),
    (-4483772.15,+2694122.11,-3638211.06,-4513088.99,+2711737.45,-3544241.79,-15.000000,+0.000000,+100000.00),
    (-4483772.15,+2694122.11,-3638211.06,-4569162.70,+2745429.93,-3646926.63,+60.000000,+0.000000,+100000.00),
    (-4483772.15,+2694122.11,-3638211.06,-4553987.18,+2736311.56,-3695568.70,+90.000000,+0.000000,+100000.00),
    (-4483772.15,+2694122.11,-3638211.06,-22037530.81,+13241484.41,-17977621.96,+90.000000,+0.000000,+25000000.00),
    (-4483772.15,+2694122.11,-3638211.06,-12377739.54,-20734736.96,+73101.58,-15.000000,+90.000000,+25000000.00),
    (-4483772.15,+2694122.11,-3638211.06,-4541199.61,+2612408.61,-3643210.10,+5.000000,+90.000000,+100000.00),
    (-4483772.15,+2694122.11,-3638211.06,-18840638.01,-17734252.71,-4887973.06,+5.000000,+90.000000,+25000000.00),
    (-4483772.15,+2694122.11,-3638211.06,-4570332.05,+2687800.88,-3687884.23,+60.000000,+90.000000,+100000.00),
    (-4483772.15,+2694122.11,-3638211.06,-26123749.02,+1113814.54,-16056505.18,+60.000000,+90.000000,+25000000.00),
    (-4483772.15,+2694122.11,-3638211.06,-4553987.18,+2736311.56,-3695568.70,+90.000000,+90.000000,+100000.00),
    (-4483772.15,+2694122.11,-3638211.06,-22037530.81,+13241484.41,-17977621.96,+90.000000,+90.000000,+25000000.00),
    (-4483772.15,+2694122.11,-3638211.06,-4418109.32,+2654667.90,-3702489.82,-15.000000,+180.000000,+100000.00),
    (-4483772.15,+2694122.11,-3638211.06,+11931934.04,-7169429.27,-19707901.30,-15.000000,+180.000000,+25000000.00),
    (-4483772.15,+2694122.11,-3638211.06,-4440913.78,+2668370.20,-3724813.60,+5.000000,+180.000000,+100000.00),
    (-4483772.15,+2694122.11,-3638211.06,+6230819.11,-3743853.83,-25288846.15,+5.000000,+180.000000,+25000000.00),
    (-4483772.15,+2694122.11,-3638211.06,-13540136.01,+8135734.51,-26295905.73,+60.000000,+180.000000,+25000000.00),
    (-4483772.15,+2694122.11,-3638211.06,-4553987.18,+2736311.56,-3695568.70,+90.000000,+180.000000,+100000.00),
    (-4483772.15,+2694122.11,-3638211.06,-4415850.30,+2765998.68,-3623365.80,-15.000000,+270.000000,+100000.00),
    (-4483772.15,+2694122.11,-3638211.06,+12496689.36,+20663264.70,+73101.58,-15.000000,+270.000000,+25000000.00),
    (-4483772.15,+2694122.11,-3638211.06,+6813271.97,+24961023.31,-4887973.06,+5.000000,+270.000000,+25000000.00),
    (-4483772.15,+2694122.11,-3638211.06,-13247797.15,+22542997.06,-16056505.18,+60.000000,+270.000000,+25000000.00),
    (-4483772.15,+2694122.11,-3638211.06,-4553987.18,+2736311.56,-3695568.70,+90.000000,+270.000000,+100000.00),
    (-4483772.15,+2694122.11,-3638211.06,-22037530.81,+13241484.41,-17977621.96,+90.000000,+270.000000,+25000000.00),
]

def test_lla_to_ecef():
    """Test the lla_to_ecef function using known test vectors with DataArray."""

    # Call the function to test
    result_ecef_da = lla_to_ecef(LLA_DA)

    # 1 cm tolerance
    tolerance = 1e-2

    np.testing.assert_allclose(
        result_ecef_da.sel(coordinate='x').values,
        ECEF_DA.sel(coordinate='x').values,
        atol=tolerance, rtol=1e-6,
        err_msg="ECEF X coordinates don't match expected values"
    )

    np.testing.assert_allclose(
        result_ecef_da.sel(coordinate='y').values,
        ECEF_DA.sel(coordinate='y').values,
        atol=tolerance, rtol=1e-6,
        err_msg="ECEF Y coordinates don't match expected values"
    )

    np.testing.assert_allclose(
        result_ecef_da.sel(coordinate='z').values,
        ECEF_DA.sel(coordinate='z').values,
        atol=tolerance, rtol=1e-6,
        err_msg="ECEF Z coordinates don't match expected values"
    )

    # Check that coordinate system attribute is set correctly
    assert result_ecef_da.attrs.get('coordinate_system') == 'ECEF'

def test_ecef_to_lla():
    """Test the lla_to_ecef function using known test vectors with DataArray."""
    # Call the function to test
    result_lla_da = ecef_to_lla(ECEF_DA)

    # 1 cm tolerance
    tolerance = 1e-2

    np.testing.assert_allclose(
        result_lla_da.sel(coordinate='latitude').values,
        LLA_DA.sel(coordinate='latitude').values,
        atol=tolerance, rtol=1e-6,
        err_msg="ECEF X coordinates don't match expected values"
    )

    np.testing.assert_allclose(
        result_lla_da.sel(coordinate='longitude').values,
        LLA_DA.sel(coordinate='longitude').values,
        atol=tolerance, rtol=1e-6,
        err_msg="ECEF Y coordinates don't match expected values"
    )

    np.testing.assert_allclose(
        result_lla_da.sel(coordinate='altitude').values,
        LLA_DA.sel(coordinate='altitude').values,
        atol=tolerance, rtol=1e-6,
        err_msg="ECEF Z coordinates don't match expected values"
    )

    # Check that coordinate system attribute is set correctly
    assert result_lla_da.attrs.get('coordinate_system') == 'LLA'

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

def test_elevation():
    """
    Test batch elevation angle calculation using predefined test vectors with xarray input.
    """
    # Group test data by receiver position
    receiver_positions = {}
    for rx, ry, rz, sx, sy, sz, el_deg, az_deg, distance in RECEIVER_SATELLITE_ELEVATION_AZIMUTH_RANGE:
        receiver_key = (rx, ry, rz)

        if receiver_key not in receiver_positions:
            receiver_positions[receiver_key] = []

        receiver_positions[receiver_key].append((sx, sy, sz, el_deg))

    # Test each receiver position with multiple satellites
    for (rx, ry, rz), satellites in receiver_positions.items():
        # Create receiver position
        receiver_ecef = ECEFPosition(rx, ry, rz)

        # Create satellite positions as ndarray
        satellite_positions = np.array([sat[:3] for sat in satellites])
        expected_elevations = np.array([sat[3] for sat in satellites])

        observables = xr.DataArray(
            satellite_positions,
            dims=['SV', 'coordinate'],
            coords={'coordinate': ['x', 'y', 'z']}
        )

        calculated_elevations = np.degrees(ecef_to_elevation(np.array([rx, ry, rz]), observables))

        # Assert all elevation angles match expected values
        np.testing.assert_allclose(
            calculated_elevations,
            expected_elevations,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Batch elevation angle calculation failed"
        )


def test_ecef_distance():
    """Test ECEF distance calculation."""

    # Test 1: Simple case - distance along x-axis
    ecef1 = xr.DataArray(
        [1000000, 0, 0],
        dims=['coordinate'],
        coords={'coordinate': ['x', 'y', 'z']}
    )
    ecef2 = xr.DataArray(
        [1001000, 0, 0],
        dims=['coordinate'],
        coords={'coordinate': ['x', 'y', 'z']}
    )

    distance = ecef_distance(ecef1, ecef2)
    assert np.allclose(distance, 1000, rtol=0, atol=1e-5)
    assert distance.attrs['units'] == 'm'

    # Test 2: 3D Pythagorean distance (3-4-5 triangle scaled by 1000)
    ecef1 = xr.DataArray(
        [0, 0, 0],
        dims=['coordinate'],
        coords={'coordinate': ['x', 'y', 'z']}
    )
    ecef2 = xr.DataArray(
        [3000, 4000, 0],
        dims=['coordinate'],
        coords={'coordinate': ['x', 'y', 'z']}
    )

    distance = ecef_distance(ecef1, ecef2)
    assert np.allclose(distance, 5000, rtol=0, atol=1e-5)

    # Test 3: Multiple points - single to multiple
    observer = xr.DataArray(
        [4000000, 3000000, 5000000],
        dims=['coordinate'],
        coords={'coordinate': ['x', 'y', 'z']}
    )

    observables = xr.DataArray(
        [[4000000, 3000000, 5001000],  # 1000m in z
         [4000000, 3001000, 5000000],  # 1000m in y
         [4001000, 3000000, 5000000]],  # 1000m in x
        dims=['satellite', 'coordinate'],
        coords={'coordinate': ['x', 'y', 'z']}
    )

    distances = ecef_distance(observer, observables)
    assert distances.shape == (3,)
    assert np.allclose(distances, [1000, 1000, 1000], rtol=0, atol=1e-5)

    # Test 4: Multiple to multiple (same dimensions)
    ecef1 = xr.DataArray(
        [[0, 0, 0],
         [1000, 0, 0]],
        dims=['point', 'coordinate'],
        coords={'coordinate': ['x', 'y', 'z']}
    )

    ecef2 = xr.DataArray(
        [[1000, 0, 0],
         [1000, 1000, 0]],
        dims=['point', 'coordinate'],
        coords={'coordinate': ['x', 'y', 'z']}
    )

    distances = ecef_distance(ecef1, ecef2)
    assert distances.shape == (2,)
    assert np.allclose(distances, [1000, 1000], rtol=0, atol=1e-5)

    # Test 5: Zero distance (same point)
    ecef1 = xr.DataArray(
        [4000000, 3000000, 5000000],
        dims=['coordinate'],
        coords={'coordinate': ['x', 'y', 'z']}
    )

    distance = ecef_distance(ecef1, ecef1)
    assert np.allclose(distance, 0, rtol=0, atol=1e-10)

    # Test 6: Keep attributes
    ecef1 = xr.DataArray(
        [1000000, 0, 0],
        dims=['coordinate'],
        coords={'coordinate': ['x', 'y', 'z']},
        attrs={'test_attr': 'test_value'}
    )
    ecef2 = xr.DataArray(
        [1001000, 0, 0],
        dims=['coordinate'],
        coords={'coordinate': ['x', 'y', 'z']}
    )

    distance = ecef_distance(ecef1, ecef2, keep_attrs=True)
    assert 'test_attr' in distance.attrs
    assert distance.attrs['test_attr'] == 'test_value'