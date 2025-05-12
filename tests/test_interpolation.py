import numpy as np
import xarray as xr
import pandas as pd

from geonss.interpolation import interpolate_orbit_positions


def test_interpolation():
    """Test basic functionality of interpolate_orbit_positions with a single satellite."""
    # Create test SP3 dataset with hourly time points
    time_points = pd.date_range(
        start='2023-01-01T00:00:00',
        end='2023-01-01T03:00:00',
        freq='1H'
    ).to_numpy()

    # Just one satellite
    satellite = ['G01']

    # Linear position data in km
    # G01 moves at 1000, 500, 250 km per hour in x, y, z
    positions = np.zeros((len(time_points), 1, 3))
    clocks = np.zeros((len(time_points), 1))

    for i, _ in enumerate(time_points):
        positions[i, 0] = [i * 1000, i * 500, i * 250]  # G01
        clocks[i, 0] = i * 0.1  # G01 clock

    # Create SP3 dataset
    sp3_ds = xr.Dataset(
        coords={
            'time': time_points,
            'sv': satellite,
            'ECEF': ['x', 'y', 'z']
        },
        data_vars={
            'position': (['time', 'sv', 'ECEF'], positions),
            'clock': (['time', 'sv'], clocks)
        }
    )

    # Create query time between SP3 time points
    query_time = pd.to_datetime(['2023-01-01T01:30:00']).to_numpy()  # 1.5 hours

    query_ds = xr.Dataset(
        coords={
            'time': query_time,
            'sv': satellite
        }
    )

    # Call interpolation function with a window size of 4
    result = interpolate_orbit_positions(sp3_ds, query_ds, 4)

    # Extract results for the query time
    pos = result.sel(time=query_time[0], sv='G01').position.values
    vel = result.sel(time=query_time[0], sv='G01').velocity.values
    clock_bias = result.sel(time=query_time[0], sv='G01').clock.values

    # Expected interpolated values (linear motion)
    expected_pos = np.array([1.5 * 1000, 1.5 * 500, 1.5 * 250]) * 1000  # convert km to m
    expected_vel = np.array([1000, 500, 250]) * 1000 / 3600  # km/h to m/s
    expected_clock = 1.5 * 0.1

    # Assert interpolated values match expected
    np.testing.assert_allclose(pos, expected_pos, rtol=1e-5)
    np.testing.assert_allclose(vel, expected_vel, rtol=1e-5)
    np.testing.assert_allclose(clock_bias, expected_clock, rtol=1e-5)
