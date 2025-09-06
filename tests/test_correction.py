import numpy as np
from geonss.correction import apply_phase_center_offset


def test_single_position_no_offset():
    # No offset applied
    ecef_pos = np.array([[6378137.0, 0.0, 0.0]])
    neu_off = np.array([[0.0, 0.0, 0.0]])
    expected = ecef_pos
    result = apply_phase_center_offset(ecef_pos, neu_off)
    np.testing.assert_allclose(result, expected, atol=1e-9)


def test_single_position_up_offset():
    # Offset is purely 'Up' by 1m
    ecef_pos = np.array([[6378137.0, 0.0, 0.0]])
    neu_off = np.array([[0.0, 0.0, 1.0]])  # N, E, U
    expected = np.array([[6378138.0, 0.0, 0.0]])
    result = apply_phase_center_offset(ecef_pos, neu_off)
    np.testing.assert_allclose(result, expected, atol=1e-9)


def test_single_position_east_offset():
    # Satellite on equator, prime meridian
    ecef_pos = np.array([[6378137.0, 0.0, 0.0]])
    neu_off = np.array([[0.0, 1.0, 0.0]])
    expected = np.array([[6378137.0, -1.0, 0.0]])
    result = apply_phase_center_offset(ecef_pos, neu_off)
    np.testing.assert_allclose(result, expected, atol=1e-9)


def test_single_position_north_offset():
    # Satellite on equator, prime meridian
    ecef_pos = np.array([[6378137.0, 0.0, 0.0]])
    neu_off = np.array([[1.0, 0.0, 0.0]])
    expected = np.array([[6378137.0, 0.0, -1.0]])
    result = apply_phase_center_offset(ecef_pos, neu_off)
    np.testing.assert_allclose(result, expected, atol=1e-9)


def test_multiple_positions():
    ecef_pos = np.array([
        [6378137.0, 0.0, 0.0],
        [0.0, 6378137.0, 0.0],
    ])
    neu_off = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ])
    expected = np.array([
        [6378138.0, 0.0, 0.0],
        [-1.0, 6378137.0, 0.0],
    ])
    result = apply_phase_center_offset(ecef_pos, neu_off)
    np.testing.assert_allclose(result, expected, atol=1e-9)


def test_polar_case_north_pole():
    # Satellite directly above North Pole
    ecef_pos = np.array([[0.0, 0.0, 6356752.0]])
    neu_off = np.array([[0.0, 1.0, 0.0]])
    expected = np.array([[1.0, 0.0, 6356752.0]])
    result = apply_phase_center_offset(ecef_pos, neu_off)
    np.testing.assert_allclose(result, expected, atol=1e-9)


def test_polar_case_south_pole():
    # Satellite directly above South Pole
    ecef_pos = np.array([[0.0, 0.0, -6356752.0]])
    neu_off = np.array([[0.0, 1.0, 0.0]])
    expected = np.array([[1.0, 0.0, -6356752.0]])
    result = apply_phase_center_offset(ecef_pos, neu_off)
    np.testing.assert_allclose(result, expected, atol=1e-9)

