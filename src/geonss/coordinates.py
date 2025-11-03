# noinspection SpellCheckingInspection
"""
GNSS Coordinate System Module

This module provides classes and utilities for handling different coordinate systems
used in GNSS applications. It includes:
- Earth-Centered Earth-Fixed (ECEF) position representation
- Latitude, Longitude, Altitude (LLA) position representation
- Conversion between coordinate systems
- Distance calculations between points
- Utilities for horizontal and vertical distance measurements

The module implements coordinate transformations using WGS-84 reference ellipsoid
parameters for accurate positioning and navigation applications.
"""

from typing import List, Any, Self

import xarray as xr
import numpy as np

from geonss.constants import EARTH_SEMI_MAJOR_AXIS, EARTH_SEMI_MINOR_AXIS, EARTH_ECCENTRICITY_SQUARED, \
                              EARTH_SEMI_MAJOR_AXIS_SQ, EARTH_SEMI_MINOR_AXIS_SQ, EARTH_MEAN_RADIUS

def lla_to_ecef(
        da: xr.DataArray,
        coord_dim: str = 'coordinate',
        keep_attrs: bool = False
) -> xr.DataArray:
    """
    Convert LLA coordinates in a xarray DataArray to ECEF coordinates.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray containing LLA coordinates with shape (..., 3)
        where the last dimension contains [latitude, longitude, altitude]
        - latitude in degrees
        - longitude in degrees
        - altitude in meters
    coord_dim : str, default 'coordinate'
        Name of the coordinate dimension (should have size 3 for lat, lon, alt)
    keep_attrs : bool, default False
        Whether to keep variable attributes from input DataArray

    Returns
    -------
    xr.DataArray
        DataArray with ECEF coordinates [x, y, z] in meters along the coordinate dimension

    Raises
    ------
    ValueError
        If coordinate dimension doesn't have size 3
    """

    # Check that coordinate dimension exists and has size 3
    if coord_dim not in da.dims:
        raise ValueError(
            f"Coordinate dimension '{coord_dim}' not found in DataArray. Available dimensions: {list(da.dims)}"
        )

    if da.sizes[coord_dim] != 3:
        raise ValueError(
            f"Coordinate dimension '{coord_dim}' must have size 3 (lat, lon, alt), got {da.sizes[coord_dim]}"
        )

    # Extract LLA coordinates
    lat_rad = np.radians(da.sel({coord_dim: 'latitude'}))
    lon_rad = np.radians(da.sel({coord_dim: 'longitude'}))
    alt = da.sel({coord_dim: 'altitude'})

    # Calculate prime vertical radius
    n = EARTH_SEMI_MAJOR_AXIS / np.sqrt(1 - EARTH_ECCENTRICITY_SQUARED * np.sin(lat_rad) ** 2)

    # Calculate ECEF coordinates
    x = (n + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (n + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (n * (1 - EARTH_ECCENTRICITY_SQUARED) + alt) * np.sin(lat_rad)

    # Stack the results back into a DataArray
    coords = np.stack([x.values, y.values, z.values], axis=-1)

    # Create the result DataArray with same dimensions as input
    result_coords = da.coords.copy()
    result_coords[coord_dim] = ['x', 'y', 'z']

    ecef_da = xr.DataArray(
        coords,
        dims=da.dims,
        coords=result_coords,
        attrs={
            'long_name': 'ECEF coordinates',
            'units': 'm',
            'coordinate_system': 'ECEF'
        }
    )

    # Set attributes
    if keep_attrs:
        ecef_da.attrs = da.attrs.copy()

    return ecef_da

def ecef_to_lla(
        da: xr.DataArray,
        coord_dim: str = 'coordinate',
        keep_attrs: bool = False
) -> xr.DataArray:
    # pylint: disable=too-many-locals
    """
    Convert ECEF coordinates in a xarray DataArray to LLA coordinates.

    This function uses a vectorized implementation of the Ferrari/Heikkinen solution
    to convert Earth-Centered, Earth-Fixed (ECEF) coordinates (X, Y, Z)
    to Geodetic coordinates (Latitude, Longitude, Altitude).

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray containing ECEF coordinates with shape (..., 3)
        where the last dimension contains [x, y, z] in meters
    coord_dim : str, default 'coordinate'
        Name of the coordinate dimension (should have size 3 for x, y, z)
    keep_attrs : bool, default False
        Whether to keep variable attributes from input DataArray

    Returns
    -------
    xr.DataArray
        DataArray with LLA coordinates [latitude, longitude, altitude] along the coordinate dimension
        - latitude in degrees
        - longitude in degrees
        - altitude in meters

    Raises
    ------
    ValueError
        If coordinate dimension doesn't have size 3
    """

    # Check that coordinate dimension exists and has size 3
    if coord_dim not in da.dims:
        raise ValueError(
            f"Coordinate dimension '{coord_dim}' not found in DataArray. Available dimensions: {list(da.dims)}"
        )

    if da.sizes[coord_dim] != 3:
        raise ValueError(
            f"Coordinate dimension '{coord_dim}' must have size 3 (x, y, z), got {da.sizes[coord_dim]}"
        )

    # Square of the first eccentricity (e^2)
    e_sq = (EARTH_SEMI_MAJOR_AXIS_SQ - EARTH_SEMI_MINOR_AXIS_SQ) / EARTH_SEMI_MAJOR_AXIS_SQ

    # Square of the second eccentricity (e'^2)
    e_prime_sq = (EARTH_SEMI_MAJOR_AXIS_SQ - EARTH_SEMI_MINOR_AXIS_SQ) / EARTH_SEMI_MINOR_AXIS_SQ

    # Extract ECEF coordinates
    x = da.sel({coord_dim: 'x'})  # x coordinate in meters
    y = da.sel({coord_dim: 'y'})  # y coordinate in meters
    z = da.sel({coord_dim: 'z'})  # z coordinate in meters

    # Vectorized Ferrari/Heikkinen Solution
    longitude_rad = np.arctan2(y, x)

    p = np.sqrt(x ** 2 + y ** 2)
    f = 54.0 * EARTH_SEMI_MINOR_AXIS_SQ * z ** 2
    g = p ** 2 + (1 - e_sq) * z ** 2 - e_sq * (EARTH_SEMI_MAJOR_AXIS_SQ - EARTH_SEMI_MINOR_AXIS_SQ)
    c = (e_sq ** 2 * f * p ** 2) / (g ** 3)
    s = np.cbrt(1 + c + np.sqrt(c ** 2 + 2 * c))
    k = s + 1 + 1 / s
    p_formula = f / (3 * k ** 2 * g ** 2)
    q = np.sqrt(1 + 2 * e_sq ** 2 * p_formula)

    r0_term1 = (-p_formula * e_sq * p) / (1 + q)
    r0_sqrt_term = (EARTH_SEMI_MAJOR_AXIS_SQ / 2) * (1 + 1 / q) - \
                   (p_formula * (1 - e_sq) * z ** 2) / (q * (1 + q)) - \
                   (p_formula * p ** 2) / 2
    r0_sqrt_term = xr.where(r0_sqrt_term < 0, 0, r0_sqrt_term)
    r0 = r0_term1 + np.sqrt(r0_sqrt_term)

    u = np.sqrt((p - e_sq * r0) ** 2 + z ** 2)
    v = np.sqrt((p - e_sq * r0) ** 2 + (1 - e_sq) * z ** 2)

    z0 = (EARTH_SEMI_MINOR_AXIS_SQ * z) / (EARTH_SEMI_MAJOR_AXIS * v)

    latitude_rad = np.arctan((z + e_prime_sq * z0) / p)
    altitude_m = u * (1 - EARTH_SEMI_MINOR_AXIS_SQ / (EARTH_SEMI_MAJOR_AXIS * v))

    # Handle edge case where p is zero (point is on the Z-axis)
    on_z_axis = p == 0
    lat_on_z_axis = np.pi / 2 * np.sign(z)
    alt_on_z_axis = np.abs(z) - EARTH_SEMI_MINOR_AXIS

    latitude_rad = xr.where(on_z_axis, lat_on_z_axis, latitude_rad)
    altitude_m = xr.where(on_z_axis, alt_on_z_axis, altitude_m)

    # Convert latitude and longitude to degrees
    latitude_deg = np.degrees(latitude_rad)
    longitude_deg = np.degrees(longitude_rad)

    # Stack the results back into a DataArray
    lla_coords = np.stack([latitude_deg.values, longitude_deg.values, altitude_m.values], axis=-1)

    # Create the result DataArray with same dimensions as input
    result_coords = da.coords.copy()
    result_coords[coord_dim] = ['latitude', 'longitude', 'altitude']

    lla_da = xr.DataArray(
        lla_coords,
        dims=da.dims,
        coords=result_coords,
        attrs={
            'long_name': 'LLA coordinates',
            'coordinate_system': 'LLA',
            'coordinate_system_description': 'Geodetic Latitude, Longitude, Altitude'
        }
    )

    # Set attributes
    if keep_attrs:
        lla_da.attrs = da.attrs.copy()

    return lla_da

def ecef_to_elevation(
        observer_ecef: np.ndarray,
        observable_ecef: xr.DataArray,
        coord_dim: str = 'coordinate',
        keep_attrs: bool = False
) -> xr.DataArray:
    """
    Convert ECEF coordinates to elevation angles from a single observer to observable(s).

    Parameters
    ----------
    observer_ecef : np.ndarray
        Observer position in ECEF coordinates as array of shape (3,)
        containing [x, y, z] in meters
    observable_ecef : xr.DataArray
        Observable position(s) in ECEF coordinates with shape (..., 3)
        where the coordinate dimension contains [x, y, z] in meters
    coord_dim : str, default 'coordinate'
        Name of the coordinate dimension (should have size 3 for x, y, z)
    keep_attrs : bool, default False
        Whether to keep variable attributes from input DataArray

    Returns
    -------
    xr.DataArray
        DataArray with elevation angles in radian for each observable

    Raises
    ------
    ValueError
        If coordinate dimension doesn't have size 3 or observer array is not size 3

    Notes
    -----
    Elevation angle is calculated in the local East-North-Up (ENU) frame at the
    observer location. Positive angles indicate the observable is above the horizon.
    """

    # Check observer array
    if observer_ecef.shape != (3,):
        raise ValueError(
            f"Observer array must have shape (3,) for [x, y, z], got {observer_ecef.shape}"
        )

    # Check that coordinate dimension exists and has size 3
    if coord_dim not in observable_ecef.dims:
        raise ValueError(
            f"Coordinate dimension '{coord_dim}' not found in observable_ecef. "
            f"Available dimensions: {list(observable_ecef.dims)}"
        )
    if observable_ecef.sizes[coord_dim] != 3:
        raise ValueError(
            f"Coordinate dimension '{coord_dim}' in observable_ecef must have size 3 (x, y, z), "
            f"got {observable_ecef.sizes[coord_dim]}"
        )

    # Convert observer numpy array to xarray DataArray
    observer_da = xr.DataArray(
        observer_ecef,
        dims=[coord_dim],
        coords={coord_dim: ['x', 'y', 'z']}
    )

    # Convert observer to LLA
    observer_lla = ecef_to_lla(observer_da, coord_dim=coord_dim)
    observer_lat_rad = np.radians(observer_lla.sel({coord_dim: 'latitude'}))
    observer_lon_rad = np.radians(observer_lla.sel({coord_dim: 'longitude'}))

    # Calculate rotation matrix components for ECEF to ENU transformation
    sin_lat = np.sin(observer_lat_rad)
    cos_lat = np.cos(observer_lat_rad)
    sin_lon = np.sin(observer_lon_rad)
    cos_lon = np.cos(observer_lon_rad)

    # Extract observer ECEF coordinates
    observer_x = observer_ecef[0]
    observer_y = observer_ecef[1]
    observer_z = observer_ecef[2]

    # Extract ECEF coordinates for observables
    observable_x = observable_ecef.sel({coord_dim: 'x'})
    observable_y = observable_ecef.sel({coord_dim: 'y'})
    observable_z = observable_ecef.sel({coord_dim: 'z'})

    # Calculate difference vectors (observable - observer)
    dx = observable_x - observer_x
    dy = observable_y - observer_y
    dz = observable_z - observer_z

    # Transform to ENU coordinates using rotation matrix
    # ENU rotation matrix:
    # [-sin_lon,              cos_lon,             0        ]
    # [-sin_lat*cos_lon, -sin_lat*sin_lon,  cos_lat ]
    # [ cos_lat*cos_lon,  cos_lat*sin_lon,  sin_lat ]

    east = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    # Calculate horizontal distance and elevation angle
    horizontal_distance = np.sqrt(east ** 2 + north ** 2)
    elevation = np.arctan2(up, horizontal_distance)

    # Set attributes
    elevation.attrs = {
        'long_name': 'Elevation angle',
        'units': 'degrees',
        'description': 'Elevation angle from observer to observable in local ENU frame'
    }

    if keep_attrs and hasattr(observable_ecef, 'attrs'):
        elevation.attrs.update(observable_ecef.attrs)

    return elevation

def ecef_distance(
        ecef1: xr.DataArray,
        ecef2: xr.DataArray,
        coord_dim: str = 'coordinate',
        keep_attrs: bool = False
) -> xr.DataArray:
    """
    Calculate the Euclidean distance between two ECEF positions.

    Parameters
    ----------
    ecef1 : xr.DataArray
        First ECEF position(s) with shape (..., 3)
        where the coordinate dimension contains [x, y, z] in meters
    ecef2 : xr.DataArray
        Second ECEF position(s) with shape (..., 3)
        where the coordinate dimension contains [x, y, z] in meters
    coord_dim : str, default 'coordinate'
        Name of the coordinate dimension (should have size 3 for x, y, z)
    keep_attrs : bool, default False
        Whether to keep variable attributes from input DataArray

    Returns
    -------
    xr.DataArray
        DataArray with distances in meters between corresponding positions

    Raises
    ------
    ValueError
        If coordinate dimension doesn't have size 3

    Notes
    -----
    Calculates the straight-line (Euclidean) distance in 3D space.
    This is not the same as the geodetic distance along Earth's surface.
    """

    # Check that coordinate dimension exists and has size 3
    for da, name in [(ecef1, 'ecef1'), (ecef2, 'ecef2')]:
        if coord_dim not in da.dims:
            raise ValueError(
                f"Coordinate dimension '{coord_dim}' not found in {name}. "
                f"Available dimensions: {list(da.dims)}"
            )
        if da.sizes[coord_dim] != 3:
            raise ValueError(
                f"Coordinate dimension '{coord_dim}' in {name} must have size 3 (x, y, z), "
                f"got {da.sizes[coord_dim]}"
            )

    # Calculate difference vector
    diff = ecef2 - ecef1

    # Calculate Euclidean distance using linalg.norm along coordinate dimension
    distance = xr.apply_ufunc(
        np.linalg.norm,
        diff,
        input_core_dims=[[coord_dim]],
        kwargs={'axis': -1}
    )

    # Set attributes
    distance.attrs = {
        'long_name': 'Distance',
        'units': 'm',
        'description': 'Euclidean distance between ECEF positions'
    }

    if keep_attrs and hasattr(ecef1, 'attrs'):
        distance.attrs.update(ecef1.attrs)

    return distance

def lla_distance(
        lla1: xr.DataArray,
        lla2: xr.DataArray,
        coord_dim: str = 'coordinate',
        keep_attrs: bool = False
) -> xr.DataArray:
    """
    Calculate the great circle distance between two LLA positions.

    Parameters
    ----------
    lla1 : xr.DataArray
        First LLA position(s) with shape (..., 3)
        where the coordinate dimension contains [latitude, longitude, altitude]
        - latitude in degrees
        - longitude in degrees
        - altitude in meters (not used in calculation)
    lla2 : xr.DataArray
        Second LLA position(s) with shape (..., 3)
        where the coordinate dimension contains [latitude, longitude, altitude]
        - latitude in degrees
        - longitude in degrees
        - altitude in meters (not used in calculation)
    coord_dim : str, default 'coordinate'
        Name of the coordinate dimension (should have size 3 for lat, lon, alt)
    keep_attrs : bool, default False
        Whether to keep variable attributes from input DataArray

    Returns
    -------
    xr.DataArray
        DataArray with great circle distances in meters between corresponding positions

    Raises
    ------
    ValueError
        If coordinate dimension doesn't have size 3

    Notes
    -----
    Calculates the great circle distance along Earth's surface using the Haversine formula.
    This ignores altitude differences and assumes a spherical Earth with mean radius.
    For more accurate geodetic distances, consider using a proper geodetic library.
    """

    # Check that coordinate dimension exists and has size 3
    for da, name in [(lla1, 'lla1'), (lla2, 'lla2')]:
        if coord_dim not in da.dims:
            raise ValueError(
                f"Coordinate dimension '{coord_dim}' not found in {name}. "
                f"Available dimensions: {list(da.dims)}"
            )
        if da.sizes[coord_dim] != 3:
            raise ValueError(
                f"Coordinate dimension '{coord_dim}' in {name} must have size 3 (lat, lon, alt), "
                f"got {da.sizes[coord_dim]}"
            )

    # Extract latitude and longitude and convert to radians
    lat1_rad = np.radians(lla1.sel({coord_dim: 'latitude'}))
    lon1_rad = np.radians(lla1.sel({coord_dim: 'longitude'}))

    lat2_rad = np.radians(lla2.sel({coord_dim: 'latitude'}))
    lon2_rad = np.radians(lla2.sel({coord_dim: 'longitude'}))

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Calculate distance using Earth's mean radius
    distance = EARTH_MEAN_RADIUS * c

    # Set attributes
    distance.attrs = {
        'long_name': 'Great circle distance',
        'units': 'm',
        'description': 'Great circle distance along Earth\'s surface (Haversine formula)'
    }

    distance = distance.drop_vars(coord_dim)

    if keep_attrs and hasattr(lla1, 'attrs'):
        distance.attrs.update(lla1.attrs)

    return distance

class ECEFPosition:
    """
    Earth-Centered, Earth-Fixed (ECEF) position representation.
    Coordinates are in meters.
    """

    def __init__(self,
                 x: float | np.floating = 0,
                 y: float | np.floating = 0,
                 z: float | np.floating = 0,
                 ):
        assert np.isfinite(x) and np.isfinite(y) and np.isfinite(z), "ECEF coordinates must be finite numbers"
        self.array = np.array([x, y, z], dtype=np.float64)

    @property
    def x(self) -> np.float64:
        """Get x position as a numpy float64."""
        return np.float64(self.array[0])

    @x.setter
    def x(self, value: float | np.floating) -> None:
        """Set x position."""
        self.array[0] = np.float64(value)

    @property
    def y(self) -> np.float64:
        """Get y position as a numpy float64."""
        return np.float64(self.array[1])

    @y.setter
    def y(self, value: float | np.floating) -> None:
        """Set y position."""
        self.array[1] = np.float64(value)

    @property
    def z(self) -> np.float64:
        """Get z position as a numpy float64."""
        return np.float64(self.array[2])

    @z.setter
    def z(self, value: float | np.floating) -> None:
        """Set z position."""
        self.array[2] = np.float64(value)

    @classmethod
    def from_tuple(cls, coordinates: tuple[float, float, float]) -> Self:
        """Create an ECEF position from a tuple (x, y, z)."""
        return cls(*coordinates)

    @classmethod
    def from_array(cls, array: np.ndarray) -> Self:
        """Create an ECEF position from a numpy array [x, y, z]."""
        assert array.shape == (3,), f"ECEF position array must have shape (3,), got {array.shape}"
        return cls(np.float64(array[0]), np.float64(array[1]), np.float64(array[2]))

    @classmethod
    def wrap_array(cls, array: np.ndarray) -> Self:
        """
        Create an ECEF position by directly referencing the provided numpy array.

        WARNING: Changes to the original array will affect this position object.

        Args:
            array: Numpy array with shape (3,) containing [x, y, z] coordinates

        Returns:
            ECEF position object referencing the provided array
        """
        assert array.shape == (3,), f"ECEF position array must have shape (3,), got {array.shape}"
        assert array.dtype == np.float64, f"Array must be of dtype np.float64, got {array.dtype}"
        position = cls.__new__(cls)
        position.array = array
        return position

    @classmethod
    def from_positions_list_mean(cls, positions: List[Self]) -> Self:
        """Calculate the mean position given a list of ECEFPosition objects."""
        # assert len(positions) > 0, "Cannot calculate mean of empty positions list"
        x_mean = np.mean([p.x for p in positions])
        y_mean = np.mean([p.y for p in positions])
        z_mean = np.mean([p.z for p in positions])
        return cls(x_mean, y_mean, z_mean)

    @classmethod
    def from_lla(cls, lla: 'LLAPosition') -> Self:
        """Convert LLA position to ECEF position."""
        return lla.to_ecef()

    def to_lla(self) -> 'LLAPosition':
        # pylint: disable=too-many-locals
        """
        Converts Earth-Centered, Earth-Fixed (ECEF) coordinates (X, Y, Z)
        to Geodetic coordinates (Latitude, Longitude, Altitude - LLA)
        using the Ferrari/Heikkinen solution.

        https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#Ferrari's_solution

        Args:
            x (float): ECEF X-coordinate in meters.
            y (float): ECEF Y-coordinate in meters.
            z (float): ECEF Z-coordinate in meters.

        Returns:
            tuple: A tuple containing:
                - latitude_deg (float): Latitude in degrees.
                - longitude_deg (float): Longitude in degrees.
                - altitude_m (float): Altitude in meters above the ellipsoid.
        """
        # Derived geodetic parameters
        equatorial_radius_sq = EARTH_SEMI_MAJOR_AXIS ** 2
        polar_radius_sq = EARTH_SEMI_MINOR_AXIS ** 2

        # e_sq is the square of the first eccentricity (e squared)
        first_eccentricity_sq = (equatorial_radius_sq - polar_radius_sq) / equatorial_radius_sq

        # e_prime_sq is the square of the second eccentricity (e' squared in the image)
        second_eccentricity_sq = (equatorial_radius_sq - polar_radius_sq) / polar_radius_sq

        # e_fourth is e_sq squared (e to the power of 4)
        first_eccentricity_fourth = first_eccentricity_sq ** 2

        # Calculate p = sqrt(X_squared + Y_squared)
        p_dist = np.linalg.norm([self.x, self.y])

        # Calculate longitude (lambda)
        # Longitude is calculated using atan2(Y, X)
        longitude_rad = np.arctan2(self.y, self.x)

        # Handle cases where the point is on the Z-axis (p_dist = 0)
        if p_dist == 0.0:
            # Latitude is +/- 90 degrees depending on the sign of Z
            latitude_rad = np.pi / 2.0 * np.copysign(1.0, self.z) if self.z != 0.0 else 0.0
            # Altitude is the absolute Z value minus the polar radius
            altitude_m = np.abs(self.z) - EARTH_SEMI_MINOR_AXIS
            if self.z == 0.0:  # Point is at the Earth's center
                altitude_m = -EARTH_SEMI_MINOR_AXIS  # Altitude is negative polar radius
            return LLAPosition(np.degrees(latitude_rad), np.degrees(longitude_rad), altitude_m)

        z_ecef_sq = self.z ** 2

        # Intermediate calculations based on the Ferrari/Heikkinen formulas:

        # F = 54 * b_squared * Z_squared
        f_term = 54.0 * polar_radius_sq * z_ecef_sq

        # G = p_squared + (1 - e_sq) * Z_squared - e_sq * (a_sq - b_sq)
        # Note: (a_sq - b_sq) = e_sq * a_sq
        g_term = p_dist ** 2 + (1.0 - first_eccentricity_sq) * z_ecef_sq - \
                 first_eccentricity_sq * (equatorial_radius_sq - polar_radius_sq)

        # c = (e_fourth * F * p_squared) / G_cubed
        if g_term == 0.0:
            # This indicates a singularity or a case not handled by the direct formulas.
            # Division by zero would occur.
            pass
        c_term = (first_eccentricity_fourth * f_term * p_dist ** 2) / (g_term ** 3)

        # s = cuberoot(1 + c + sqrt(c_squared + 2*c))
        # Argument for the square root: c_squared + 2*c
        s_sqrt_arg = c_term ** 2 + 2.0 * c_term
        # If s_sqrt_arg is negative, np.sqrt will result in NaN or raise an error for scalars.

        s_cbrt_arg = 1.0 + c_term + np.sqrt(s_sqrt_arg)
        s_term = np.cbrt(s_cbrt_arg)

        # k = s + 1 + 1/s
        if s_term == 0.0:
            # Division by zero would occur.
            pass
        k_term = s_term + 1.0 + 1.0 / s_term

        # P_formula = F / (3 * k_squared * G_squared) (using P_formula to avoid clash with p_dist)
        p_formula_term = f_term / (3.0 * k_term ** 2 * g_term ** 2)

        # Q = sqrt(1 + 2 * e_fourth * P_formula)
        q_sqrt_arg = 1.0 + 2.0 * first_eccentricity_fourth * p_formula_term
        q_term = np.sqrt(q_sqrt_arg)

        # Denominators for r0 calculation terms
        q_plus_1 = 1.0 + q_term
        if q_term == 0.0 or q_plus_1 == 0.0:
            # Division by zero would occur in r0 calculation.
            pass

        # r0 = [-P_formula*e_sq*p / (1+Q)] + sqrt{[a_sq/2 * (1 + 1/Q)]
        #        - [P_formula*(1-e_sq)*Z_sq / (Q*(1+Q))] - [P_formula*p_sq/2]}
        r0_term1 = (-p_formula_term * first_eccentricity_sq * p_dist) / q_plus_1

        r0_sqrt_subterm1 = (equatorial_radius_sq / 2.0) * (1.0 + 1.0 / q_term) if q_term != 0.0 else np.inf
        r0_sqrt_subterm2 = (p_formula_term * (1.0 - first_eccentricity_sq) * z_ecef_sq) / (q_term * q_plus_1) \
            if (q_term != 0.0 and q_plus_1 != 0.0) else np.inf
        r0_sqrt_subterm3 = (p_formula_term * p_dist ** 2) / 2.0

        r0_sqrt_arg = r0_sqrt_subterm1 - r0_sqrt_subterm2 - r0_sqrt_subterm3
        r0_parameter = r0_term1 + np.sqrt(r0_sqrt_arg)

        # U = sqrt((p - e_sq*r0)_squared + Z_squared)
        u_sqrt_arg_term1_sq = (p_dist - first_eccentricity_sq * r0_parameter) ** 2
        u_term = np.sqrt(u_sqrt_arg_term1_sq + z_ecef_sq)

        # V = sqrt((p - e_sq*r0)_squared + (1-e_sq)*Z_squared)
        # (p - e_sq*r0)_squared is the same as u_sqrt_arg_term1_sq
        v_term = np.sqrt(u_sqrt_arg_term1_sq + (1.0 - first_eccentricity_sq) * z_ecef_sq)

        # z0 = (b_sq * Z) / (a * V)
        if v_term == 0.0:
            # Division by zero.
            pass
        z0_parameter = (polar_radius_sq * self.z) / (EARTH_SEMI_MAJOR_AXIS * v_term) \
            if v_term != 0.0 else np.inf

        # Altitude (h) = U * (1 - b_sq / (a*V))
        altitude_m = u_term * (1.0 - (polar_radius_sq / (EARTH_SEMI_MAJOR_AXIS * v_term))) \
            if v_term != 0.0 else u_term

        # Latitude (phi) = arctan[(Z + e_prime_sq * z0) / p]
        latitude_numerator = self.z + second_eccentricity_sq * z0_parameter
        # p_dist is in the denominator, already checked for p_dist = 0 at the start.
        latitude_rad = np.arctan(latitude_numerator / p_dist)

        return LLAPosition(np.degrees(latitude_rad), np.degrees(longitude_rad), altitude_m)

    def to_tuple(self) -> tuple[np.float64, np.float64, np.float64]:
        """Convert to tuple (x, y, z)."""
        return self.x, self.y, self.z

    def distance_to(self, other: Self) -> np.float64:
        """Calculate the distance to another ECEF position in meters."""
        return np.linalg.norm(other.array - self.array)

    def horizontal_and_altitude_distance_to(self, other: Self) -> tuple[float, float]:
        """
        Calculate horizontal distance and altitude difference between this position and another.

        The positions are first converted to LLA coordinates, then the horizontal distance
        and altitude difference are calculated.

        Args:
            other: Another ECEFPosition object

        Returns:
            Tuple containing:
                - horizontal distance in meters
                - altitude difference in meters (positive if other is higher than self)
        """
        # Convert both positions to LLA
        self_lla = self.to_lla()
        other_lla = other.to_lla()

        # Use the LLA method to calculate distances
        return self_lla.horizontal_and_altitude_distance_to(other_lla)

    def elevation_angle(self, observable: Self | np.ndarray) -> np.float64 | np.ndarray:
        # pylint: disable=too-many-locals
        """
        Calculate elevation angle from this position to one or multiple observables.

        Args:
            observable: Single ECEFPosition or numpy array of shape (n, 3) with ECEF coordinates

        Returns:
            Elevation angle(s) in radians - single value or numpy array
        """
        # Convert to LLA to get geodetic coordinates
        lla = self.to_lla()
        lat_rad = np.radians(lla.latitude)
        lon_rad = np.radians(lla.longitude)

        # Create the local East-North-Up (ENU) rotation matrix
        sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
        sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)

        # Rotation matrix from ECEF to ENU
        rotation = np.array([
            [-sin_lon, cos_lon, 0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
        ])

        # Handle both single ECEFPosition and numpy array of coordinates
        if isinstance(observable, ECEFPosition):
            vector_ecef = observable.array - self.array
            vector_enu = rotation @ vector_ecef

            # Calculate horizontal distance
            horizontal_distance = np.sqrt(vector_enu[0] ** 2 + vector_enu[1] ** 2)

            # Calculate elevation angle
            elevation = np.arctan2(vector_enu[2], horizontal_distance)
        else:
            # For numpy array of shape (n, 3)
            vectors_ecef = observable - self.array

            # Apply rotation to each vector (more efficiently)
            vectors_enu = np.dot(vectors_ecef, rotation.T)  # shape (n, 3)

            # Calculate horizontal distance for each vector
            horizontal_distances = np.sqrt(vectors_enu[:, 0] ** 2 + vectors_enu[:, 1] ** 2)

            # Calculate elevation angles
            elevation = np.arctan2(vectors_enu[:, 2], horizontal_distances)

        return np.float64(elevation)

    def rotate_z(self, angle: float) -> Self:
        """
        Rotate the position around the Z-axis by the specified angle.

        This performs an in-place rotation of the position vector using a
        standard 3D rotation matrix for Z-axis rotation.

        Args:
            angle: Rotation angle in radians (positive is counterclockwise when viewed from +Z)

        Returns:
            Self reference for method chaining
        """
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        # Create rotation matrix around Z-axis
        rotation_matrix = np.array([
            [cos_angle, -sin_angle, 0],
            [sin_angle, cos_angle, 0],
            [0, 0, 1]
        ])

        # Apply rotation in-place
        self.array = rotation_matrix @ self.array
        return self

    def __copy__(self):
        """Create a shallow copy of this position."""
        return ECEFPosition.from_array(self.array.copy())

    def __eq__(self, other: Any) -> bool:
        """Check equality with another ECEFPosition using numpy's allclose for array comparison."""
        return bool(np.allclose(self.array, other.array))

    def __repr__(self) -> str:
        return f"ECEF({float(self.x):.3f}, {float(self.y):.3f}, {float(self.z):.3f}) m"


# TODO: to and from ISO 6709 format string (e.g., "+12.345678-098.765432+123.456") including WGS-84
class LLAPosition:
    """
    Latitude, Longitude, Altitude (LLA) position representation.
    Latitude and longitude are in degrees, altitude in meters.
    """

    def __init__(self,
                 latitude: float | np.floating = 0,
                 longitude: float |  np.floating = 0,
                 altitude: float | np.floating = 0
                 ):
        assert -90 <= latitude <= 90, f"Latitude must be between -90 and 90 degrees, got {latitude}"
        assert -180 <= longitude <= 180, f"Longitude must be between -180 and 180 degrees, got {longitude}"
        assert np.isfinite(altitude), f"Altitude must be a finite number, got {altitude}"
        self.array = np.array([latitude, longitude, altitude], dtype=np.float64)

    @property
    def latitude(self) -> np.float64:
        """Get latitude as a numpy float64."""
        return np.float64(self.array[0])

    @property
    def longitude(self) -> np.float64:
        """Get longitude as a numpy float64."""
        return np.float64(self.array[1])

    @property
    def altitude(self) -> np.float64:
        """Get altitude as a numpy float64."""
        return np.float64(self.array[2])

    @property
    def lat(self) -> np.float64:
        """Get latitude as a numpy float64."""
        return self.latitude

    @property
    def lon(self) -> np.float64:
        """Get longitude as a numpy float64."""
        return self.longitude

    @property
    def alt(self) -> np.float64:
        """Get altitude as a numpy float64."""
        return self.altitude

    @classmethod
    def from_tuple(cls, coordinates: (float, float, float)) -> Self:
        """Create an LLA position from a tuple (latitude, longitude, altitude)."""
        return cls(*coordinates)

    @classmethod
    def from_array(cls, array: np.ndarray) -> Self:
        """Create an LLA position from a numpy array [latitude, longitude, altitude]."""
        assert array.shape == (3,), f"LLA position array must have shape (3,), got {array.shape}"
        return cls(np.float64(array[0]), np.float64(array[1]), np.float64(array[2]))

    @classmethod
    def from_ecef(cls, ecef: 'ECEFPosition') -> Self:
        """Convert ECEF position to LLA position."""
        return ecef.to_lla()

    def to_ecef(self) -> 'ECEFPosition':
        """Convert to ECEF position."""
        lat_rad = np.radians(self.latitude)
        lon_rad = np.radians(self.longitude)

        # Calculate prime vertical radius
        n = EARTH_SEMI_MAJOR_AXIS / np.sqrt(1 - EARTH_ECCENTRICITY_SQUARED * np.sin(lat_rad) ** 2)

        # Calculate ECEF coordinates
        x = (n + self.altitude) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (n + self.altitude) * np.cos(lat_rad) * np.sin(lon_rad)
        z = (n * (1 - EARTH_ECCENTRICITY_SQUARED) + self.altitude) * np.sin(lat_rad)

        return ECEFPosition(x, y, z)

    def to_tuple(self) -> tuple[np.float64, np.float64, np.float64]:
        """Convert to tuple (latitude, longitude, altitude)."""
        return self.latitude, self.longitude, self.altitude

    def horizontal_and_altitude_distance_to(self, other: Self) -> tuple[np.float64, np.float64]:
        """
        Calculate horizontal distance and altitude difference between this position and another.

        The horizontal distance is calculated by creating zero-altitude versions of both positions
        and converting to ECEF for an accurate distance measurement. The altitude difference
        is calculated directly as the difference in altitude values.

        Args:
            other: Another LLAPosition object

        Returns:
            Tuple containing:
                - horizontal distance in meters
                - altitude difference in meters (positive if other is higher than self)
        """
        # Calculate altitude difference directly
        altitude_diff = other.altitude - self.altitude

        # Create zero-altitude positions for horizontal distance calculation
        self_flat = LLAPosition(self.latitude, self.longitude, np.float64(0))
        other_flat = LLAPosition(other.latitude, other.longitude, np.float64(0))

        # Calculate horizontal distance using ECEF conversion
        horizontal_distance = self_flat.distance_to(other_flat)

        return horizontal_distance, altitude_diff

    def distance_to(self, other: Self) -> np.float64:
        """Calculate the distance to another LLA position in meters."""
        return self.to_ecef().distance_to(other.to_ecef())

    def google_maps_link(self) -> str:
        """Generate a Google Maps link for this position."""
        return f"https://www.google.com/maps?q={float(self.latitude)},{float(self.longitude)}&h={float(self.altitude)}"

    def __copy__(self):
        """Create a shallow copy of this position."""
        return LLAPosition.from_array(self.array.copy())

    def __repr__(self) -> str:
        lat_direction = "N" if self.latitude >= 0 else "S"
        lon_direction = "E" if self.longitude >= 0 else "W"
        return (
            f"LLA({np.abs(self.latitude):.6f}°{lat_direction}, "
            f"{np.abs(self.longitude):.6f}°{lon_direction}, "
            f"{self.altitude:.3f} m)"
        )

    def __eq__(self, other: Self) -> bool:
        return bool(np.allclose(self.array, other.array))
