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

from typing import Tuple, List, Any, Union

import numpy as np

from geonss.constants import EARTH_SEMI_MAJOR_AXIS, EARTH_SEMI_MINOR_AXIS, EARTH_ECCENTRICITY_SQUARED


class ECEFPosition:
    """
    Earth-Centered, Earth-Fixed (ECEF) position representation.
    Coordinates are in meters.
    """

    def __init__(self,
                 x: Union[float, np.floating] = 0,
                 y: Union[float, np.floating] = 0,
                 z: Union[float, np.floating] = 0
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
        self.array[2] = (np.float64(value))

    @classmethod
    def from_tuple(cls, coordinates: (float, float, float)) -> 'ECEFPosition':
        """Create an ECEF position from a tuple (x, y, z)."""
        return cls(*coordinates)

    @classmethod
    def from_array(cls, array: np.ndarray) -> 'ECEFPosition':
        """Create an ECEF position from a numpy array [x, y, z]."""
        assert array.shape == (3,), f"ECEF position array must have shape (3,), got {array.shape}"
        return cls(np.float64(array[0]), np.float64(array[1]), np.float64(array[2]))

    # TODO: Maybe it is possible to make this the default behavior. Could make it faster
    @classmethod
    def wrap_array(cls, array: np.ndarray) -> 'ECEFPosition':
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
    def from_positions_list_mean(cls, positions: List['ECEFPosition']) -> 'ECEFPosition':
        """Calculate the mean position given a list of ECEFPosition objects."""
        # assert len(positions) > 0, "Cannot calculate mean of empty positions list"
        x_mean = np.mean([p.x for p in positions])
        y_mean = np.mean([p.y for p in positions])
        z_mean = np.mean([p.z for p in positions])
        return cls(x_mean, y_mean, z_mean)

    @classmethod
    def from_lla(cls, lla: 'LLAPosition') -> 'ECEFPosition':
        """Convert LLA position to ECEF position."""
        return lla.to_ecef()

    def to_lla(self) -> 'LLAPosition':
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

        # r0 = [-P_formula*e_sq*p / (1+Q)] + sqrt{[a_sq/2 * (1 + 1/Q)] - [P_formula*(1-e_sq)*Z_sq / (Q*(1+Q))] - [P_formula*p_sq/2]}
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

    def to_tuple(self) -> Tuple[np.float64, np.float64, np.float64]:
        """Convert to tuple (x, y, z)."""
        return self.x, self.y, self.z

    def distance_to(self, other: 'ECEFPosition') -> np.float64:
        """Calculate the distance to another ECEF position in meters."""
        return np.linalg.norm(other.array - self.array)

    def horizontal_and_altitude_distance_to(self, other: 'ECEFPosition') -> (float, float):
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

    def elevation_angle(self, observable: 'ECEFPosition' | np.ndarray) -> np.float64 | np.ndarray:
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
            return np.float64(elevation)
        else:
            # For numpy array of shape (n, 3)
            vectors_ecef = observable - self.array

            # Apply rotation to each vector (more efficiently)
            vectors_enu = np.dot(vectors_ecef, rotation.T)  # shape (n, 3)

            # Calculate horizontal distance for each vector
            horizontal_distances = np.sqrt(vectors_enu[:, 0] ** 2 + vectors_enu[:, 1] ** 2)

            # Calculate elevation angles
            elevations = np.arctan2(vectors_enu[:, 2], horizontal_distances)

            return elevations

    def rotate_z(self, angle: float) -> 'ECEFPosition':
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

    def copy(self) -> 'ECEFPosition':
        """Convenience method to create a shallow copy of this position."""
        return self.__copy__()

    def __copy__(self):
        """Create a shallow copy of this position."""
        return ECEFPosition.from_array(self.array.copy())

    def __eq__(self, other: Any) -> bool:
        """Check equality with another ECEFPosition using numpy's allclose for array comparison."""
        return bool(np.allclose(self.array, other.array))

    def __add__(self, other: 'ECEFPosition') -> 'ECEFPosition':
        """Add another ECEFPosition to this one and return a new ECEFPosition representing the sum"""
        return ECEFPosition.from_array(self.array + other.array)

    def __sub__(self, other: 'ECEFPosition') -> 'ECEFPosition':
        """Subtract another ECEFPosition from this one and return a new ECEFPosition representing the difference"""
        return ECEFPosition.from_array(self.array - other.array)

    def __iadd__(self, other: 'ECEFPosition') -> 'ECEFPosition':
        """Add another ECEFPosition to this one in-place and return self after addition"""
        self.array += other.array
        return self

    def __isub__(self, other: 'ECEFPosition') -> 'ECEFPosition':
        """Subtract another ECEFPosition from this one in-place and return self after subtraction"""
        self.array -= other.array
        return self

    def __repr__(self) -> str:
        return f"ECEF({float(self.x):.3f}, {float(self.y):.3f}, {float(self.z):.3f}) m"


# TODO: to and from ISO 6709 format string (e.g., "+12.345678-098.765432+123.456") including WGS-84
class LLAPosition:
    """
    Latitude, Longitude, Altitude (LLA) position representation.
    Latitude and longitude are in degrees, altitude in meters.
    """

    def __init__(self,
                 latitude: Union[float, np.floating] = 0,
                 longitude: Union[float, np.floating] = 0,
                 altitude: Union[float, np.floating] = 0
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
    def lat(self):
        """Get latitude as a numpy float64."""
        return self.latitude

    @property
    def lon(self):
        """Get longitude as a numpy float64."""
        return self.longitude

    @property
    def alt(self):
        """Get altitude as a numpy float64."""
        return self.altitude

    @classmethod
    def from_tuple(cls, coordinates: (float, float, float)) -> 'LLAPosition':
        """Create an LLA position from a tuple (latitude, longitude, altitude)."""
        return cls(*coordinates)

    @classmethod
    def from_array(cls, array: np.ndarray) -> 'LLAPosition':
        """Create an LLA position from a numpy array [latitude, longitude, altitude]."""
        assert array.shape == (3,), f"LLA position array must have shape (3,), got {array.shape}"
        return cls(np.float64(array[0]), np.float64(array[1]), np.float64(array[2]))

    @classmethod
    def from_ecef(cls, ecef: 'ECEFPosition') -> 'LLAPosition':
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

    def to_tuple(self) -> (np.float64, np.float64, np.float64):
        """Convert to tuple (latitude, longitude, altitude)."""
        return self.latitude, self.longitude, self.altitude

    def horizontal_and_altitude_distance_to(self, other: 'LLAPosition') -> (np.float64, np.float64):
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

    def distance_to(self, other: 'LLAPosition') -> np.float64:
        """Calculate the distance to another LLA position in meters."""
        return self.to_ecef().distance_to(other.to_ecef())

    def google_maps_link(self) -> str:
        """Generate a Google Maps link for this position."""
        return f"https://www.google.com/maps?q={float(self.latitude)},{float(self.longitude)}&h={float(self.altitude)}"

    def copy(self) -> 'LLAPosition':
        """Convenience method to create a shallow copy of this position."""
        return self.__copy__()

    def __copy__(self):
        """Create a shallow copy of this position."""
        return LLAPosition.from_array(self.array.copy())

    def __repr__(self) -> str:
        lat_direction = "N" if self.latitude >= 0 else "S"
        lon_direction = "E" if self.longitude >= 0 else "W"
        return f"LLA({np.abs(self.latitude):.6f}°{lat_direction}, {np.abs(self.longitude):.6f}°{lon_direction}, {self.altitude:.3f} m)"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, LLAPosition):
            return False
        return bool(np.allclose(self.array, other.array))
