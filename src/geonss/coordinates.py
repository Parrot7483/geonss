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

from geonss.constants import *


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
        self.array = np.array([x, y, z], dtype=np.float64)

    @property
    def x(self) -> np.float64:
        """Get x position as a numpy float64."""
        return np.float64(self.array[0])

    @x.setter
    def x(self, value: Union[float, np.floating]) -> None:
        """Set x position."""
        self.array[0] = np.float64(value)

    @property
    def y(self) -> np.float64:
        """Get y position as a numpy float64."""
        return np.float64(self.array[1])

    @y.setter
    def y(self, value: Union[float, np.floating]) -> None:
        """Set y position."""
        self.array[1] = np.float64(value)

    @property
    def z(self) -> np.float64:
        """Get z position as a numpy float64."""
        return np.float64(self.array[2])

    @z.setter
    def z(self, value: Union[float, np.floating]) -> None:
        """Set z position."""
        self.array[2] = (np.float64(value))

    @classmethod
    def from_tuple(cls, coordinates: Tuple[float, float, float]) -> 'ECEFPosition':
        """Create an ECEF position from a tuple (x, y, z)."""
        return cls(*coordinates)

    @classmethod
    def from_array(cls, array: np.ndarray) -> 'ECEFPosition':
        """Create an ECEF position from a numpy array [x, y, z]."""
        return cls(np.float64(array[0]), np.float64(array[1]), np.float64(array[2]))

    @classmethod
    def from_positions_list_mean(cls, positions: List['ECEFPosition']) -> 'ECEFPosition':
        """Calculate the mean position given a list of ECEFPosition objects."""
        x_mean = np.mean([p.x for p in positions])
        y_mean = np.mean([p.y for p in positions])
        z_mean = np.mean([p.z for p in positions])
        return cls(x_mean, y_mean, z_mean)

    @classmethod
    def from_lla(cls, lla: 'LLAPosition') -> 'ECEFPosition':
        """Convert LLA position to ECEF position."""
        return lla.to_ecef()

    def to_lla(self) -> 'LLAPosition':
        """Convert to LLA (Latitude, Longitude, Altitude) position."""
        # Calculate longitude
        longitude = np.arctan2(self.y, self.x)

        # Calculate distance from Z axis
        p = np.sqrt(self.x ** 2 + self.y ** 2)

        # Handle special case when point is on or near Z-axis
        if p < 1e-10:  # Nearly zero
            latitude = np.pi / 2 if self.z > 0 else - np.pi / 2  # North or South Pole
            altitude = np.abs(self.z) - EARTH_SEMI_MAJOR_AXIS
            return LLAPosition(np.degrees(latitude), np.degrees(longitude), altitude)

        # Initial latitude guess
        latitude = np.arctan2(self.z, p * (1 - EARTH_ECCENTRICITY_SQUARED))

        # Iterative latitude calculation
        for _ in range(5):
            n = EARTH_SEMI_MAJOR_AXIS / np.sqrt(1 - EARTH_ECCENTRICITY_SQUARED * np.sin(latitude) ** 2)
            h = p / np.cos(latitude) - n

            # Add small constant to prevent division by zero
            denominator = p * (1 - EARTH_ECCENTRICITY_SQUARED * n / (n + h + 1e-10))
            latitude = np.arctan2(self.z, denominator)

        # Calculate final height
        n = EARTH_SEMI_MAJOR_AXIS / np.sqrt(1 - EARTH_ECCENTRICITY_SQUARED * np.sin(latitude) ** 2)
        altitude = p / np.cos(latitude) - n

        # Convert to degrees
        latitude_deg = np.degrees(latitude)
        longitude_deg = np.degrees(longitude)

        return LLAPosition(latitude_deg, longitude_deg, altitude)

    def to_tuple(self) -> Tuple[np.float64, np.float64, np.float64]:
        """Convert to tuple (x, y, z)."""
        return self.x, self.y, self.z

    def distance_to(self, other: 'ECEFPosition') -> np.float64:
        """Calculate the distance to another ECEF position in meters."""
        return np.linalg.norm(other.array - self.array)

    def horizontal_and_altitude_distance_to(self, other: 'ECEFPosition') -> Tuple[float, float]:
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

    # TODO: Calculate elevation angle directly in ECEF
    def calculate_elevation_angle(self, satellite: 'ECEFPosition') -> np.float64:
        """
        Calculate elevation angle from this position to a satellite.

        Args:
            satellite: Satellite position

        Returns:
            Elevation angle in radians
        """
        # Convert receiver position to LLA for reference frame
        receiver_lla = self.to_lla()

        # Calculate local tangent plane vectors
        lat_rad = np.radians(receiver_lla.latitude)
        lon_rad = np.radians(receiver_lla.longitude)

        # Up vector in ECEF
        up = np.array([
            np.cos(lat_rad) * np.cos(lon_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            np.sin(lat_rad)
        ])

        # Calculate vector from receiver to satellite
        sat_receiver_vector = satellite.array - self.array

        # Normalize vector
        los_length = np.linalg.norm(sat_receiver_vector)
        if los_length == 0:
            return np.float64(np.pi / 2)  # 90 degrees if same position

        los_unit = sat_receiver_vector / los_length

        # Calculate elevation angle (angle between los_unit and up)
        elevation_angle_rad = np.arcsin(np.dot(los_unit, up))

        return elevation_angle_rad

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

    @classmethod
    def from_tuple(cls, coordinates: Tuple[float, float, float]) -> 'LLAPosition':
        """Create an LLA position from a tuple (latitude, longitude, altitude)."""
        return cls(*coordinates)

    @classmethod
    def from_array(cls, array: np.ndarray) -> 'LLAPosition':
        """Create an LLA position from a numpy array [latitude, longitude, altitude]."""
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

    def to_tuple(self) -> Tuple[np.float64, np.float64, np.float64]:
        """Convert to tuple (latitude, longitude, altitude)."""
        return self.latitude, self.longitude, self.altitude

    def horizontal_and_altitude_distance_to(self, other: 'LLAPosition') -> Tuple[np.float64, np.float64]:
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
