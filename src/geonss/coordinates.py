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

    def __init__(self, x: Union[float, np.floating], y: Union[float, np.floating], z: Union[float, np.floating]):
        self.x = np.float64(x)
        self.y = np.float64(y)
        self.z = np.float64(z)

    @property
    def array(self) -> np.ndarray:
        """Get position as a numpy array."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)

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

        # Initial latitude guess
        latitude = np.arctan2(self.z, p * (1 - EARTH_ECCENTRICITY_SQUARED))

        # Iterative latitude calculation
        for _ in range(5):
            n = EARTH_SEMI_MAJOR_AXIS / np.sqrt(1 - EARTH_ECCENTRICITY_SQUARED * np.sin(latitude) ** 2)
            h = p / np.cos(latitude) - n
            latitude = np.arctan2(self.z, p * (1 - EARTH_ECCENTRICITY_SQUARED * n / (n + h)))

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

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [x, y, z]."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    def distance_to(self, other: 'ECEFPosition') -> float:
        """Calculate the distance to another ECEF position in meters."""
        return float(np.sqrt((other.x - self.x) ** 2 +
                             (other.y - self.y) ** 2 +
                             (other.z - self.z) ** 2))

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

    def __repr__(self) -> str:
        return f"ECEF({float(self.x):.3f}, {float(self.y):.3f}, {float(self.z):.3f}) m"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ECEFPosition):
            return False
        return bool(np.allclose(self.array, other.array))


# TODO: to and from ISO 6709 format string (e.g., "+12.345678-098.765432+123.456") including WGS-84
class LLAPosition:
    """
    Latitude, Longitude, Altitude (LLA) position representation.
    Latitude and longitude are in degrees, altitude in meters.
    """

    def __init__(self, latitude: Union[float, np.floating], longitude: Union[float, np.floating],
                 altitude: Union[float, np.floating]):
        self.latitude = np.float64(latitude)
        self.longitude = np.float64(longitude)
        self.altitude = np.float64(altitude)

    @property
    def array(self) -> np.ndarray:
        """Get position as a numpy array."""
        return np.array([self.latitude, self.longitude, self.altitude], dtype=np.float64)

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

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [latitude, longitude, altitude]."""
        return np.array([self.latitude, self.longitude, self.altitude], dtype=np.float64)

    def horizontal_and_altitude_distance_to(self, other: 'LLAPosition') -> Tuple[float, float]:
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
        altitude_diff = float(other.altitude - self.altitude)

        # Create zero-altitude positions for horizontal distance calculation
        self_flat = LLAPosition(self.latitude, self.longitude, np.float64(0))
        other_flat = LLAPosition(other.latitude, other.longitude, np.float64(0))

        # Calculate horizontal distance using ECEF conversion
        horizontal_distance = self_flat.distance_to(other_flat)

        return horizontal_distance, altitude_diff

    def distance_to(self, other: 'LLAPosition') -> float:
        """Calculate the distance to another LLA position in meters."""
        return self.to_ecef().distance_to(other.to_ecef())

    def google_maps_link(self) -> str:
        """Generate a Google Maps link for this position."""
        return f"https://www.google.com/maps?q={float(self.latitude)},{float(self.longitude)}&h={float(self.altitude)}"

    def __repr__(self) -> str:
        lat_direction = "N" if self.latitude >= 0 else "S"
        lon_direction = "E" if self.longitude >= 0 else "W"
        return f"LLA({abs(float(self.latitude)):.6f}°{lat_direction}, {abs(float(self.longitude)):.6f}°{lon_direction}, {float(self.altitude):.3f} m)"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, LLAPosition):
            return False
        return bool(np.allclose(self.array, other.array))
