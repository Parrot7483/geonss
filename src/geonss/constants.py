# noinspection SpellCheckingInspection

import numpy as np

# WGS-84 Product of the gravitational constant (G) and the Earth's mass
GM = np.float64(3.986004418e14)  # m^3/s^2

# WGS-84 value for the Earth's rotation rate
OMEGA_E = np.float64(7.2921151467e-5)  # rad/s

# WGS-84 Earth semi-major axis
EARTH_SEMI_MAJOR_AXIS = np.float64(6378137.0)  # m

# WGS-84 Earth semi-minor axis
EARTH_SEMI_MINOR_AXIS = np.float64(6356752.314245)  # m

# WGS-84 Earth flattening
EARTH_FLATTENING = np.float64(1 / 298.257223563)
EARTH_ECCENTRICITY_SQUARED = 2 * EARTH_FLATTENING - EARTH_FLATTENING ** 2  # eccentricity squared

# WGS-84 Speed of Light
SPEED_OF_LIGHT = np.float64(299792458.0)  # m/s

# Relativity Constant
REL_CONST = np.float64(-4.442807633e-10)  # sec/sqrt(m)
