"""
This module provides utility functions for handling and processing GNSS data.
"""

from pathlib import Path
import numpy as np
import xarray as xr
from geonss.coordinates import ecef_distance, lla_distance, ecef_to_lla


def drop_nan_vars(ds: xr.Dataset) -> xr.Dataset:
    """
    Removes data variables from an xarray Dataset if they contain only NaN values.

    Args:
        ds: The input xarray Dataset.

    Returns:
        A new xarray Dataset with the all-NaN variables removed.
    """
    vars_to_drop = \
        [name for name, da in ds.data_vars.items() if da.isnull().all()]

    if vars_to_drop:
        return ds.drop_vars(vars_to_drop)

    return ds


def get_project_root() -> Path:
    """
    Return the project root directory.
    """
    return Path(__file__).resolve().parent


def get_project_output() -> Path:
    """
    Return the project output directory.
    """
    path = get_project_root().parent.parent / 'output'

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    return path

def lla_to_google_maps(position: xr.DataArray) -> str:
    """Generate a Google Maps link for this position."""
    lat = float(position.loc['latitude'].item())
    lon = float(position.loc['longitude'].item())
    alt = float(position.loc['altitude'].item())

    return f"https://www.google.com/maps?q={lat},{lon}&h={alt}"

def ecef_to_std(position: xr.DataArray) -> str:
    x = float(position.loc['x'].item())
    y = float(position.loc['y'].item())
    z = float(position.loc['z'].item())

    return f"ECEF({float(x):.3f}, {float(y):.3f}, {float(z):.3f}) m"

def lla_to_str(position: xr.DataArray) -> str:
    lat = float(position.loc['latitude'].item())
    lon = float(position.loc['longitude'].item())
    alt = float(position.loc['altitude'].item())

    lat_direction = "N" if lat >= 0 else "S"
    lon_direction = "E" if lon >= 0 else "W"

    return f"LLA({np.abs(lat):.6f}°{lat_direction}, {np.abs(lon):.6f}°{lon_direction}, {alt:.3f} m)"

def print_distance_information(reference: xr.DataArray, positions: xr.DataArray):
    """
    Calculate the distance between a reference position and a list of positions.

    Args:
        reference (ECEFPosition): The reference ECEF position.
        positions (list[ECEFPosition]): A list of ECEF positions.

    Returns:
        dict: A dictionary with the distances and their indices.
    """
    reference_lla = ecef_to_lla(reference, coord_dim='ECEF')
    positions_lla = ecef_to_lla(positions, coord_dim='ECEF')

    distances = ecef_distance(reference, positions, coord_dim='ECEF')

    horizontal_distances = lla_distance(reference_lla, positions_lla, coord_dim='ECEF')
    altitude_distances = reference_lla.loc['altitude'] - positions_lla.sel({'ECEF': 'altitude'})

    mean_distance = np.mean(distances)
    mean_horizontal_distance = np.mean(horizontal_distances)
    mean_altitude_distance = np.mean(altitude_distances)

    std_distance = np.std(distances)
    std_horizontal_distance = np.std(horizontal_distances)
    std_altitude_distance = np.std(altitude_distances)

    max_distance = np.max(distances)
    max_horizontal_distance = np.max(horizontal_distances)
    max_altitude_distance = np.max(altitude_distances)

    # Create one big output string
    output = f"""
    Distance Information:
    | Metric         | Distance (m)   | Horizontal (m) | Altitude (m)   |
    |----------------|----------------|----------------|----------------|
    | Mean           | {mean_distance:14.2f} | {mean_horizontal_distance:14.2f} | {mean_altitude_distance:14.2f} |
    | Std. Deviation | {std_distance:14.2f} | {std_horizontal_distance:14.2f} | {std_altitude_distance:14.2f} |
    | Max            | {max_distance:14.2f} | {max_horizontal_distance:14.2f} | {max_altitude_distance:14.2f} |
    """

    print(output)
