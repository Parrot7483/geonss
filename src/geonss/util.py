"""
This module provides utility functions for handling and processing GNSS data.
"""

from pathlib import Path
import numpy as np
import xarray as xr
from geonss.coordinates import ECEFPosition


def distance(da1: xr.DataArray, da2: xr.DataArray):
    """
    Calculate the Euclidean distance between satellite positions in two datasets.

    Args:
        da1 (xr.DataArray): First dataset with 'ECEF' coordinate
        da2 (xr.DataArray) Second dataset with 'ECEF' coordinate

    Returns:
        xr.DataArray: Euclidean distance between positions with dimensions (time, sv)
    """
    # Calculate the position differences along ECEF dimension
    position_diff = da1 - da2

    # Use np.linalg.norm to calculate the Euclidean distance along the ECEF dimension
    return xr.apply_ufunc(
        lambda x: np.linalg.norm(x, axis=-1),
        position_diff,
        input_core_dims=[['ECEF']],
        vectorize=True
    )


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


def print_distance_information(reference: ECEFPosition, positions: list[ECEFPosition]):
    """
    Calculate the distance between a reference position and a list of positions.

    Args:
        reference (ECEFPosition): The reference ECEF position.
        positions (list[ECEFPosition]): A list of ECEF positions.

    Returns:
        dict: A dictionary with the distances and their indices.
    """
    distances = [reference.distance_to(pos) for pos in positions]

    horizontal_distances, altitude_distances = zip(*[
        reference.horizontal_and_altitude_distance_to(pos) for pos in positions
    ])

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
