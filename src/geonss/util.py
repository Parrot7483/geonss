import numpy as np
import xarray as xr


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
    vars_to_drop = [name for name, da in ds.data_vars.items() if da.isnull().all()]
    if vars_to_drop:
        return ds.drop_vars(vars_to_drop)
    else:
        return ds
