"""
Module for parsing ANTEX files and converting them to xarray Datasets.

This module provides functions to parse ANTEX files containing satellite antenna
phase center offset (PCO) data and convert it into structured xarray Datasets.
"""
from datetime import datetime
import logging
import os

from midgard import parsers
from platformdirs import user_cache_dir
import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def parse_antex_to_pco_xarray(path: str) -> xr.Dataset:
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    """Convert ANTEX data to an xarray Dataset containing Satellite Phase Center Offset (PCO).

    The dataset structure includes NEU offsets and associated metadata indexed by
    a multi-index of (time, sv, frequency).

    Args:
        path: Path to the ANTEX file.

    Returns:
        xr.Dataset: Dataset containing PCO values.
                    Dimensions: (indexed by time, sv, frequency; NEU dimension for offset)
                    Coordinates: time, sv, frequency, NEU, valid_until, cospar_id, sat_code, sat_type
                    Data variables: offset
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"ANTEX file not found: {path}")

    # --- 1. Parse ANTEX ---
    try:
        # Use the actual midgard parser
        parser = parsers.parse_file(parser_name="antex", file_path=path)
        data = parser.as_dict()
        meta = parser.meta
    except Exception as e:
        logger.error(
            "Failed to parse ANTEX file '%s' using midgard: %s", path, e)
        raise RuntimeError(
            f"Midgard parser failed for ANTEX file: {path}") from e

    # --- 2. Flatten Data ---
    records = []
    offsets = []
    # Filter for satellite antennas only (Standard 3-char GNSS identifiers)
    satellite_keys = {key for key in data if isinstance(
        key, str) and len(key) == 3 and key[0] in 'GREJCISM'}

    for sat in satellite_keys:
        valid_periods = data.get(sat, {})  # Use .get for safety
        if not isinstance(valid_periods, dict):
            logger.warning(
                "Unexpected data structure for satellite %s in parsed data. Skipping.", sat)
            continue

        for valid_from_str, values in valid_periods.items():
            # Check if 'values' is a dictionary before proceeding
            if not isinstance(values, dict):
                logger.warning(
                    "Skipping invalid period data for %s at %s. Expected dict, got %s.",
                    sat, valid_from_str, type(values)
                )
                continue

            try:
                # Use pd.to_datetime for more robust parsing
                valid_from_ts = pd.to_datetime(valid_from_str)
                valid_until_ts = pd.to_datetime(values["valid_until"])
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(
                    "Skipping period for %s due to invalid/missing date format: %s or 'valid_until'. Error: %s",
                    sat,
                    valid_from_str,
                    e
                )
                continue

            cospar_id = values.get("cospar_id", "")
            sat_code = values.get("sat_code", "")
            sat_type = values.get("sat_type", "")

            records.append({
                "sv": sat,
                "time": valid_from_ts,
                "valid_until": valid_until_ts,
                "cospar_id": cospar_id,
                "sat_code": sat_code,
                "sat_type": sat_type
            })

            for freq_id, freq_data in values.items():
                # Skip non-frequency entries (metadata fields)
                # Check if freq_data is a dict and has the 'neu' key
                if not isinstance(freq_data, dict) or "neu" not in freq_data:
                    continue

                neu_offset = freq_data["neu"]
                # Ensure NEU data is a list/tuple/array of 3 numbers
                if not isinstance(neu_offset, (list, tuple, np.ndarray)) or len(neu_offset) != 3:
                    logger.warning(
                        "Skipping invalid NEU offset format/length for %s, %s at %s: %s",
                        sat,
                        freq_id,
                        valid_from_ts,
                        neu_offset
                    )
                    continue
                try:
                    # Convert to float, handle potential Nones or non-numeric values
                    offset_n, offset_e, offset_u = map(float, neu_offset)
                except (ValueError, TypeError) as e:
                    logger.warning(
                        "Skipping NEU offset with non-numeric values for %s, %s at %s: %s. Error: %s",
                        sat,
                        freq_id,
                        valid_from_ts,
                        neu_offset,
                        e
                    )
                    continue

                offsets.append({
                    "sv": sat,
                    "time": valid_from_ts,
                    "frequency": freq_id,
                    "offset_n": offset_n,
                    "offset_e": offset_e,
                    "offset_u": offset_u,
                })

    # --- 3. Create DataFrame ---
    # If 'records' is empty, this will create an empty DataFrame
    df = pd.DataFrame(records)
    df_offset = pd.DataFrame(offsets)

    # --- 4. Prepare for xarray ---
    # Set the multi-index directly
    try:
        df = df.set_index(["time", "sv"])
        df_offset = df_offset.set_index(["time", "sv", "frequency"])
    except KeyError as e:
        logger.error(
            "Failed to set multi-index. Missing column: %s. Columns present: %s", e, df.columns.tolist()
        )
        raise RuntimeError(
            "DataFrame construction failed, cannot set multi-index.") from e

    # --- 5. Convert to xarray Dataset ---
    try:
        ds = xr.Dataset.from_dataframe(df)
        ds_offset = xr.Dataset.from_dataframe(df_offset)
    except Exception as e:
        logger.error("Failed to convert DataFrame to xarray Dataset: %s", e)
        raise RuntimeError("xarray Dataset conversion failed.") from e

    # --- 6. Combine Offsets into a single DataArray ---
    # Stack the individual offset variables into a new 'offset' variable with a 'NEU' dimension
    try:
        ds_offset_aligned = ds_offset[['offset_n', 'offset_e', 'offset_u']].to_array(
            dim='NEU', name='offset')
        # Assign proper coordinate values to the new NEU dimension
        ds_offset_aligned = ds_offset_aligned.assign_coords(
            NEU=['n', 'e', 'u'])
    except KeyError as e:
        logger.error(
            "Failed to combine offsets. Missing variable: %s. Variables present: %s}",
            e, list(ds.data_vars)
        )
        raise RuntimeError(
            "Offset combination failed due to missing variables.") from e

    # --- 7. Final Dataset ---
    # Drop the original individual offset variables
    # ds_offset_aligned = ds_offset_aligned.drop_vars(['offset_n', 'offset_e', 'offset_u'])
    # Merge the combined offset variable back into the dataset
    ds = xr.merge([ds, ds_offset_aligned])

    # --- 8. Reorder NEU ---
    ds = ds.transpose('time', 'sv', 'frequency', 'NEU')

    # --- 8. Add Attributes ---
    ds.attrs.update({
        "version": meta.get("version"),
        "sat_sys": meta.get("sat_sys"),
        "pcv_type": meta.get("pcv_type"),
        "description": "Satellite antenna phase center offset (PCO) values from ANTEX file",
        "source_file": path
    })

    # Add variable/coordinate attributes
    ds['offset'].attrs.update({
        "long_name": "Phase center offset (NEU)",
        "units": "m",
        "description": "North, East, Up components of the PCO in the satellite body-fixed reference frame"
    })
    # Attributes for coordinates (now index levels or separate coordinates)
    ds['time'].attrs.update({"long_name": "Validity start time"})
    ds['NEU'].attrs.update(
        {"long_name": "NEU components", "description": "North, East, Up"})
    ds['sv'].attrs.update({"long_name": "Satellite vehicle identifier"})
    ds['frequency'].attrs.update({"long_name": "Frequency identifier"})
    ds['valid_until'].attrs.update({"long_name": "Validity end time"})
    ds['cospar_id'].attrs.update({"long_name": "COSPAR ID"})
    ds['sat_code'].attrs.update({"long_name": "Satellite Code"})
    ds['sat_type'].attrs.update({"long_name": "Satellite Type"})

    return ds


def load_cached_antex(antex_path: str) -> xr.Dataset:
    """
    Load an ANTEX file and return as xarray Dataset with PCO values.
    The processed dataset is cached as a NetCDF file.

    Args:
        antex_path (str): Path to the ANTEX file (.atx)

    Returns:
        xr.Dataset: Dataset containing satellite PCO values from the ANTEX file
    """

    # Get the user cache directory for the application
    cache_dir = user_cache_dir("georinex")
    os.makedirs(cache_dir, exist_ok=True)

    # Get the base name of the ANTEX file
    base_name = os.path.basename(antex_path)

    # Get the file modification date to use in the cache filename
    mod_time = os.path.getmtime(antex_path)
    mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d')

    # Create the cache file path
    cache_file = os.path.join(cache_dir, f"{base_name}.{mod_date}.nc")

    # Check if the cached dataset already exists
    if os.path.isfile(cache_file):
        logger.info("Using cached ANTEX data from %s", cache_file)
        ds = xr.open_dataset(cache_file)
    else:
        logger.info("Processing ANTEX file from %s", antex_path)

        # Parse the ANTEX file
        ds = parse_antex_to_pco_xarray(antex_path)

        # Save the dataset to cache as NetCDF
        ds.to_netcdf(cache_file)
        logger.info("Saved ANTEX data to cache at %s", cache_file)

    return ds
