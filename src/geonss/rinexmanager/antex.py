import xarray as xr
import pandas as pd
import numpy as np
from midgard import parsers
from datetime import datetime
import logging


logger = logging.getLogger(__name__)

def parse_antex_to_pco_xarray(path):
    """Convert ANTEX data to an xarray Dataset containing Satellite Phase Center Offset (PCO) and
    Phase

    Args:
        path: Path to the ANTEX file

    Returns:
        xr.Dataset: Dataset containing PCO values for each satellite and frequency
    """
    # Parse the ANTEX file
    parser = parsers.parse_file(parser_name="antex", file_path=path)
    data = parser.as_dict()

    # Filter for satellite antennas only (3-char keys like 'E01')
    satellites = {key: data[key] for key in data if isinstance(key, str) and len(key) == 3}

    # Create lists to store satellite metadata and frequency-specific data
    satellite_data = []
    frequency_data = []

    # Process all satellites
    for sat, valid_periods in satellites.items():
        # Process each valid time period
        for valid_from, values in valid_periods.items():
            valid_from_ts = pd.Timestamp(valid_from)
            valid_until_ts = pd.Timestamp(values["valid_until"])

            # Get satellite metadata for this period
            cospar_id = values.get("cospar_id", "")
            sat_code = values.get("sat_code", "")
            sat_type = values.get("sat_type", "")

            # Add satellite metadata entry
            satellite_data.append({
                "sv": sat,
                "valid_from": valid_from_ts,
                "valid_until": valid_until_ts,
                "cospar_id": cospar_id,
                "sat_code": sat_code,
                "sat_type": sat_type
            })

            # Process all available frequencies for this satellite in this period
            for freq_id, freq_data in values.items():
                # Skip non-frequency entries (metadata fields)
                if not isinstance(freq_data, dict) or "neu" not in freq_data:
                    continue

                # Add frequency data point
                frequency_data.append({
                    "sv": sat,
                    "valid_from": valid_from_ts,
                    "frequency": freq_id,
                    "north": freq_data["neu"][0],
                    "east": freq_data["neu"][1],
                    "up": freq_data["neu"][2],
                })

    # Convert satellite data to DataFrame
    sat_df = pd.DataFrame(satellite_data)
    sat_df_indexed = sat_df.set_index(["sv", "valid_from"])

    # Convert frequency data to DataFrame
    freq_df = pd.DataFrame(frequency_data)
    freq_df_indexed = freq_df.set_index(["sv", "frequency", "valid_from"])

    # Combine into a single xarray Dataset
    ds = xr.merge([
        xr.Dataset.from_dataframe(dataframe=sat_df_indexed),
        xr.Dataset.from_dataframe(dataframe=freq_df_indexed)
    ])

    # Add dataset metadata
    ds.attrs.update({
        "version": parser.meta.get("version"),
        "sat_sys": parser.meta.get("sat_sys"),
        "pcv_type": parser.meta.get("pcv_type"),
        "description": "Satellite antenna phase center offset values from ANTEX file",
    })

    # Add variable metadata
    ds['north'].attrs.update({
        "long_name": "North component of phase center offset",
        "units": "m",
        "description": "Offset in North direction in satellite body-fixed reference frame"
    })

    ds['east'].attrs.update({
        "long_name": "East component of phase center offset",
        "units": "m",
        "description": "Offset in East direction in satellite body-fixed reference frame"
    })

    ds['up'].attrs.update({
        "long_name": "Up component of phase center offset",
        "units": "m",
        "description": "Offset in Up direction in satellite body-fixed reference frame"
    })

    ds['cospar_id'].attrs.update({
        "long_name": "COSPAR ID",
        "description": "Committee on Space Research satellite identifier"
    })

    ds['sat_code'].attrs.update({
        "long_name": "Satellite Code",
        "description": "Satellite specific identification code"
    })

    ds['sat_type'].attrs.update({
        "long_name": "Satellite Type",
        "description": "Type or model of the satellite"
    })

    return ds

# TODO: Generalize this function to handle multiple satellites and frequencies
def get_pco_correction(
    ds: xr.Dataset,
    satellite: str,
    frequency: str,
    time: datetime | np.datetime64,
    check_valid_until: bool = True
) -> np.ndarray | None:
    """Get phase center offset for a satellite and frequency at a given time

    Args:
        ds: xarray Dataset with PCO data
        satellite: Satellite ID (e.g., 'E01')
        frequency: Frequency name (e.g., 'E05' for E5a)
        time: Timestamp for the observation (datetime or numpy.datetime64)
        check_valid_until: If True, ensures time is before valid_until (default: True)

    Returns:
        numpy.ndarray: NEU offsets [north, east, up] in m or None if not found
    """
    # Convert time to numpy datetime64
    if isinstance(time, datetime):
        time = np.datetime64(time)
    elif not isinstance(time, np.datetime64):
        raise TypeError(f"Expected datetime.datetime or numpy.datetime64, got {type(time)}")

    # Get data for this satellite and frequency
    subset = ds.sel(sv=satellite, frequency=frequency).dropna(dim="valid_from")

    # Find entries where the time falls within the valid period
    if check_valid_until:
        valid_until = np.array([np.datetime64(ts) for ts in subset.valid_until.values])
        mask = (subset.valid_from.values <= time) & (time < valid_until)
    else:
        mask = (subset.valid_from.values <= time)

    # Apply mask to find valid entries
    valid_indices = np.where(mask)[0]

    if len(valid_indices) == 0:
        return None

    # Assert that there is only one valid correction for this time
    assert len(valid_indices) == 1, f"Found {len(valid_indices)} PCO corrections for {satellite}, {frequency} at {time}. Expected exactly one."

    # Get the matching entry (simplified as we expect only one)
    idx = valid_indices[0]

    offset = subset.isel(valid_from=idx)

    return np.array([
        offset.north.values,
        offset.east.values,
        offset.up.values
    ])


def load_cached_antex(antex_path: str) -> xr.Dataset:
    """
    Load an ANTEX file and return as xarray Dataset with PCO values.
    The processed dataset is cached as a NetCDF file.

    Args:
        antex_path (str): Path to the ANTEX file (.atx)

    Returns:
        xr.Dataset: Dataset containing satellite PCO values from the ANTEX file
    """
    from platformdirs import user_cache_dir
    from datetime import datetime
    import os.path

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