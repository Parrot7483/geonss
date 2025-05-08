import logging
import os
from datetime import datetime
from datetime import timedelta

import xarray as xr
from platformdirs import user_cache_dir

import georinex as gr

logger = logging.getLogger(__name__)


def load_cached(
        rinex_path: str,
        use: set[str] | None = None,
        tlim: tuple[datetime, datetime] | None = None,
        meas: list[str] | None = None,
        verbose: bool = False
) -> xr.Dataset:
    """
    Load a rinex file and return a xarray Dataset. The processed dataset is cached.

    Parameters:
        rinex_path (str): The path to the rinex file.
        use (set[str], optional): Single character(s) for constellation type filter.
                            G=GPS, R=GLONASS, E=Galileo, S=SBAS, J=QZSS, C=BeiDou, I=IRNSS
        tlim (tuple[datetime, datetime], optional): Time range to load from the file.
                            Format: (start_time, end_time)
        meas: List[str], optional): List of measurement types to load.
        verbose (bool, optional): If True, print detailed information about the loading process.
    Returns:
        xr.Dataset: The dataset loaded from cache or created from the rinex file.
    """

    # Get the user cache directory for the application.
    cache_dir = user_cache_dir("georinex")
    os.makedirs(cache_dir, exist_ok=True)

    # Create a cache file name based on the rinex file name.
    base_name = os.path.basename(rinex_path)

    # Handle the constellation filter in the cache file name
    constellation_suffix = ""
    if use:
        constellation_list = sorted([sv.lower() for sv in use])
        # Sort the constellation characters and make them lowercase for the filename
        constellation_suffix = "." + "".join(constellation_list)

    # Handle the time limit in the cache file name
    time_suffix = ""
    if tlim:
        # For start time: just remove microseconds (truncate)
        start_time = tlim[0].replace(microsecond=0)

        # For end time: always round up to next second if there are any microseconds
        end_time = tlim[1].replace(microsecond=0)
        if tlim[1].microsecond > 0:
            end_time = end_time + timedelta(seconds=1)

        # Format as ISO string with seconds precision and use double dash (--) instead of slash
        start_str = start_time.isoformat()
        end_str = end_time.isoformat()
        time_suffix = f".{start_str}--{end_str}"

    measurment_suffix = ""
    if meas:
        # Sort the constellation characters and make them lowercase for the filename
        meas_lower = sorted([m.lower() for m in meas])

        measurment_suffix = "." + "-".join(meas_lower)

    cache_file = os.path.join(
        cache_dir,
        base_name +
        constellation_suffix +
        time_suffix +
        measurment_suffix +
        ".nc"
    )

    if os.path.isfile(cache_file):
        logger.info("Using cached netcdf from %s", cache_file)
        ds = xr.open_dataset(cache_file)
    else:
        logger.info("Processing rinex file from %s", rinex_path)
        """ 
        TODO: Because of a bug in georinex we can not pass meas to gr.load.
        Our observation files contain lines with only phase measurements.
        This causes gr.load to fail with a ValueError.
        Instead we need to filter the dataset after loading.
        """
        ds = gr.load(rinex_path, use=use, tlim=tlim, verbose=verbose)

        if meas:
            logger.info("BUG in georinex: Cannot pass meas to gr.load. Filtering after loading.")
            ds = ds[meas]
            ds = ds.dropna(dim="time", how="all")

        # TODO: This attribute is non standard ad messes with the netcdf file.
        if 'time_offset' in ds.attrs:
            del ds.attrs['time_offset']

        ds.to_netcdf(cache_file)
        logger.info("Saved netcdf to cache at %s", cache_file)

    return ds
