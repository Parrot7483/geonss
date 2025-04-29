from datetime import datetime
from typing import Optional, TextIO
import base64
import io
import os
import gzip
import logging
import shutil

import georinex as gr
import paramiko
import xarray as xr
from platformdirs import user_cache_dir


# noinspection SpellCheckingInspection
# ESA GNSS host ECDSA key for authentication
GNSS_ESA_HOST_KEY_STRING = "ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBPuHKpcr9wJQreRG7u1lur/PXGqFAzOUe5tHn4g2HhYhmsWeV4TB6lTL26NabaAHRYaFppo9cmSfM6W73YnxeCQ=" # pylint: disable=line-too-long


logger = logging.getLogger(__name__)


def download_navigation_message(
        date: datetime,
        station: str
) -> Optional[TextIO]:
    """
    Connects to the server, navigates to the directory based on the provided date,
    retrieves the Navigation message file for the specified observation station,
    decompresses the file using gzip, and returns a text stream.

    :param date: A Python datetime object representing the date.
    :param station: The observation station identifier (e.g., 'CORD00ARG').
    :return: A TextIO stream containing the decompressed file content if found, otherwise None.
    """

    # Check if the date is in the future
    if date > datetime.now():
        logger.error(
            "The provided date %s is in the future.",
            date.strftime('%Y-%m-%d'))
        return None

    # Create an SSH client
    client = paramiko.SSHClient()

    # Reject connection if host key has changed
    client.set_missing_host_key_policy(paramiko.RejectPolicy())

    # Split the key type and key data from the string
    key_type, key_data = GNSS_ESA_HOST_KEY_STRING.split(' ', 1)

    # Decode the key data from base64
    key_bytes = base64.b64decode(key_data)

    # Use the from_type_string method to load the host key
    host_key_obj = paramiko.pkey.PKey.from_type_string(key_type, key_bytes)

    # Add the host key to the in-memory key store
    client.get_host_keys().add('[gssc.esa.int]:2200', key_type, host_key_obj)

    # Connect using the anonymous username and empty password.
    # Do not use system client keys.
    client.connect(
        'gssc.esa.int',
        port=2200,
        username='anonymous',
        password='',
        allow_agent=False,
        look_for_keys=False
    )

    # Initialize the SFTP client
    sftp = client.open_sftp()

    # Calculate the nth day of the year (DDD) using the given date
    start_of_year = datetime(date.year, 1, 1)

    # Days since the start of the year (1-based)
    nth_day = (date - start_of_year).days + 1

    # Build the path to the directory for the given date:
    base_path = f'gnss/data/daily/{date.year}/{nth_day:03d}'

    # Construct the expected file name based on the station and date
    expected_filename = f"{station}_R_{date.year}{nth_day:03d}0000_01D_MN.rnx.gz"

    # Construct the full path to the file
    full_path = os.path.join(base_path, expected_filename)
    logger.info("Searching for file: %s", full_path)

    try:
        # Check if the file exists
        sftp.stat(full_path)
        logger.info("Found file: %s", full_path)

        # Download the file content to memory
        with sftp.open(full_path, 'rb') as file_handle:

            file_content = file_handle.read()

            # Decompress the gzipped file content
            with gzip.GzipFile(fileobj=io.BytesIO(file_content), mode='rb') as gz_file:
                # Read the decompressed content
                decompressed_content = gz_file.read().decode()

                # Create an io.StringIO object with the decompressed content
                string_io = io.StringIO(decompressed_content)

        return string_io

    except FileNotFoundError:
        logger.error("File not found: %s", full_path)
        return None
    except Exception as e:
        logger.error("Error accessing %s: %s", full_path, e)
        return None
    finally:
        # Close the SFTP session and SSH client
        sftp.close()
        client.close()


def load_cached_navigation_message(date: datetime, station: str) -> xr.Dataset:
    """
    Load a cached ephemeris message file, or download it if not already cached.
    The processed dataset is cached as a NetCDF file.

    Parameters:
        date (datetime): The date for the ephemeris message.
        station (str): The observation station identifier (e.g., 'CORD00ARG').

    Returns:
        xr.Dataset: The dataset loaded from cache or created from the ephemeris message file.
    """
    # Get the user cache directory for the application.
    cache_dir = user_cache_dir("georinex")
    os.makedirs(cache_dir, exist_ok=True)

    # Create a netcdf cache file name based on the station and date.
    cache_file_name = f"{station}_{date.strftime('%Y%m%d')}_navigation.nc"
    cache_file = os.path.join(cache_dir, cache_file_name)

    # Create a netcdf cache file name based on the station and date.
    cache_file_name_txt = f"{station}_{date.strftime('%Y%m%d')}_navigation.rnx"
    cache_file_txt = os.path.join(cache_dir, cache_file_name_txt)

    # Check if the cached dataset already exists.
    if os.path.isfile(cache_file):
        logger.info("Using cached netcdf from %s", cache_file)
        ds = xr.open_dataset(cache_file)
    elif os.path.isfile(cache_file_txt):
        logger.info("Using cached txt from %s", cache_file)

        # Convert the BytesIO file content into a xarray Dataset
        ds = gr.load(cache_file_txt)

        # Save the dataset to cache as NetCDF
        ds.to_netcdf(cache_file)
        logger.info("Saved netcdf to cache at %s", cache_file)
    else:
        logger.info(
            "Downloading ephemeris message for %s on %s",
            station,
            date.strftime('%Y-%m-%d'))

        # Download the ephemeris message (in memory).
        file_content = download_navigation_message(date, station)

        if file_content is not None:
            logger.info(
                "Processing downloaded ephemeris message file for %s",
                station)

            # Save the dataset to cache as original rinex file
            with open(cache_file_txt, "w") as fd:
                file_content.seek(0)
                shutil.copyfileobj(file_content, fd)
            logger.info("Saved txt to cache at %s", cache_file_txt)

            # Convert the BytesIO file content into a xarray Dataset
            file_content.seek(0)
            ds = gr.load(file_content)

            # Save the dataset to cache as NetCDF
            ds.to_netcdf(cache_file)
            logger.info("Saved netcdf to cache at %s", cache_file)
        else:
            logger.error(
                "Failed to download ephemeris message for %s on %s",
                station,
                date.strftime('%Y-%m-%d'))
            raise FileNotFoundError(
                "No file found for %s on %s",
                station,
                date.strftime('%Y-%m-%d'))

    return ds


def load_cached_rinex(rinex_path: str, use: set[str] = None) -> xr.Dataset:
    """
    Load a rinex file and return a xarray Dataset. The processed dataset is cached.

    Parameters:
        rinex_path (str): The path to the rinex file.
        use (str, optional): Single character(s) for constellation type filter.
                            G=GPS, R=GLONASS, E=Galileo, S=SBAS, J=QZSS, C=BeiDou, I=IRNSS

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
        constellation_list = sorted([ sv.lower() for sv in use])

        # Sort the constellation characters and make them lowercase for the filename
        constellation_suffix = "." + "".join(constellation_list)

    cache_file = os.path.join(cache_dir, base_name + constellation_suffix + ".nc")

    if os.path.isfile(cache_file):
        logger.info("Using cached netcdf from %s", cache_file)
        ds = xr.open_dataset(cache_file)
    else:
        logger.info("Processing rinex file from %s", rinex_path)
        # Pass the use parameter to gr.load (should be uppercase for the API)
        ds = gr.load(rinex_path, use=use)
        ds.to_netcdf(cache_file)
        logger.info("Saved netcdf to cache at %s", cache_file)

    return ds
