"""
Module for writing data to SP3 files.
"""
import sys

import xarray as xr
import pandas as pd


def write_sp3_from_xarray(dataset: xr.Dataset, output_filepath: str, satellite_id: str = "UNK"):
    # pylint: disable=too-many-locals
    """
    Writes an xarray Dataset to an SP3 format text file compatible with georinex parser.
    """

    # Ensure satellite_id is exactly 3 characters (required by SP3 format)
    if len(satellite_id) > 3:
        satellite_id = satellite_id[:3]
    elif len(satellite_id) < 3:
        satellite_id = satellite_id.ljust(3)

    # Determine if we're writing to a file or stdout
    is_stdout = output_filepath is None or output_filepath in ["-", ""]

    # Open file if needed, otherwise use stdout
    with (sys.stdout if is_stdout else open(output_filepath, 'w', encoding="utf-8")) as f:
        # Get start time and number of epochs
        start_time = pd.to_datetime(dataset['time'].values[0])
        num_epochs = len(dataset['time'])

        # Write version line with proper time format
        microsec = start_time.microsecond / 1e6
        f.write(f"#cP{start_time.year:4d} {start_time.month:2d} {start_time.day:2d} "
                f"{start_time.hour:2d} {start_time.minute:2d} {start_time.second + microsec:11.8f}  "
                f"{num_epochs:5d} ITRF SP3 GEONSS     GPS 0  0 0\n")

        # Empty line that georinex expects to skip
        f.write("\n")

        # Write the + line with number of satellites (1) at positions 3-6
        f.write(
            f"+   1  {satellite_id}  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n")

        # Standard format SP3 header comment lines
        f.write("++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n")
        f.write("%c M  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n")
        f.write("%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n")
        f.write("%f  1.2500000  1.025000000  0.00000000000  0.000000000000000\n")
        f.write("%i    0    0    0    0      0      0      0      0         0\n")
        f.write("%i    0    0    0    0      0      0      0      0         0\n")
        f.write("/* GNSS single point positioning solution\n")
        f.write("/* Iterative Reweighted Least Squares (IRLS)\n")
        f.write("/*\n")  # End of header marker

        # Write Data Records
        for i in range(num_epochs):
            epoch_time = pd.to_datetime(dataset['time'].values[i])
            year = epoch_time.year
            month = epoch_time.month
            day = epoch_time.day
            hour = epoch_time.hour
            minute = epoch_time.minute
            second_float = epoch_time.second + epoch_time.microsecond / 1e6

            # Write epoch header line
            f.write(
                f"*  {year:4d} {month:2d} {day:2d} {hour:2d} {minute:2d} {second_float:11.8f}\n")

            # Position data - extract position array at time i first
            pos = dataset['position'].isel(time=i)
            pos_x, pos_y, pos_z = pos.sel(ECEF="x").item(), pos.sel(
                ECEF="y").item(), pos.sel(ECEF="z").item()
            vel = dataset['velocity'].isel(time=i)
            vel_x, vel_y, vel_z = vel.sel(ECEF="x").item(), vel.sel(
                ECEF="y").item(), vel.sel(ECEF="z").item()
            clk_bias = dataset['clock'].isel(time=i).item()

            # Write satellite data with exact SP3 format spacing
            f.write(
                f"P{satellite_id}{pos_x:14.6f}{pos_y:14.6f}{pos_z:14.6f}{clk_bias:14.6f}\n")
            f.write(
                f"V{satellite_id}{vel_x:14.6f}{vel_y:14.6f}{vel_z:14.6f}{0.000000:14.6f}\n")

        f.write("EOF\n")  # End of file marker
