import xarray as xr
import pandas as pd

def write_sp3_from_xarray(dataset: xr.Dataset, output_filepath: str, satellite_id: str = "G01"):
    """
    Writes an xarray Dataset to an SP3 format text file.

    Args:
        dataset (xr.Dataset): The input xarray Dataset.
                              It must contain 'time', 'position' (with ECEF coordinates),
                              and 'clock_bias' data variables.
        output_filepath (str): The path to the output SP3 file.
        satellite_id (str): The satellite ID to use in the SP3 file (e.g., "G01", "E12").
    """

    with open(output_filepath, 'w') as f:
        # --- Write SP3 Header (Simplified Example) ---
        f.write("#c SP3-c File: Created using geonss python library\n") # Version and type
        # A more complete header would include start epoch, number of epochs, coordinate system, etc.
        # For simplicity, we'll use a basic start time from the dataset.
        start_time = pd.to_datetime(dataset['time'].values[0])
        num_epochs = len(dataset['time'])

        f.write(f"#c {start_time.year:04d} {start_time.month:02d} {start_time.day:02d} {start_time.hour:02d} "
                f"{start_time.minute:02d} {start_time.second + start_time.microsecond/1e6:011.8f} "
                f"GPS {num_epochs:4d} {' '*27}\n") # Simplified time line

        f.write("+ NSAT\n") # Number of satellites line
        f.write(f"  1 {satellite_id}\n") # Assuming only one satellite as per the dataset structure for now

        f.write("#c GNSS single point positioning solution\n")
        f.write("#c Iterative Reweighted Least Squares (IRLS)\n")

        f.write("##\n") # End of header marker

        # --- Write Data Records ---
        for i in range(num_epochs):
            epoch_time = pd.to_datetime(dataset['time'].values[i])
            year = epoch_time.year
            month = epoch_time.month
            day = epoch_time.day
            hour = epoch_time.hour
            minute = epoch_time.minute
            second_float = epoch_time.second + epoch_time.microsecond / 1e6

            # Write epoch header line
            f.write(f"* {year:4d} {month:2d} {day:2d} {hour:2d} {minute:2d} {second_float:11.8f}\n")

            # Position data (assuming ECEF coordinates are x, y, z in order)
            # The dataset shows ECEF as a coordinate with values 'x', 'y', 'z'
            # We need to extract them based on this structure.
            pos_x = dataset['position'].sel(ECEF='x').isel(time=i).item()
            pos_y = dataset['position'].sel(ECEF='y').isel(time=i).item()
            pos_z = dataset['position'].sel(ECEF='z').isel(time=i).item()

            # Clock bias data
            clk_bias = dataset['clock_bias'].isel(time=i).item()

            # Write satellite data line
            # Format: P<SAT_ID> <X_POS> <Y_POS> <Z_POS> <CLOCK_BIAS> [X_VEL Y_VEL Z_VEL CLOCK_RATE_CHANGE] [ACC_FLAGS]
            # Velocities, clock rate change, and accuracy flags are optional and not in the provided dataset.
            f.write(f"P{satellite_id:<3s}{pos_x:14.6f}{pos_y:14.6f}{pos_z:14.6f}{clk_bias:14.6f}\n")
            # If you had more satellites, you would loop through them here for the same epoch.

        f.write("EOF\n") # End of file marker

    print(f"SP3 file '{output_filepath}' written successfully.")
