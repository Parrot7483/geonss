import multipledispatch as md
import numpy as np
import scipy as sp
import xarray as xr

from geonss.constellation import get_constellation, Constellation
from geonss.correction import apply_phase_center_offset


def _collapse_sparse_dimension(ds: xr.Dataset, dim: str) -> xr.Dataset:
    """
    Collapses a specified dimension of a Dataset, keeping the single non-NaN value.

    This function applies to all data variables in the dataset that contain the
    specified dimension. It assumes that for any combination of other dimensions,
    there should be at most one valid (non-NaN) value along the specified dimension.

    Args:
        ds: The input xarray Dataset.
        dim: The name of the dimension to collapse (e.g., 'time_bins').

    Returns:
        A new Dataset with the specified dimension removed from all data variables.

    Raises:
        ValueError: If any slice along the dimension contains more than one non-NaN value.
    """
    # Create an empty dataset without the dimension to be collapsed
    collapsed_ds = xr.Dataset(coords={
        k: v for k, v in ds.coords.items() if k != dim and dim not in v.dims
    })

    # Process each data variable
    for var_name, da in ds.data_vars.items():
        if dim in da.dims:
            # Count the number of non-NaN values along the specified dimension
            valid_counts = da.count(dim=dim)

            # Check if any slice had more than one valid value
            if np.any(valid_counts > 1):
                invalid_indices = np.argwhere(valid_counts.values > 1)
                raise ValueError(
                    f"Variable '{var_name}': Dimension '{dim}' cannot be collapsed because "
                    f"at least one slice along it contains more than one non-NaN value. "
                    f"Example problematic index location(s): {invalid_indices[:5]}"
                )

            # If the check passes, perform the reduction
            collapsed_ds[var_name] = da.max(dim=dim, skipna=True)
        else:
            # Copy variables that don't have the dimension
            collapsed_ds[var_name] = da

    # Preserve dataset attributes
    collapsed_ds.attrs.update(ds.attrs)

    return collapsed_ds


def correct_positions_with_antex(pos_ds: xr.Dataset, antex_ds: xr.Dataset) -> xr.Dataset:
    """
    Corrects satellite ECEF positions using ANTEX Phase Center Offset (PCO) data.

    This function iterates through each satellite in the input position dataset,
    finds the corresponding PCO data from the ANTEX dataset based on the satellite
    system's default frequency, validates the timeliness of the ANTEX data,
    and applies the PCO correction to the satellite's ECEF position vectors.

    Args:
        pos_ds: An xarray Dataset containing satellite position data.
                Must have dimensions ('time', 'sv') and a data variable
                'position' with shape (time, sv, 3) representing ECEF coordinates.
        antex_ds: An xarray Dataset containing ANTEX PCO data. Must have
                  dimensions ('sv', 'frequency', 'time') and data variables
                  'offset' (shape: sv, frequency, time, 3) for NEU offsets
                  and 'valid_until' (shape: sv, frequency, time) for validity timestamps.

    Returns:
        An xarray Dataset with the same structure as pos_ds, but
        with the 'position' data variable updated with the PCO-corrected
        ECEF coordinates.

    Raises:
        ValueError: If input datasets lack required variables or dimensions,
                    or if ANTEX data for a required time point is outdated.
        RuntimeError: For unexpected issues during processing.
    """
    # Input Validation (structure and content)
    if 'position' not in pos_ds.data_vars or 'sv' not in pos_ds.coords or 'time' not in pos_ds.coords:
        raise ValueError("pos_ds Dataset is missing required variables/coordinates ('position', 'sv', 'time').")
    if 'offset' not in antex_ds.data_vars or 'valid_until' not in antex_ds.data_vars or 'sv' not in antex_ds.coords or 'frequency' not in antex_ds.coords or 'time' not in antex_ds.coords:
        raise ValueError(
            "antex_ds Dataset is missing required variables/coordinates ('offset', 'valid_until', 'sv', 'frequency', 'time').")

    def _process_satellite_group(satellite_position_group: xr.Dataset):
        """
        Process a dataset containing position data for a single satellite.
        """

        satellite_id: str = satellite_position_group.sv.values[0]
        single_satellite_data = satellite_position_group.squeeze('sv')  # Remove sv dimension

        # Determine Constellation and Default Frequency
        constellation = get_constellation(satellite_id)
        if constellation == Constellation.GALILEO:
            default_frequency = 'E01'
        elif constellation == Constellation.GPS:
            default_frequency = 'G01'
        elif constellation == Constellation.BEIDOU:
            default_frequency = 'C01'
        else:
            raise ValueError(f"Unsupported constellation: {constellation} for satellite {satellite_id}")

        # Select and Validate ANTEX Data
        try:
            satellite_antex_data = antex_ds.sel(sv=satellite_id, frequency=default_frequency).dropna(dim='time',
                                                                                                     subset=['offset'],
                                                                                                     how='all')
        except KeyError:
            raise ValueError(f"Could not find ANTEX data for SV={satellite_id}, Freq={default_frequency}")

        if satellite_antex_data.time.size == 0:
            raise ValueError(
                f"No valid ANTEX offset data found for SV={satellite_id}, Freq={default_frequency} after dropping NaNs.")

        # Pad-select ANTEX data corresponding to the position timestamps
        try:
            selected_antex_valid_until = satellite_antex_data.valid_until.sel(time=single_satellite_data.time,
                                                                              method='pad')
            selected_antex_offsets = satellite_antex_data.offset.sel(time=single_satellite_data.time, method='pad')
        except KeyError as e:
            raise ValueError(
                f"Error selecting ANTEX data for SV={satellite_id} at times {single_satellite_data.time.values}. Missing time points? Error: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during ANTEX data selection for {satellite_id}: {e}")

        # Check if any selected ANTEX data record has expired
        if np.any(selected_antex_valid_until.values < single_satellite_data.time.values):
            invalid_times = single_satellite_data.time[
                selected_antex_valid_until.values < single_satellite_data.time.values].values
            raise ValueError(f"Satellite {satellite_id} requires ANTEX data for times {invalid_times}, "
                             f"but the available ANTEX data (selected via padding) is no longer valid for these times.")

        # Apply Correction
        original_positions = single_satellite_data.position.values
        neu_offsets = selected_antex_offsets.values

        corrected_positions: np.ndarray = apply_phase_center_offset(original_positions, neu_offsets)

        # Update the position data in the dataset copy
        processed_group = single_satellite_data.copy()
        processed_group['position'].values = corrected_positions

        return processed_group

    # Group by satellite and apply the processing function to each group
    corrected_positions_ds = pos_ds.groupby('sv').map(_process_satellite_group)

    # Transposed corrected positions to match original dataset structure
    transposed_positions_ds = corrected_positions_ds.transpose('time', 'sv', 'ECEF')

    # Ensure descriptions are accurate after correction
    transposed_positions_ds['position'].attrs.update({
        'description': 'Interpolated satellite position corrected with ANTEX Phase Center Offset',
        'units': 'meters',
    })

    return transposed_positions_ds


@md.dispatch(xr.Dataset, xr.Dataset)
def interpolate_orbit_positions(
        sp3_full_data: xr.Dataset,
        query_times: xr.Dataset,
) -> xr.Dataset:
    """
    Wrapper to call with default window size of 8.
    """
    return interpolate_orbit_positions(sp3_full_data, query_times, 8)


@md.dispatch(xr.Dataset, xr.Dataset, int)
def interpolate_orbit_positions(
        sp3_full_data: xr.Dataset,
        query_times: xr.Dataset,
        window: int,
) -> xr.Dataset:
    """
    Interpolates satellite positions and velocities from SP3 data onto query times.

    Uses Lagrange polynomials fitted to a specified window of SP3 data points
    centered around the query time interval. Calculates velocity by taking the
    analytical derivative of the position polynomials.

    Args:
        sp3_full_data: xarray Dataset containing precise orbit data (e.g., from SP3 file).
                  Expected coordinates: 'time', 'sv', 'ECEF'.
                  Required variables: 'position' (in km).
        query_times: xarray Dataset containing the satellite vehicle IDs ('sv') and
                     times ('time') at which to interpolate the position and velocity.
                     Expected coordinates: 'time', 'sv'.
        window: The number of SP3 data points to use for the Lagrange interpolation window.
                Must be at least 2. Default is 8. A larger window uses more SP3 points.

    Returns:
        A new xarray Dataset containing the interpolated positions (in meters)
        and velocities (in meters per second).
        Coordinates: 'time', 'sv', 'ECEF'.
        Variables: 'position', 'velocity'.

    Raises:
        ValueError: If a group in query_times contains multiple satellites (internal check).
        ValueError: If the specified window size is less than 2 or higher than 20.
        ValueError: If not enough SP3 points are available near the query times
                    to form the requested window size for a satellite.
        KeyError: If query times fall completely outside the time range of the SP3 data
                  for a given satellite.
    """
    # TODO: Calculate drift
    # TODO: Add parallelization option
    # TODO: Add possibility to extrapolate positions using an option
    # TODO: Add possibility to use a different interpolation method (e.g., cubic (hermitian) spline)
    # TODO: Evaluate using BarycentricInterpolator (faster evaluation but can not use derivative)

    # Validate window size argument
    if window < 2 or window > 20:
        raise ValueError("Window size must be at least 2 and at most 20 for Lagrange interpolation")

    def _calculate_slice_indices(
            sv_time_group: xr.Dataset,
            sp3_sv_data: xr.Dataset,
            satellite_id: str,
    ):
        # Get the first and last elements in the bin
        group_first_time: np.datetime64 = sv_time_group.time.min().values
        group_last_time: np.datetime64 = sv_time_group.time.max().values

        # Find SP3 points bracketing the time bucket
        try:
            # Find the SP3 time point immediately before or at the start of the group interval
            sp3_left_bracket_time = sp3_sv_data.time.sel(time=group_first_time, method='pad').values
            # Find the SP3 time point immediately after or at the end of the group interval
            sp3_right_bracket_time = sp3_sv_data.time.sel(time=group_last_time, method='backfill').values
        except KeyError:
            # This occurs if the group times are entirely outside the SP3 time range
            raise KeyError(
                f"Query times for SV {satellite_id} ({group_first_time} to {group_last_time}) "
                f"fall completely outside the available SP3 time range "
                f"({sp3_sv_data.time.min().values} to {sp3_sv_data.time.max().values})."
            )

        # Get the integer indices in the SP3 time array corresponding to these bracketing times
        sp3_left_bracket_index = np.int64(np.where(sp3_sv_data.time.values == sp3_left_bracket_time)[0][0])
        sp3_right_bracket_index = np.int64(np.where(sp3_sv_data.time.values == sp3_right_bracket_time)[0][0])

        # Calculate window start/end indices based on the desired window size
        half_window_floor = window // 2
        # Use ceil for the upper half to handle odd window sizes correctly
        half_window_ceil = (window + 1) // 2

        # Calculate initial window boundaries
        num_sp3_points = len(sp3_sv_data.time)
        start_index = max(0, sp3_left_bracket_index - half_window_floor)
        end_index = min(num_sp3_points, sp3_right_bracket_index + half_window_ceil)

        # Extend the window if it is too small
        if (end_index - start_index) < window:
            if start_index == 0 and end_index < num_sp3_points:
                # Hit the beginning, try extending the end
                end_index = min(num_sp3_points, start_index + window)
            elif end_index == num_sp3_points and start_index > 0:
                # Hit the end, try extending the start
                start_index = max(0, end_index - window)

        # Ensure we have enough points after clamping and adjustments
        if (end_index - start_index) < window:
            raise ValueError(
                f"Cannot select {window} SP3 points for SV {satellite_id} "
                f"near time interval {group_first_time} to {group_last_time}. "
                f"Only {end_index - start_index} points available after clamping to SP3 data bounds "
                f"({sp3_sv_data.time.min().values} to {sp3_sv_data.time.max().values}). "
                f"Consider reducing window size or checking SP3 data coverage."
            )

        return start_index, end_index

    def _interpolate_group(sv_time_group: xr.Dataset) -> xr.Dataset:
        """
        Internal helper function to perform interpolation for a single satellite's time group.

        Selects a window of SP3 data, performs boundary checks, fits Lagrange
        polynomials, calculates derivatives, interpolates position/velocity,
        and returns results in a Dataset.
        """
        # Check that ds only contains one satellite
        if len(np.unique(sv_time_group.sv.values)) != 1:
            raise ValueError("Dataset must contain only one satellite")

        # Extract the single satellite ID for easier access and clearer error messages
        satellite_id = sv_time_group.sv.values[0]

        # Select the SP3 data relevant to this specific satellite
        sp3_sv_data = sp3_full_data.sel(sv=satellite_id)

        start_index, end_index = _calculate_slice_indices(sv_time_group, sp3_sv_data, satellite_id)

        # Create the slice object for selecting the window data
        sp3_window_indices = slice(start_index, end_index)

        # Select the subset of SP3 data corresponding to the calculated window
        sp3_sv_window_data = sp3_sv_data.isel(time=sp3_window_indices)

        # Convert datetime64 to relative time in seconds for numerical stability
        # Use the first time point *in the selected window* as the reference time
        reference_time_ns = sp3_sv_window_data.time.values[0].astype(np.int64)

        # Calculate SP3 times within the window relative to the reference time, in seconds
        sp3_times_relative_sec = (sp3_sv_window_data.time.values.astype(np.int64) - reference_time_ns) / 1e9

        # Calculate query times within the group relative to the same reference time, in seconds
        group_times_relative_sec = (sv_time_group.time.values.astype(np.int64) - reference_time_ns) / 1e9

        # Extract ECEF coordinates from the SP3 window data
        # Convert position from kilometers (SP3 standard) to meters
        x_coords_m = sp3_sv_window_data.sel(ECEF='x').position.values * 1000.0
        y_coords_m = sp3_sv_window_data.sel(ECEF='y').position.values * 1000.0
        z_coords_m = sp3_sv_window_data.sel(ECEF='z').position.values * 1000.0

        # Extract clock bias from the SP3 window data
        clock = sp3_sv_window_data.clock.values

        try:
            # Create the Lagrange polynomials for each ECEF coordinate
            lagrange_poly_x = sp.interpolate.lagrange(sp3_times_relative_sec, x_coords_m)
            lagrange_poly_y = sp.interpolate.lagrange(sp3_times_relative_sec, y_coords_m)
            lagrange_poly_z = sp.interpolate.lagrange(sp3_times_relative_sec, z_coords_m)

            # Interpolate positions (in meters) at the relative query times
            interpolated_x_m = lagrange_poly_x(group_times_relative_sec)
            interpolated_y_m = lagrange_poly_y(group_times_relative_sec)
            interpolated_z_m = lagrange_poly_z(group_times_relative_sec)
        except ValueError as e:
            raise ValueError(f"Error creating Lagrange polynomial for SV {satellite_id}: {e}")

        try:
            # Calculate the derivatives of the Lagrange polynomials for velocity
            lagrange_poly_x_derivative = lagrange_poly_x.deriv()
            lagrange_poly_y_derivative = lagrange_poly_y.deriv()
            lagrange_poly_z_derivative = lagrange_poly_z.deriv()

            # Interpolate velocities (in meters per second) at the relative query times
            interpolated_vx_m_per_s = lagrange_poly_x_derivative(group_times_relative_sec)
            interpolated_vy_m_per_s = lagrange_poly_y_derivative(group_times_relative_sec)
            interpolated_vz_m_per_s = lagrange_poly_z_derivative(group_times_relative_sec)
        except ValueError as e:
            raise ValueError(f"Error creating Lagrange polynomial derivative for SV {satellite_id}: {e}")

        try:
            # Create the Lagrange polynomial for clock bias
            lagrange_poly_clock = sp.interpolate.lagrange(sp3_times_relative_sec, clock)

            # Interpolate clock bias at the relative query times
            interpolated_clock = lagrange_poly_clock(group_times_relative_sec)
        except ValueError as e:
            raise ValueError(f"Error creating Lagrange polynomial for clock bias for SV {satellite_id}: {e}")

        return xr.Dataset(
            coords={
                'time': sv_time_group.time.values,
                'ECEF': ['x', 'y', 'z'],
            },
            data_vars={
                'position': xr.DataArray(
                    data=np.stack([interpolated_x_m, interpolated_y_m, interpolated_z_m], axis=-1),
                    dims=('time', 'ECEF'),
                    coords={
                        'time': sv_time_group.time.values,
                        'ECEF': ['x', 'y', 'z'],
                    },
                ),
                'velocity': xr.DataArray(
                    data=np.stack([interpolated_vx_m_per_s, interpolated_vy_m_per_s, interpolated_vz_m_per_s], axis=-1),
                    dims=('time', 'ECEF'),
                    coords={
                        'time': sv_time_group.time.values,
                        'ECEF': ['x', 'y', 'z'],
                    },
                ),
                'clock': xr.DataArray(
                    data=interpolated_clock,
                    dims='time',
                    coords={
                        'time': sv_time_group.time.values,
                    },
                ),
            },
        )

    # Group the query times by satellite vehicle ID and time bin
    grouped_result = query_times.groupby(
        sv=xr.groupers.UniqueGrouper(),
        time=xr.groupers.BinGrouper(bins=sp3_full_data.time.values),
    ).map(_interpolate_group)

    # Collapse the time_bins dimension
    reduced_result = _collapse_sparse_dimension(grouped_result, dim='time_bins')

    # Add attributes
    annotated_result = reduced_result.assign_attrs({
        'interpolation_method': f'Lagrange polynomial (window size: {window})',
        'source': 'SP3 precise orbit data and query times',
    })

    annotated_result['position'] = reduced_result.position.assign_attrs({
        'units': 'meters',
        'description': 'Interpolated satellite position',
    })

    annotated_result['velocity'] = reduced_result.velocity.assign_attrs({
        'units': 'meters per second',
        'description': 'Interpolated satellite velocity',
    })

    annotated_result['clock'] = reduced_result.clock.assign_attrs({
        'units': 'microseconds',
        'description': 'Interpolated clock bias',
    })

    transposed_result = annotated_result.transpose('time', 'sv', 'ECEF')

    return transposed_result


@md.dispatch(xr.Dataset, xr.Dataset, xr.Dataset)
def interpolate_orbit_positions(
        sp3_full_data: xr.Dataset,
        query_times: xr.Dataset,
        antex: xr.Dataset,
) -> xr.Dataset:
    """
    Wrapper to call with default window size of 8.
    """
    return interpolate_orbit_positions(sp3_full_data, query_times, antex, 8)


@md.dispatch(xr.Dataset, xr.Dataset, xr.Dataset, int)
def interpolate_orbit_positions(
        sp3_full_data: xr.Dataset,
        query_times: xr.Dataset,
        antex_ds: xr.Dataset,
        window: int,
) -> xr.Dataset:
    """
    Interpolates satellite positions, velocities, and clock biases from SP3 data
    onto query times and subsequently corrects the positions using ANTEX PCO data.

    This function first performs Lagrange interpolation on the SP3 data to estimate
    satellite states (position, velocity, clock bias) at the specified query times.
    It then applies Antenna Phase Center Offset (PCO) corrections to the
    interpolated ECEF positions using the provided ANTEX dataset.

    Args:
        sp3_full_data: xarray Dataset containing precise orbit data (e.g., from SP3 file).
                  Expected coordinates: 'time', 'sv', 'ECEF'.
                  Required variables: 'position' (in km), 'clock' (in microseconds).
        query_times: xarray Dataset containing the satellite vehicle IDs ('sv') and
                     times ('time') at which to interpolate the state.
                     Expected coordinates: 'time', 'sv'.
        antex_ds: An xarray Dataset containing ANTEX PCO data. Must have
                  dimensions ('sv', 'frequency', 'time') and data variables
                  'offset' (shape: sv, frequency, time, 3) for NEU offsets
                  and 'valid_until' (shape: sv, frequency, time) for validity timestamps.
                  Used to correct the interpolated positions.
        window: The number of SP3 data points to use for the Lagrange interpolation window.
                Must be at least 2 and at most 20. Default is 8.

    Returns:
        A new xarray Dataset containing the interpolated velocities and clock biases,
        and the interpolated *and* PCO-corrected positions.
        Coordinates: 'time', 'sv', 'ECEF'.
        Variables:
            'position': PCO-corrected interpolated ECEF position (meters).
            'velocity': Interpolated ECEF velocity (meters per second).
            'clock': Interpolated clock bias (microseconds).

    Raises:
        ValueError: If input datasets lack required variables/dimensions.
        ValueError: If the specified window size is invalid.
        ValueError: If not enough SP3 points are available for interpolation.
        ValueError: If ANTEX data for a required time point is outdated or missing.
        KeyError: If query times fall outside the SP3 data range for a satellite.
        RuntimeError: For unexpected issues during processing (e.g., ANTEX selection).
    """
    # Interpolate positions, velocities, and clock biases using the base function
    interpolated_ds = interpolate_orbit_positions(sp3_full_data, query_times, window)

    # Correct the interpolated positions using the ANTEX data
    # The correct_positions_with_antex function modifies the 'position' variable
    # and keeps other variables (like velocity, clock) if they exist.
    corrected_ds = correct_positions_with_antex(interpolated_ds, antex_ds)

    # Update attributes to reflect both interpolation and correction
    corrected_ds.attrs.update({
        'interpolation_method': f'Lagrange polynomial (window size: {window})',
        'position_correction': 'ANTEX Phase Center Offset (PCO)',
        'source': 'SP3 precise orbit data, query times, and ANTEX data',
    })

    return corrected_ds
