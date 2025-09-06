import os
import matplotlib.pyplot as plt
import numpy as np

def plot_distance(distance_da, connect_points=False):
    """
    Generates a scatter or line plot of distance over time for each satellite.
    Legend entries are sorted by the average distance of each satellite (descending).

    Args:
        distance_da (xr.DataArray): DataArray with 'time' and 'sv' dimensions.
        connect_points (bool, optional): If True, connect points with lines.
                                         Defaults to False (scatter plot).
    """
    # Calculate the mean distance for each satellite (sv)
    mean_distances = distance_da.mean(dim='time')

    # Get the satellite IDs (sv coordinate values) sorted by their mean distance
    # Argsort gives indices for ascending order
    ascending_indices = np.argsort(mean_distances.values)
    # Reverse the indices to get descending order (largest average first)
    sorted_sv_indices = ascending_indices[::-1]

    sorted_sv_names = distance_da['sv'].values[sorted_sv_indices]

    # Reorder the original DataArray according to the sorted satellite names
    # This ensures the plotting function processes them in the desired order for the legend
    sorted_distance_da = distance_da.sel(sv=sorted_sv_names)

    # Basic plot setup
    plt.figure(figsize=(24, 12))

    if connect_points:
        # Generate the line plot using the sorted data
        sorted_distance_da.plot.line(
            x='time',
            hue='sv',
        )
    else:
        # Generate the scatter plot using the sorted data
        sorted_distance_da.plot.scatter(
            x='time',
            hue='sv',
            s=1,
            edgecolors='face',
        )

    # Basic labels and title
    plt.title("Satellite Distance Variation")
    plt.xlabel("Time")
    plt.ylabel("Distance (m)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def latlon_distance(diff: list, use_arcseconds=False, figsize=(24, 12), filename='latlon_differences.svg'):
    """
    Plot coordinate differences in a scatter plot with zero-centered crosshair and save it.

    Args:
        diff: List of tuples containing (latitude_diff, longitude_diff, altitude_diff)
        use_arcseconds: If True, convert to arcseconds (default: False)
        figsize: Figure size as tuple (width, height) (default: (24, 12))
        filename: Output file name (default: 'latlon_differences.svg')

    Returns:
        str: Absolute path to the saved file
    """
    # Extract dx (longitude) and dy (latitude) components from 'diff'
    # Swapped to correctly map longitude to x-axis and latitude to y-axis
    lon_diff = [item[1] for item in diff]  # longitude on x-axis
    lat_diff = [item[0] for item in diff]  # latitude on y-axis

    # Convert to arcseconds if requested (1 degree = 3600 arcseconds)
    if use_arcseconds:
        lon_diff = [d * 3600 for d in lon_diff]
        lat_diff = [d * 3600 for d in lat_diff]

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot of the differences
    ax.scatter(lon_diff, lat_diff, label='Coordinate Differences', color='blue', marker='o', s=10)

    # Set plot limits
    max_val_x = np.max(np.abs(lon_diff)) if lon_diff else 0
    max_val_y = np.max(np.abs(lat_diff)) if lat_diff else 0
    max_val = max(max_val_x, max_val_y)
    plot_limit = max_val * 1.05

    ax.set_xlim(-plot_limit, plot_limit)
    ax.set_ylim(-plot_limit, plot_limit)

    # Set aspect ratio to be equal to make it a "round silhouette"
    ax.set_aspect('equal', adjustable='box')

    # Add a zero-centered crosshair for reference
    ax.axhline(0, color='black', lw=0.75, linestyle='--')
    ax.axvline(0, color='black', lw=0.75, linestyle='--')

    # Add labels and title
    unit = "arcseconds" if use_arcseconds else "degrees"
    ax.set_xlabel(f"Longitude Difference [{unit}]")
    ax.set_ylabel(f"Latitude Difference [{unit}]")
    ax.set_title("Zero-Centered Coordinate Differences")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.7)

    # Save plot
    plt.tight_layout()
    plt.savefig(filename, format='svg', bbox_inches='tight')
    plt.close(fig)

    return os.path.abspath(filename)
