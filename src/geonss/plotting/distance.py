import numpy as np
import matplotlib.pyplot as plt

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

