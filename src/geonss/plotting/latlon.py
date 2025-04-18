import matplotlib.pyplot as plt
import os
from geonss.coordinates import LLAPosition
from typing import List


def create_latlon_plot(lat_min, lat_max, lon_min, lon_max):
    """
    Create and configure a latitude-longitude plot for geographic mapping.

    Args:
        lat_min: Minimum latitude boundary
        lat_max: Maximum latitude boundary
        lon_min: Minimum longitude boundary
        lon_max: Maximum longitude boundary

    Returns:
        tuple: Figure and axes objects for the lat-lon plot
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Configure lat-lon plot
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.grid(True, linestyle='--', alpha=0.6, zorder=1)
    ax.ticklabel_format(useOffset=False, style='plain')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    return fig, ax


def plot_latlon_points(ax, positions, color='blue', size=5, label='Points', zorder=1):
    """
    Plot a collection of latitude-longitude points on the given axes.

    Args:
        ax: Matplotlib axes object
        positions: List of LLAPosition objects containing lat-lon coordinates
        color: Point color
        size: Point size
        label: Legend label
        zorder: Z-order for drawing

    Returns:
        Scatter plot object
    """
    if not positions:
        return None

    lons = [float(pos.longitude) for pos in positions]
    lats = [float(pos.latitude) for pos in positions]
    return ax.scatter(lons, lats, color=color, s=size, label=label, zorder=zorder)


def save_latlon_plot(fig, filename='gnss_positions.svg'):
    """
    Save the latitude-longitude plot to a file and close the figure.

    Args:
        fig: Matplotlib figure object
        filename: Output file name

    Returns:
        str: Absolute path to the saved file
    """
    plt.legend()
    plt.savefig(filename, format='svg', bbox_inches='tight')
    plt.close(fig)
    return os.path.abspath(filename)

def plot_positions_in_latlon(
        true_position: LLAPosition,
        computed_positions: List[LLAPosition],
        margin=0.0001,
        filename='lla_positions.svg',
) -> str:
    """
    Plot true and computed positions on a latitude-longitude map and save as SVG.

    Args:
        true_position: LLAPosition object of true position (lat-lon coordinates)
        computed_positions: List of LLAPosition objects for computed positions (lat-lon coordinates)
        filename: Output SVG file name
        margin: Map margin in degrees around the lat-lon boundaries

    Returns:
        str: Absolute path to the saved file
    """
    # Calculate lat-lon map bounds
    all_lats = [float(true_position.latitude)] + [float(pos.latitude) for pos in computed_positions]
    all_lons = [float(true_position.longitude)] + [float(pos.longitude) for pos in computed_positions]
    lat_min, lat_max = min(all_lats) - margin, max(all_lats) + margin
    lon_min, lon_max = min(all_lons) - margin, max(all_lons) + margin

    # Create latitude-longitude coordinate system
    fig, ax = create_latlon_plot(lat_min, lat_max, lon_min, lon_max)

    # Plot lat-lon points
    plot_latlon_points(
        ax,
        computed_positions,
        color='blue',
        size=5,
        label='Computed Positions',
        zorder=1
    )
    plot_latlon_points(
        ax,
        [true_position],
        color='red',
        size=25,
        label='True Position',
        zorder=3
    )

    # Save lat-lon plot
    return save_latlon_plot(fig, filename)