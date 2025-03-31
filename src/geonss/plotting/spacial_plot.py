import matplotlib.pyplot as plt
import os
from geonss.coordinates import LLAPosition
from typing import List


def create_coordinate_system(lat_min, lat_max, lon_min, lon_max):
    """
    Create and configure the coordinate system for the map.

    Args:
        lat_min: Minimum latitude
        lat_max: Maximum latitude
        lon_min: Minimum longitude
        lon_max: Maximum longitude

    Returns:
        tuple: Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Configure plot
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.grid(True, linestyle='--', alpha=0.6, zorder=1)
    ax.ticklabel_format(useOffset=False, style='plain')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    return fig, ax


def plot_points(ax, positions, color='blue', size=5, label='Points', zorder=1):
    """
    Plot a collection of points on the given axes.

    Args:
        ax: Matplotlib axes object
        positions: List of LLAPosition objects
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


def plot_point(ax, position, color='red', size=25, label='Point', zorder=3):
    """
    Plot a single point on the given axes.

    Args:
        ax: Matplotlib axes object
        position: LLAPosition object
        color: Point color
        size: Point size
        label: Legend label
        zorder: Z-order for drawing

    Returns:
        Scatter plot object
    """
    return ax.scatter(float(position.longitude), float(position.latitude), color=color, s=size, label=label, zorder=zorder)


def save_plot(fig, filename='gnss_positions.svg'):
    """
    Save the plot to a file and close the figure.

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

def plot_coordinates_on_map(
        true_position: LLAPosition,
        computed_position: LLAPosition,
        computed_positions: List[LLAPosition],
        filename='gnss_positions.svg',
        margin=0.0001
) -> str:
    """
    Plot true and computed positions on a simple map and save as SVG.

    Args:
        true_position: LLAPosition object of true position
        computed_position: LLAPosition object of computed position
        computed_positions: List of LLAPosition objects for computed positions
        filename: Output SVG file name
        margin: Map margin in degrees

    Returns:
        str: Absolute path to the saved file
    """
    # Calculate map bounds
    all_lats = [float(true_position.latitude), float(computed_position.latitude)] + [float(pos.latitude) for pos in computed_positions]
    all_lons = [float(true_position.longitude), float(computed_position.longitude)] + [float(pos.longitude) for pos in computed_positions]
    lat_min, lat_max = min(all_lats) - margin, max(all_lats) + margin
    lon_min, lon_max = min(all_lons) - margin, max(all_lons) + margin

    # Create coordinate system
    fig, ax = create_coordinate_system(lat_min, lat_max, lon_min, lon_max)

    # Plot points
    plot_points(ax, computed_positions, color='blue', size=5, label='Computed Positions', zorder=1)
    plot_point(ax, computed_position, color='green', size=25, label='Computed Position', zorder=2)
    plot_point(ax, true_position, color='red', size=25, label='True Position', zorder=3)

    # Save plot
    return save_plot(fig, filename)