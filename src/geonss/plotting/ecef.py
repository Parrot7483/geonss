import matplotlib.pyplot as plt
import os
from typing import List, Optional

from geonss.coordinates import ECEFPosition


def create_ecef_plot(positions: List[ECEFPosition], margin: float = 25.0):
    """
    Create and configure a 3D ECEF plot.

    Args:
        positions: List of ECEFPosition objects to determine plot boundaries
        margin: Additional space around the points in meters

    Returns:
        tuple: Figure and axes objects for the 3D ECEF plot
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Calculate bounds
    all_x = [float(pos.x) for pos in positions]
    all_y = [float(pos.y) for pos in positions]
    all_z = [float(pos.z) for pos in positions]

    x_min, x_max = min(all_x) - margin, max(all_x) + margin
    y_min, y_max = min(all_y) - margin, max(all_y) + margin
    z_min, z_max = min(all_z) - margin, max(all_z) + margin

    # Set axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    # Configure plot
    ax.set_xlabel('ECEF X (m)')
    ax.set_ylabel('ECEF Y (m)')
    ax.set_zlabel('ECEF Z (m)')
    ax.grid(True)

    return fig, ax


def plot_ecef_points(
        ax,
        positions: List[ECEFPosition],
        color: str = 'blue',
        size: float = 5,
        label: str = 'Points',
        zorder: int = 1
) -> Optional[plt.Artist]:
    """
    Plot a collection of ECEF points on the given 3D axes.

    Args:
        ax: Matplotlib 3D axes object
        positions: List of ECEFPosition objects
        color: Point color
        size: Point size
        label: Legend label
        zorder: Z-order for drawing

    Returns:
        Scatter plot object or None if positions is empty
    """
    if not positions:
        return None

    x_coords = [float(pos.x) for pos in positions]
    y_coords = [float(pos.y) for pos in positions]
    z_coords = [float(pos.z) for pos in positions]

    return ax.scatter(x_coords, y_coords, z_coords, color=color, s=size, label=label, zorder=zorder)


def save_ecef_plot(fig, filename: str = 'ecef_positions.svg') -> str:
    """
    Save the ECEF plot to a file and close the figure.

    Args:
        fig: Matplotlib figure object
        filename: Output file name

    Returns:
        str: Absolute path to the saved file
    """
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, format='svg', bbox_inches='tight')
    plt.close(fig)
    return os.path.abspath(filename)


def plot_positions_in_ecef(
        true_position: ECEFPosition,
        computed_positions: List[ECEFPosition],
        filename: str = 'ecef_positions.svg',
        margin: float = 25.0,
) -> str:
    """
    Plot true and computed positions in ECEF coordinate system and save as SVG.

    Args:
        true_position: ECEFPosition object of true position
        computed_positions: List of ECEFPosition objects for computed positions
        filename: Output SVG file name
        margin: Map margin in meters around the ECEF boundaries

    Returns:
        str: Absolute path to the saved file
    """
    # Combine all positions for calculating bounds
    all_positions = [true_position] + computed_positions

    # Create 3D ECEF coordinate system
    fig, ax = create_ecef_plot(all_positions, margin)

    # Plot ECEF points
    plot_ecef_points(
        ax,
        computed_positions,
        color='blue',
        size=5,
        label='Computed Positions',
        zorder=1
    )
    plot_ecef_points(
        ax,
        [true_position],
        color='red',
        size=25,
        label='True Position',
        zorder=3
    )

    # Save ECEF plot
    return save_ecef_plot(fig, filename)