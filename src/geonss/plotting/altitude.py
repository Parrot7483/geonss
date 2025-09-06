import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from geonss.coordinates import LLAPosition


def plot_altitude_differences(
        true_position: LLAPosition,
        computed_positions: List[LLAPosition],
        filename = 'altitude_differences.svg'
) -> str:
    """
    Plot altitude differences between computed positions and true position.

    Args:
        true_position: LLAPosition object of true position
        computed_positions: List of LLAPosition objects for computed positions
        filename: Output SVG file name

    Returns:
        str: Absolute path to the saved file
    """
    # Calculate altitude differences
    true_alt = float(true_position.altitude)
    alt_diffs = [float(pos.altitude) - true_alt for pos in computed_positions]

    # Create x-axis values (position index)
    indices = list(range(len(computed_positions)))

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot altitude differences
    ax.scatter(indices, alt_diffs, color='blue', s=15, alpha=0.7, label='Altitude Differences')

    # Add zero reference line
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, label='True Altitude')

    # Add mean line
    mean_diff = float(np.mean(alt_diffs))
    ax.axhline(y=mean_diff, color='green', linestyle='-', linewidth=1, label=f'Mean Diff: {mean_diff:.2f}m')

    # Configure plot
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlabel('Position Index')
    ax.set_ylabel('Altitude Difference (m)')
    ax.set_title('Altitude Differences from True Position')

    # Add statistics
    std_dev = np.std(alt_diffs)
    rms = np.sqrt(np.mean(np.square(alt_diffs)))
    text = f'Mean: {mean_diff:.2f}m\nStd Dev: {std_dev:.2f}m\nRMS: {rms:.2f}m'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)

    # Add legend
    ax.legend()

    # Save plot
    plt.tight_layout()
    plt.savefig(filename, format='svg', bbox_inches='tight')
    plt.close(fig)

    return os.path.abspath(filename)
