"""
Numerical algorithms for GNSS processing.

This module provides general mathematical algorithms used across the GNSS processing pipeline
"""

import numpy as np
from typing import Optional, Callable, Tuple
import logging


logger = logging.getLogger(__name__)

def weighted_least_squares(
        geometry_matrix: np.ndarray,
        residuals: np.ndarray,
        weights: Optional[np.ndarray] = None,
        damping_factor: np.float64 = np.float64(0.0)
) -> np.ndarray:
    """
    Solve a general weighted least squares problem.

    Solves the minimization problem:
        min || W^(1/2) (Gx - r) ||^2 + damping_factor * ||x||^2

    Args:
        geometry_matrix: Design matrix G relating parameters to measurements
        residuals: Residual vector r (observed minus computed)
        weights: Optional weights for measurements. If None, equal weights are used
        damping_factor: Tikhonov regularization parameter (default: 0.0)

    Returns:
        Solution vector x
    """
    # Handle default weights (equal weighting)
    if weights is None:
        weights = np.ones(len(residuals))

    # Create weight matrix (diagonal matrix with weights)
    weight_matrix = np.diag(weights)

    # Handle regularization
    if damping_factor > 0:
        # Add regularization term to normal equations
        n = geometry_matrix.shape[1]
        reg_matrix = damping_factor * np.eye(n)
        normal_matrix = geometry_matrix.T @ weight_matrix @ geometry_matrix + reg_matrix
    else:
        # Standard weighted least squares
        normal_matrix = geometry_matrix.T @ weight_matrix @ geometry_matrix

    # Solve normal equations
    solution = np.linalg.inv(normal_matrix) @ geometry_matrix.T @ weight_matrix @ residuals

    return solution


def iterative_least_squares(
        initial_state: np.ndarray,
        model_fn: Callable[..., Tuple[np.ndarray, np.ndarray, np.ndarray]],
        iterations: int = 10,
        **kwargs
) -> np.ndarray:
    """
    General Minimal Iterative Least Squares (ILS) solver.

    Args:
        initial_state: Initial parameter vector
        model_fn: Function to build geometry matrix and residuals
        iterations: Number of iterations
        **kwargs: Additional keyword arguments passed to model_fn

    Returns:
        Final parameter vector
    """
    # Initialize state
    state = initial_state.copy()

    # Iterative least squares solution
    for iteration in range(iterations):
        # Build model for current state
        geometry_matrix, residuals, _ = model_fn(state, **kwargs)

        # Use the weighted least squares solver without weights
        state_update = weighted_least_squares(
            geometry_matrix=geometry_matrix,
            residuals=residuals,
            weights=None,
        )

        # Update state
        state += state_update

    return state


def iterative_reweighted_least_squares(
        initial_state: np.ndarray,
        model_fn: Callable[..., Tuple[np.ndarray, np.ndarray, np.ndarray]],
        loss_fn: Callable,
        convergence_fn: Callable[[np.ndarray], bool],
        max_iterations: int = 10,
        min_measurements: int = 4,
        damping_factor: np.float64 = np.float64(0.0),
        **kwargs
) -> np.ndarray:
    """
    General Iterative Reweighted Least Squares (IRLS) solver with robust loss functions.

    Args:
        initial_state: Initial parameter vector
        model_fn: Function to build geometry matrix, residuals, and weights
        loss_fn: Loss function class to use for robust estimation
        convergence_fn: Function to check if solution has converged
        max_iterations: Maximum number of iterations
        min_measurements: Minimum number of measurements required for robust solution
        damping_factor: Tikhonov regularization parameter (default: 0.0)
        **kwargs: Additional keyword arguments passed to model_fn

    Returns:
        Final parameter vector
    """
    # Initialize state
    state = initial_state.copy()
    previous_residuals = None

    # Iterative solution
    for iteration in range(max_iterations):
        # Build model for current state
        geometry_matrix, residuals, weights = model_fn(state, **kwargs)

        # Apply robust weighting if we have previous residuals
        if previous_residuals is not None:
            # Calculate robust weights based on previous residuals
            robust_weights = loss_fn(previous_residuals)
            weights *= robust_weights

        # Check if we have enough measurements with sufficient weight
        effective_measurements = np.sum(weights > 1e-10)
        if effective_measurements < min_measurements or len(residuals) < min_measurements:
            raise ValueError(f"Robust weighting reduced effective measurements below minimum "
                           f"({effective_measurements}/{min_measurements})")

        # Use weighted least squares solver
        state_update = weighted_least_squares(
            geometry_matrix=geometry_matrix,
            residuals=residuals,
            weights=weights,
            damping_factor=damping_factor
        )

        # Update state
        state += state_update

        # Store residuals for next iteration
        previous_residuals = residuals

        # Check for convergence
        if convergence_fn(state_update):
            logger.debug(f"Converged after {iteration + 1} iterations")
            break
    else:
        logger.debug(f"Reached maximum number of iterations ({max_iterations})")

    return state


def huber_weight(residuals: np.ndarray, k: float = 1.345) -> np.ndarray:
    """
    Calculate weights from Huber's M-estimator.

    Linear for large residuals, quadratic for small residuals.
    Default parameter k=1.345 gives 95% efficiency for normal distributions.

    Args:
        residuals: Residual vector
        k: Tuning constant (default: 1.345)

    Returns:
        Weight vector
    """
    # Normalize residuals by their median absolute deviation
    mad = np.median(np.abs(residuals - np.median(residuals)))
    if mad < 1e-10:  # Avoid division by zero
        return np.ones_like(residuals)

    normalized_residuals = np.abs(residuals) / (1.4826 * mad)

    # Calculate Huber weights
    weights = np.ones_like(normalized_residuals)
    large_residuals = normalized_residuals > k
    weights[large_residuals] = k / normalized_residuals[large_residuals]

    return weights


def tukey_weight(residuals: np.ndarray, c: float = 4.685) -> np.ndarray:
    """
    Calculate weights using Tukey's bi-weight function.

    Completely rejects outliers beyond the threshold.
    Default parameter c=4.685 gives 95% efficiency for normal distributions.

    Args:
        residuals: Residual vector
        c: Tuning constant (default: 4.685)

    Returns:
        Weight vector
    """
    # Normalize residuals by their median absolute deviation
    mad = np.median(np.abs(residuals - np.median(residuals)))
    if mad < 1e-10:  # Avoid division by zero
        return np.ones_like(residuals)

    normalized_residuals = np.abs(residuals) / (1.4826 * mad)

    # Calculate Tukey weights
    weights = np.ones_like(normalized_residuals)
    large_residuals = normalized_residuals > c

    u = normalized_residuals[~large_residuals] / c
    weights[~large_residuals] = np.square(1 - np.square(u))
    weights[large_residuals] = 0.0

    return weights