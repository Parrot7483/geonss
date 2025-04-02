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


def iterative_weighted_least_squares(
        initial_state: np.ndarray,
        build_model_fn: Callable[..., Tuple[np.ndarray, np.ndarray, np.ndarray]],
        check_convergence_fn: Callable[[np.ndarray], bool],
        max_iterations: int = 10,
        damping_factor: np.float64 = np.float64(0.0),
        **kwargs
) -> np.ndarray:
    """
    General sequential least squares solver.

    Args:
        initial_state: Initial parameter_vector vector
        build_model_fn: Function to build geometry matrix, residuals, and weights
        check_convergence_fn: Function to check if solution has converged
        max_iterations: Maximum number of iterations
        damping_factor: Tikhonov regularization parameter (default: 0.0)
        **kwargs: Additional keyword arguments passed to build_model_fn

    Returns:
        Final parameter_vector vector
    """
    # Initialize parameter_vector
    state = initial_state.copy()

    # Iterative least squares solution
    for iteration in range(max_iterations):
        # Build model for current parameter_vector
        geometry_matrix, residuals, weights = build_model_fn(state, **kwargs)

        # Use the general weighted least squares solver
        state_update = weighted_least_squares(
            geometry_matrix=geometry_matrix,
            residuals=residuals,
            weights=weights,
            damping_factor=damping_factor
        )

        # Check for convergence
        if check_convergence_fn(state_update):
            logger.debug(f"Converged after {iteration + 1} iterations")
            break

        # Update parameter_vector
        state += state_update
    else:
        logger.debug(f"Reached maximum number of iterations ({max_iterations})")

    return state