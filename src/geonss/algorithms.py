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
    Solve a general weighted least squares problem using np.linalg.lstsq

    Solves the minimization problem:
        min || W^(1/2) (Gx - r) ||^2 + damping_factor * ||x||^2

    Args:
        geometry_matrix: Design matrix G (M x N) relating parameters to measurements.
        residuals: Residual vector r (M x 1) (observed minus computed).
        weights: Optional 1D array of weights (M,) for measurements.
                 If None, equal weights (1.0) are used.
        damping_factor: Tikhonov regularization parameter (lambda >= 0).

    Returns:
        Solution vector x (N x 1).

    Raises:
        ValueError: If input dimensions are inconsistent.
    """
    M, N = geometry_matrix.shape
    if len(residuals) != M:
        raise ValueError(f"Dimension mismatch: geometry_matrix rows ({M}) != residuals length ({len(residuals)})")

    if damping_factor < 0:
        raise ValueError("Damping factor must be non-negative.")

    # Handle weights
    if weights is None:
        sqrt_weights = np.ones(M)
    else:
        if len(weights) != M:
            raise ValueError(f"Dimension mismatch: geometry_matrix rows ({M}) != weights length ({len(weights)})")
        # Ensure weights are non-negative before taking sqrt
        weights = np.maximum(weights, 0.0)
        sqrt_weights = np.sqrt(weights)

    # Apply weights to geometry matrix and residuals
    # Use broadcasting via [:, np.newaxis] for element-wise multiplication
    G_prime = geometry_matrix * sqrt_weights[:, np.newaxis]
    r_prime = residuals * sqrt_weights

    # Handle regularization/damping by augmenting the system
    if damping_factor > 0:
        # Augment G' with sqrt(lambda)*I below it
        damping_matrix = np.sqrt(damping_factor) * np.eye(N)
        G_augmented = np.vstack((G_prime, damping_matrix))

        # Augment r' with zeros below it
        r_augmented = np.concatenate((r_prime, np.zeros(N)))

        # Solve the augmented system using least squares
        solution, _, _, _ = np.linalg.lstsq(G_augmented, r_augmented, rcond=None)

    else:
        # Solve the standard weighted system G'x = r' using least squares
        solution, _, _, _ = np.linalg.lstsq(G_prime, r_prime, rcond=None)

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
        loss_fn: Callable[[np.ndarray], np.ndarray],
        convergence_fn: Callable[[np.ndarray], bool],
        max_iterations: int = 10,
        damping_factor: np.float64 = np.float64(0.0),
        weight_epsilon: float = 1e-10,
        sigma_epsilon: float = 1e-10,
        **kwargs
) -> np.ndarray:
    """
    General Iterative Reweighted Least Squares (IRLS) solver with robust loss functions.

    Args:
        initial_state: Initial parameter vector [x, y, z, clk_bias, isb1, ...]
        model_fn: Function taking state and **kwargs, returning:
                  (geometry_matrix, raw_residuals, a_priori_weights).
        loss_fn: Robust weight function taking standardized residuals and
                 returning a multiplicative weight factor (0 to 1).
        convergence_fn: Function taking state_update vector and returning True
                        if converged.
        max_iterations: Maximum number of iterations.
        damping_factor: Tikhonov regularization parameter.
        weight_epsilon: Threshold below which weights are considered zero.
        sigma_epsilon: Threshold below which sigma (derived from weights) is invalid.
        **kwargs: Additional keyword arguments passed to model_fn.

    Returns:
        Final estimated parameter vector.

    Raises:
        ValueError: If processing fails due to insufficient measurements,
                    model function errors, or WLS solver errors.
    """
    state = initial_state.astype(np.float64, copy=True)
    previous_residuals: Optional[np.ndarray] = None
    previous_apriori_weights: Optional[np.ndarray] = None
    converged = False

    for iteration in range(max_iterations):
        logger.debug(f"IRLS Iteration {iteration + 1}, Current State: {state}")

        # 1. Build model for current state (H, r, W_apriori)
        geometry_matrix, current_residuals, current_apriori_weights = model_fn(state, **kwargs)

        # 2. Calculate final weights for this iteration's WLS
        weights = current_apriori_weights.copy() # Start with a priori weights

        if iteration > 0 and previous_residuals is not None and previous_apriori_weights is not None:
            # Check for dimension mismatch
            if len(previous_residuals) == len(current_apriori_weights):
                # Calculate standard deviations (sigma) from PREVIOUS a priori weights
                valid_prev_weights_mask = previous_apriori_weights > weight_epsilon
                sigma = np.full_like(previous_apriori_weights, np.inf)
                sigma[valid_prev_weights_mask] = 1.0 / np.sqrt(previous_apriori_weights[valid_prev_weights_mask])

                # Standardize PREVIOUS residuals using PREVIOUS sigma
                standardized_residuals = np.zeros_like(previous_residuals)
                valid_sigma_mask = sigma < (1.0 / sigma_epsilon) # Avoid infinite or near-zero sigma
                standardized_residuals[valid_sigma_mask] = previous_residuals[valid_sigma_mask] / sigma[valid_sigma_mask]

                # Calculate robust weight factor using loss_fn
                try:
                    robust_weights_factor = loss_fn(standardized_residuals)
                    robust_weights_factor[~np.isfinite(robust_weights_factor)] = 0.0
                    robust_weights_factor = np.maximum(robust_weights_factor, 0.0)
                except Exception as e:
                    logger.warning(f"Error in loss_fn during iteration {iteration + 1}: {e}. Using a priori weights.", exc_info=True)
                    robust_weights_factor = np.ones_like(standardized_residuals) # Fallback

                # Apply robust factor to CURRENT a priori weights
                weights *= robust_weights_factor
                logger.debug(f"Applied robust weighting. Min/Max factor: "
                             f"{np.min(robust_weights_factor):.3f}/{np.max(robust_weights_factor):.3f}")
            else:
                logger.warning(f"Measurement count mismatch iteration {iteration + 1}. Using only a priori weights.")
                # 'weights' already holds current_apriori_weights
        else:
            logger.debug("First iteration, using only a priori weights.")

        # 3. Solve Weighted Least Squares for the state *update*
        try:
            state_update = weighted_least_squares(
                geometry_matrix=geometry_matrix,
                residuals=current_residuals,
                weights=weights,
                damping_factor=damping_factor
            )
            # Basic check for finite result
            if not np.all(np.isfinite(state_update)):
                raise ValueError("WLS solver returned non-finite state update.")
        except np.linalg.LinAlgError as e:
            logger.error(f"WLS failed in iteration {iteration + 1}: {e}", exc_info=True)
            raise ValueError(f"Linear algebra error during WLS solution: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in WLS call: {e}", exc_info=True)
            raise ValueError(f"Unexpected error during WLS solution") from e

        # 4. Update state
        state += state_update
        logger.debug(f"State update norm: {np.linalg.norm(state_update):.4g}")

        # 5. Store results for next iteration
        previous_residuals = current_residuals.copy()
        previous_apriori_weights = current_apriori_weights.copy()

        # 6. Check for convergence
        try:
            if convergence_fn(state_update):
                logger.debug(f"Converged after {iteration + 1} iterations.")
                converged = True
                break
        except Exception as e:
            logger.warning(f"Convergence check failed in iteration {iteration + 1}: {e}", exc_info=True)

    # End of loop
    if not converged:
        logger.debug(f"Reached maximum iterations ({max_iterations}) without convergence.")

    logger.debug(f"Final IRLS State: {state}")
    return state


def huber_weight(standardized_residuals: np.ndarray, k: float = 1.345) -> np.ndarray:
    """
    Calculate Huber weight factors for residuals already standardized by their
    a priori standard deviation (sigma).

    Args:
        standardized_residuals: Residual vector (residual / sigma).
        k: Tuning constant (default: 1.345). Should be non-negative.

    Returns:
        Weight factor vector (values between 0 and 1).
    """
    if k < 0:
        raise ValueError("Huber threshold k must be non-negative.")
    abs_std_res = np.abs(standardized_residuals.astype(np.float64, copy=False))
    weights = np.ones_like(abs_std_res)
    large_res_mask = abs_std_res > k
    if k > 0:
        weights[large_res_mask] = k / abs_std_res[large_res_mask]
    else: # k == 0
        weights[abs_std_res > 1e-15] = 0.0
    weights[~np.isfinite(standardized_residuals)] = 0.0
    return weights


def tukey_weight(standardized_residuals: np.ndarray, c: float = 4.685) -> np.ndarray:
    """
    Calculate weights using Tukey's bi-weight function for residuals already
    standardized by their a priori standard deviation (sigma).

    Args:
        standardized_residuals: Residual vector (residual / sigma).
        c: Tuning constant (default: 4.685). Should be positive.

    Returns:
        Weight factor vector (values between 0 and 1).
    """
    if c <= 0:
        raise ValueError("Tukey threshold c must be positive.")
    abs_std_res = np.abs(standardized_residuals.astype(np.float64, copy=False))
    weights = np.zeros_like(abs_std_res)
    within_threshold_mask = abs_std_res <= c
    if np.any(within_threshold_mask):
        u = abs_std_res[within_threshold_mask] / c
        weights[within_threshold_mask] = np.square(1 - np.square(u))
    weights[~np.isfinite(standardized_residuals)] = 0.0
    return weights