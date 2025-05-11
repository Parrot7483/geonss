import numpy as np

def apply_phase_center_offset(ecef_position: np.ndarray, neu_offset: np.ndarray) -> np.ndarray:
    """
    Apply phase center offset correction to satellite ECEF positions using vectorized operations.
    """

    ecef_position = np.asarray(ecef_position)
    neu_offset = np.asarray(neu_offset)

    if ecef_position.ndim != 2 or ecef_position.shape[1] != 3:
        raise ValueError(
            f"Internal Error: ecef_positions must be an (N, 3) array, got shape {ecef_position.shape}")
    if neu_offset.ndim != 2 or neu_offset.shape[1] != 3:
        raise ValueError(f"Internal Error: neu_offsets must be an (N, 3) array, got shape {neu_offset.shape}")
    if ecef_position.shape[0] != neu_offset.shape[0]:
        raise ValueError(
            f"Internal Error: Number of ECEF positions ({ecef_position.shape[0]}) "
            f"must match number of NEU offsets ({neu_offset.shape[0]})")

    num_positions = ecef_position.shape[0]
    if num_positions == 0:
        return np.empty((0, 3), dtype=ecef_position.dtype)

    # Position magnitude (N,)
    position_magnitudes = np.linalg.norm(ecef_position, axis=1)

    # Avoid division by zero if any position is at the origin
    position_magnitudes_safe = np.maximum(position_magnitudes, 1e-12)

    # unit_vector_up: Unit vector pointing from Earth center to satellite (N, 3)
    unit_vector_up = ecef_position / position_magnitudes_safe[:, np.newaxis]

    # unit_vector_east: Perpendicular to Z_sat and Earth's rotation axis (approx East) (N, 3)
    earth_rotation_axis = np.array([0.0, 0.0, 1.0])
    unit_vector_east = np.cross(earth_rotation_axis, unit_vector_up, axisa=0, axisb=1)

    # Norm of unit_vector_east (N,)
    east_vector_norm = np.linalg.norm(unit_vector_east, axis=1)

    # Handle polar case (where cross product is near zero)
    is_polar_region = east_vector_norm < 1e-10
    east_vector_norm_safe = np.maximum(east_vector_norm, 1e-12)

    # Normalize unit_vector_east (N, 3)
    unit_vector_east = unit_vector_east / east_vector_norm_safe[:, np.newaxis]

    # For polar cases, set unit_vector_east to a default (e.g., along ECEF X-axis)
    default_east_polar = np.array([1.0, 0.0, 0.0])
    unit_vector_east[is_polar_region] = default_east_polar

    # unit_vector_north: Completes right-handed system (approx North) (N, 3)
    unit_vector_north = np.cross(unit_vector_up, unit_vector_east, axisa=1, axisb=1)

    # Stack the basis vectors into N rotation matrices (N, 3, 3)
    # Rotation transforms from Satellite Body Frame [E, N, U] to ECEF Frame [X, Y, Z]
    rotation_matrices = np.stack([unit_vector_east, unit_vector_north, unit_vector_up], axis=-1)

    # Reorder NEU offset [N, E, U] to Satellite Body Frame [E, N, U]
    # neu_offsets columns: 0=N, 1=E, 2=U
    # satellite_body_offsets columns: 0=E (east), 1=N (north), 2=U (up)
    satellite_body_offsets = neu_offset[:, [1, 0, 2]]  # Shape (N, 3)

    # Apply N rotation matrices to N body_offset vectors using einsum for batch matrix-vector multiplication
    # 'nij,nj->ni': For each i, multiply matrix R[i] (shape 3x3) by vector body_offset[i] (shape 3) -> result[i] (shape 3)
    ecef_offsets = np.einsum('nij,nj->ni', rotation_matrices, satellite_body_offsets)  # Shape (N, 3)

    # Subtract ECEF offset from original ECEF position (PCO points from Antenna Phase Center to Center of Mass)
    corrected_ecef_positions = ecef_position - ecef_offsets

    return corrected_ecef_positions
