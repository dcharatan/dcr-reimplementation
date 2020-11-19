import numpy as np


def is_rotation_matrix(R: np.ndarray) -> bool:
    # Check for the correct type and shape.
    if not (isinstance(R, np.ndarray) and R.shape == (3, 3)):
        return False

    # Check that it's a rotation matrix.
    return np.allclose(np.eye(3), R @ R.T) and np.allclose(np.linalg.det(R), 1)


def is_translation_vector(t: np.ndarray) -> bool:
    return isinstance(t, np.ndarray) and t.shape == (3,)