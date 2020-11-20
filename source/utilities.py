import numpy as np


def is_rotation_matrix(R: np.ndarray) -> bool:
    # Check for the correct type and shape.
    if not (isinstance(R, np.ndarray) and R.shape == (3, 3)):
        return False

    # Check that it's a rotation matrix.
    return np.allclose(np.eye(3), R @ R.T, atol=1e-2) and np.allclose(
        np.linalg.det(R), 1, atol=1e-4
    )


def is_translation_vector(t: np.ndarray) -> bool:
    return isinstance(t, np.ndarray) and t.shape == (3,)


def is_image(image: np.ndarray) -> bool:
    return (
        isinstance(image, np.ndarray)
        and image.dtype == np.uint8
        and image.shape[2] == 3
    )
