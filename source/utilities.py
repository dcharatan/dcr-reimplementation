import numpy as np


def is_rotation_matrix(R: np.ndarray) -> bool:
    # Check for the correct type and shape.
    if not (isinstance(R, np.ndarray) and R.shape == (3, 3) and R.dtype == np.float64):
        return False

    # Check that it's a rotation matrix.
    return np.allclose(np.eye(3), R @ R.T) and np.allclose(np.linalg.det(R), 1)


def is_translation_vector(t: np.ndarray) -> bool:
    return isinstance(t, np.ndarray) and t.shape == (3,) and t.dtype == np.float64


def is_image(image: np.ndarray) -> bool:
    return (
        isinstance(image, np.ndarray)
        and image.dtype == np.uint8
        and image.shape[2] == 3
    )


def convert_angles_to_matrix(
    x_angle: float, y_angle: float, z_angle: float
) -> np.ndarray:
    # Takes angles in degrees of a rotation about x, y and z axes respectively
    # and returns a matrix!
    x_angle = np.deg2rad(x_angle)
    y_angle = np.deg2rad(y_angle)
    z_angle = np.deg2rad(z_angle)

    x_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(x_angle), -np.sin(x_angle)],
            [0.0, np.sin(x_angle), np.cos(x_angle)],
        ]
    )

    y_matrix = np.array(
        [
            [np.cos(y_angle), 0.0, np.sin(y_angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(y_angle), 0.0, np.cos(y_angle)],
        ]
    )

    z_matrix = np.array(
        [
            [np.cos(z_angle), -np.sin(z_angle), 0.0],
            [np.sin(z_angle), np.cos(z_angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    return np.matmul(z_matrix, np.matmul(y_matrix, x_matrix))
