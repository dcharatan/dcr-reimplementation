import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Note: Lower AFD is better
def compute_feature_distance(image_ref: np.ndarray, image_curr: np.ndarray) -> float:
    # Code adapted from OpenCV documentation example
    # https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(image_ref, None)
    kp2, des2 = sift.detectAndCompute(image_curr, None)
    # FLANN parameters for nearest neighbor search
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    pts1 = []
    pts2 = []

    # Ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.70 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    # The findHomography function implicitly uses RANSAC to correct matches based on
    # the backprojection error. Thus, use the mask returned from this function
    # to filter down the matches and make the method 'robust'
    _, mask = cv.findHomography(pts1, pts2, cv.RANSAC)
    mask = mask.astype("bool")
    mask = mask.squeeze(1)
    pts1 = pts1[mask]
    pts2 = pts2[mask]

    # Compute AFD based on the formula using these filtered match points
    running_sum = 0.0
    for i in range(len(pts1)):
        running_sum += np.linalg.norm(pts1[i] - pts2[i])

    afd = running_sum / len(pts1)

    return afd, pts1, pts2


def is_rotation_matrix(R: np.ndarray) -> bool:
    # Check for the correct type and shape.
    if not (isinstance(R, np.ndarray) and R.shape == (3, 3) and R.dtype == np.float64):
        return False

    # Check that it's a rotation matrix.
    return np.allclose(np.eye(3), R @ R.T, atol=1e-2) and np.allclose(
        np.linalg.det(R), 1, atol=1e-2
    )


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


def make_rotation_matrix(location, target):
    """Make a rotation matrix for a camera at location that's pointing towards
    target. This assumes that up is (0, 0, -1) like it is in OpenCV.
    """
    y_up = np.array([0, 0, -1])
    camera_to_target = target - location
    z = camera_to_target
    z /= np.linalg.norm(z)
    x = np.cross(y_up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    return np.stack([x, y, z], axis=1)