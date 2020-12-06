from source.algorithm.FengDynamicRelocalizer import FengDynamicRelocalizer
from source.algorithm.CameraRig import CameraRig
from ..camera.CameraBlender import CameraBlender
from ..camera_pose_estimation.FivePointEstimator import FivePointEstimator
from ..utilities import is_rotation_matrix, convert_angles_to_matrix
from scipy.spatial.transform import Rotation
import numpy as np
import cv2

camera = CameraBlender((1200, 1600, 3), "data/blender-scenes/camera_test.blend")
camera_location = np.array((0, 0, -30), dtype=np.float64)
R = np.eye(3, dtype=np.float64)

# Render the images.
image_original = camera.render_with_pose(R, camera_location)
cv2.imwrite("tmp_original.png", image_original)

tests = [
    (10, 0, 0, "x"),
    (0, 10, 0, "y"),
    (0, 0, 10, "z"),
    (10, 10, 0, "xy"),
    (0, 10, 10, "yz"),
    (10, 0, 10, "zx"),
]

# Once these have been rendered, visually inspect them to confirm that they
# match OpenCV's coordinate system. Rotations around x should move the cube
# down, rotations around y should move the cube to the left, and rotations
# around z should make the left corners dip down slightly.
fpe = FivePointEstimator()
for x, y, z, name in tests:
    R_test = convert_angles_to_matrix(x, y, z)
    image_test = camera.render_with_pose(R_test, camera_location)
    cv2.imwrite(f"tmp_{name}.png", image_test)
