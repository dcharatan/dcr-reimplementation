from source.algorithm.FengDynamicRelocalizer import FengDynamicRelocalizer
from source.algorithm.CameraRig import CameraRig
from ..camera.CameraPanda3D import CameraPanda3D
from ..camera_pose_estimation.FivePointEstimator import FivePointEstimator
import numpy as np
import cv2

# Create the camera.
camera = CameraPanda3D((600, 800, 3), "models/environment")

# Define the camera's position.
# In Panda3D, the camera looks along the positive Y axis, and camera up is negative Z. #nice
camera_location_a = np.array((100, 200, -1000), dtype=np.float64)
camera_location_b = np.array((100, 200, -1000), dtype=np.float64)
camera_target_a = np.array((0, 0, 0), dtype=np.float64)
camera_target_b = np.array((0, 0, 0), dtype=np.float64)

# Make a rotation matrix.
def make_rotation_matrix(location, target, y_up):
    camera_to_target = target - location
    z = camera_to_target
    z /= np.linalg.norm(z)
    x = np.cross(y_up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    return np.stack([x, y, z], axis=1)


tilted = np.array((0, 1, 1), dtype=np.float64)
tilted = tilted / np.linalg.norm(tilted)

R_a = make_rotation_matrix(camera_location_a, camera_target_a, np.array((0, 1, 0)))
R_b = make_rotation_matrix(camera_location_b, camera_target_b, np.array((0, -1, 0)))


# Render the images.
image_a = camera.render_with_pose(R_a, camera_location_a)
image_b = camera.render_with_pose(R_b, camera_location_b)
cv2.imwrite("tmp_camera_image_a.png", image_a)
cv2.imwrite("tmp_camera_image_b.png", image_b)

# Run Feng's algorithm.
fpe = FivePointEstimator()
rig = CameraRig(camera, np.eye(3), np.zeros((3,)))
algo = FengDynamicRelocalizer(rig, fpe, 0.0, -1.0)
recreation = algo.recreate_image(image_a, R_b, camera_location_b)
cv2.imwrite("tmp_camera_image_a_recreation.png", recreation)