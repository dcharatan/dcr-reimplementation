from ..camera.CameraBlender import CameraBlender
from ..utilities import make_rotation_matrix
import numpy as np
import cv2

# Create the camera.
camera = CameraBlender((1200, 1600, 3), "data/blender-scenes/tram.blend")

# Define the camera's position.
# In Panda3D, the camera looks along the positive Y axis, and camera up is negative Z. #nice
camera_location = np.array((10, -12, 2), dtype=np.float64)
camera_target = np.array((0, 0, 0), dtype=np.float64)

# Make a rotation matrix.
R = make_rotation_matrix(camera_location, camera_target)

# Render an image.
image = camera.render_with_pose(R, np.float64(camera_location))
cv2.imwrite("tmp_camera_image.png", image)
