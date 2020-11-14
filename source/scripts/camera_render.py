from ..camera.CameraPanda3D import CameraPanda3D
import numpy as np
import cv2

# Create the camera.
camera = CameraPanda3D((600, 800, 3), "models/environment")

# Define the camera's position.
# In Panda3D, the camera looks along the positive Y axis, and camera up is negative Z. #nice
camera_location = np.array((100, -1000, 200))
camera_target = np.zeros(3)

# Make a rotation matrix.
z_up = np.array((0, 0, -1))
camera_to_target = camera_target - camera_location
camera_to_target /= np.linalg.norm(camera_to_target)
y = camera_to_target
x = np.cross(y, z_up)
z = np.cross(x, y)
R = np.stack([x, y, z], axis=1)

# Render an image.
image = camera.render_with_pose(R, camera_location)
cv2.imwrite("tmp_camera_image.png", image)
