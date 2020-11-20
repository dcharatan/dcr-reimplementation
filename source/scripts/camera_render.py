from ..camera.CameraPanda3D import CameraPanda3D
import numpy as np
import cv2

# Create the camera.
camera = CameraPanda3D((600, 800, 3), "models/environment")

# Define the camera's position.
# In Panda3D, the camera looks along the positive Y axis, and camera up is negative Z. #nice
camera_location = np.array((100, 200, -1000))
camera_target = np.zeros(3)

# Make a rotation matrix.
y_up = np.array((0, 1, 0))
camera_to_target = camera_target - camera_location
z = camera_to_target
z /= np.linalg.norm(z)
x = np.cross(y_up, z)
x /= np.linalg.norm(x)
y = np.cross(z, x)
y /= np.linalg.norm(y)
R = np.stack([x, y, z], axis=1)

# Render an image.
image = camera.render_with_pose(R, camera_location)
cv2.imwrite("tmp_camera_image.png", image)
