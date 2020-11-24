from ..camera.CameraBlender import CameraBlender
import numpy as np
import cv2

# Create the camera.
camera = CameraBlender((1200, 1600, 3), "data/blender-scenes/spring.blend")

# Define the camera's position.
# In Panda3D, the camera looks along the positive Y axis, and camera up is negative Z. #nice
camera_location = np.array((9, -8, 9), dtype=np.float64)
camera_target = np.array((5, 5, 0), dtype=np.float64)

# Make a rotation matrix.
y_up = np.array((0, 0, 1))
target_to_camera = camera_location - camera_target
z = target_to_camera
z /= np.linalg.norm(z)
x = np.cross(y_up, z)
x /= np.linalg.norm(x)
y = np.cross(z, x)
y /= np.linalg.norm(y)
R = np.stack([x, y, z], axis=1)

# Render an image.
image = camera.render_with_pose(R, np.float64(camera_location))
cv2.imwrite("tmp_camera_image.png", image)
