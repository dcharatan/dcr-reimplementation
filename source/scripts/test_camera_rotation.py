from ..camera.CameraBlender import CameraBlender
from ..utilities import (
    convert_angles_to_matrix,
    make_rotation_matrix,
)
import cv2

# Create the camera.
camera = CameraBlender((1200, 1600, 3), "data/blender-scenes/spring.blend")
camera_location = np.array((9, -9, 9), dtype=np.float64)
camera_target = np.array((0, 0, 0), dtype=np.float64)
R = make_rotation_matrix(camera_location, camera_target)

# Render the images.
image_original = camera.render_with_pose(R, camera_location)
cv2.imwrite("tmp_original.png", image_original)
rotation = convert_angles_to_matrix(10, 0, 0)
for i in range(36):
    image_test = camera.render_with_pose(R, camera_location)
    cv2.imwrite(f"tmp_rotation_{i * 10}_deg.png", image_test)
    R = R @ rotation
