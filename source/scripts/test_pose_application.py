from source.algorithm.FengDynamicRelocalizer import FengDynamicRelocalizer
from source.algorithm.CameraRig import CameraRig
from ..camera.CameraBlender import CameraBlender
from ..camera_pose_estimation.FivePointEstimator import FivePointEstimator
from ..plotting.plot_convergence import plot_t_convergence
from ..utilities import convert_angles_to_matrix, make_rotation_matrix
import numpy as np
import cv2

initial_camera_location = np.array((9, -8, 9), dtype=np.float64)
target_camera_location = np.array((10, -7, 8), dtype=np.float64)
initial_camera_subject = np.array((0, 0, 0), dtype=np.float64)
target_camera_subject = np.array((0, 0, 0), dtype=np.float64)

# Create the camera.
camera = CameraBlender((1200, 1600, 3), "data/blender-scenes/spring.blend")
R_target = make_rotation_matrix(target_camera_location, target_camera_subject)
R_initial = make_rotation_matrix(initial_camera_location, initial_camera_subject)

R_initial = np.stack(
    [
        np.array([0.56788277, 0.44651194, -0.69147397]),
        np.array([0.82307106, -0.31615759, 0.47180335]),
        np.array([-0.00794891, -0.83706121, -0.5470515]),
    ]
)
initial_camera_location = np.array([9.53397978, -7.377932, 8.42737709])

# Render the images.
im_target = camera.render_with_pose(R_target, target_camera_location)
im_initial = camera.render_with_pose(R_initial, initial_camera_location)
cv2.imwrite("tmp_TPA_target_pose.png", im_target)
cv2.imwrite("tmp_TPA_initial_pose.png", im_initial)

# Estimate pose.
fpe = FivePointEstimator()
R_delta, t_delta = fpe.estimate_pose(im_target, im_initial, camera.get_K())

# Apply the pose estimate.
scale = np.linalg.norm(target_camera_location - initial_camera_location)
delta_t = R_initial @ t_delta * scale
estimated_location = initial_camera_location + delta_t
im_estimate = camera.render_with_pose(
    R_initial @ R_delta,
    estimated_location,
)
cv2.imwrite(f"tmp_TPA_estimate.png", im_estimate)

print(np.linalg.norm(estimated_location - target_camera_location))