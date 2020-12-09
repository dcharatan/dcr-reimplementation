from source.algorithm.FengDynamicRelocalizer import FengDynamicRelocalizer
from source.algorithm.CameraRig import CameraRig
from ..camera.CameraBlender import CameraBlender
from ..camera_pose_estimation.FivePointEstimator import FivePointEstimator
from ..plotting.plot_convergence import plot_t_convergence
from ..utilities import convert_angles_to_matrix, make_rotation_matrix
import numpy as np
import cv2


def triangulateAndCountInliers(pts1, pts2, R, T, inv_K):
    positive_inlier_count = 0
    num_pts = pts1.shape[0]
    cat_ones = np.ones((num_pts, 1))
    pts1_cated = np.concatenate((pts1, cat_ones), 1)
    pts2_cated = np.concatenate((pts2, cat_ones), 1)
    gamma1 = np.matmul(inv_K, pts1_cated.reshape(3, -1))
    gamma2 = np.matmul(inv_K, pts2_cated.reshape(3, -1))

    for i in range(num_pts):
        R_gamma1_expanded = np.expand_dims(np.matmul(-R, gamma1[:, i]), 1)
        gamma2_expanded = np.expand_dims(gamma2[:, i], 1)
        R_gamma1_gamma2_concat = np.concatenate((R_gamma1_expanded, gamma2_expanded), 1)
        R_gamma1_gamma2_pinv = np.linalg.pinv(R_gamma1_gamma2_concat)
        rho1_rho2 = np.matmul(R_gamma1_gamma2_pinv, T)
        if rho1_rho2[0] > 0 and rho1_rho2[1] > 0:
            positive_inlier_count += 1

    return positive_inlier_count


initial_camera_location = np.array((9, -8, 9), dtype=np.float64)
target_camera_location = np.array((10, -7, 8), dtype=np.float64)
initial_camera_subject = np.array((0, 0, 0), dtype=np.float64)
target_camera_subject = np.array((0, 0, 0), dtype=np.float64)

# Create the camera.
camera = CameraBlender((1200, 1600, 3), "data/blender-scenes/spring.blend")
R_target = make_rotation_matrix(target_camera_location, target_camera_subject)
R_initial = make_rotation_matrix(initial_camera_location, initial_camera_subject)

# R_initial = np.stack(
#     [
#         np.array([0.56788277, 0.44651194, -0.69147397]),
#         np.array([0.82307106, -0.31615759, 0.47180335]),
#         np.array([-0.00794891, -0.83706121, -0.5470515]),
#     ]
# )
# initial_camera_location = np.array([9.53397978, -7.377932, 8.42737709])

# Render the images.
im_target = camera.render_with_pose(R_target, target_camera_location)
im_initial = camera.render_with_pose(R_initial, initial_camera_location)
cv2.imwrite("tmp_TPA_target_pose.png", im_target)
cv2.imwrite("tmp_TPA_initial_pose.png", im_initial)

# Estimate pose.
fpe = FivePointEstimator()
R_delta, t_delta = fpe.estimate_pose(im_target, im_initial, camera.get_K())

# t_delta = -t_delta

# Apply the pose estimate.
scale = np.linalg.norm(target_camera_location - initial_camera_location)
delta_t = R_initial @ t_delta * scale
estimated_location = initial_camera_location + delta_t
im_estimate = camera.render_with_pose(
    R_initial @ R_delta,
    estimated_location,
)
cv2.imwrite(f"tmp_TPA_estimate.png", im_estimate)

print(
    "Distance between estimated and true locations: "
    + str(np.linalg.norm(estimated_location - target_camera_location))
)

# IMPORTANT: GT R is just the R from here, GT T is just the initial - final T
# Use this to plot Epipolar Geometry based on a new F Matrix
