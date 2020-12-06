import numpy as np
import cv2
import os
from scipy.spatial.transform import Rotation, Slerp
from ..camera.CameraBlender import CameraBlender
from ..plotting.plot_convergence import plot_t_convergence, plot_r_convergence
from ..plotting.plot_feature_distance import plot_feature_distance
from .SettingsLoader import SettingsLoader

# This uses the same settings file as camera_recreate_pose.py. The remaining
# settings are animation-specific. Make sure you've actually run
# camera_recreate_pose to populate the results folder first.
SETTINGS_FILE = "data/blender-scenes/forest.json"
save_images = True
render_plots = True
render_feature_distance = True
frames_per_pose = 15
smoothing = 2.2

# Settings go here.
settings = SettingsLoader.load_settings(SETTINGS_FILE)
camera = CameraBlender(
    tuple(settings["image_shape"].astype(np.int64).tolist()), settings["scene"]
)


def with_folder(file_name: str):
    return os.path.join(settings["save_folder"], file_name)


# Load the intermediate poses.
poses = np.load(with_folder("tmp_intermediate_poses.npz"))
R_log = poses["R_log"]
t_log = poses["t_log"]
R_target = poses["R_target"]
t_target = poses["t_target"]
assert R_log.shape[0] == t_log.shape[0]
num_poses = R_log.shape[0]
frame_index = 0
reference = camera.render_with_pose(R_target, t_target)

# Define linear interpolation functions.
def interpolate_t(t_left, t_right, t):
    return t_left + (t_right - t_left) * t


def interpolate_R(R_left, R_right, t):
    slerp = Slerp([0, 1], Rotation.from_matrix([R_left, R_right]))
    return slerp(t).as_matrix()


def smooth(t):
    factor = np.power(0.5, 1 - smoothing)
    if t < 0.5:
        return factor * np.power(t, smoothing)
    else:
        return 1 - factor * np.power(1 - t, smoothing)


# Define a camera rendering helper.
def render(R, t, time):
    global frame_index
    global reference

    # Render the image.
    image = camera.render_with_pose(R, t)
    if save_images:
        cv2.imwrite(with_folder(f"tmp_animation_frame_{frame_index}.png"), image)

    # Render the plots.
    if render_plots:
        plot_t_convergence(
            t_target,
            t_log,
            None,
            time,
            with_folder(f"tmp_t_plot_{frame_index}.png"),
        )
        plot_r_convergence(
            R_target,
            R_log,
            time,
            with_folder(f"tmp_R_plot_{frame_index}.png"),
        )

    # Render the feature distance plot.
    if render_feature_distance:
        plot_feature_distance(
            reference, image, with_folder(f"tmp_feature_distance_{frame_index}.png")
        )

    frame_index += 1


# Render in-between frames.
for pose_index in range(num_poses):
    # Render the pose itself.
    render(R_log[pose_index], t_log[pose_index], pose_index)

    # Render intermediate poses.
    if pose_index != num_poses - 1:
        # Get the current and next poses.
        R_left = R_log[pose_index]
        t_left = t_log[pose_index]
        R_right = R_log[pose_index + 1]
        t_right = t_log[pose_index + 1]

        # Render interpolated frames.
        for i in range(1, frames_per_pose):
            progress = smooth(i / frames_per_pose)
            R = interpolate_R(R_left, R_right, progress)
            t = interpolate_t(t_left, t_right, progress)
            render(R, t, pose_index + progress)

print("Done.")
