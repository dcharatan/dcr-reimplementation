from source.algorithm.FengDynamicRelocalizer import FengDynamicRelocalizer
from source.algorithm.CameraRig import CameraRig
from ..camera.CameraBlender import CameraBlender
from ..camera_pose_estimation.FivePointEstimator import FivePointEstimator
from ..plotting.plot_convergence import plot_t_convergence, plot_r_convergence
from ..utilities import make_rotation_matrix, convert_angles_to_matrix
from ..logging.PoseLogger import PoseLogger
from .SettingsLoader import SettingsLoader
import os
import numpy as np
import cv2
import git
import json
from pathlib import Path
from ..utilities import parse_args_for_settings_file

SETTINGS_FILE = parse_args_for_settings_file()

# Load the settings and create the result directory.
settings = SettingsLoader.load_settings(SETTINGS_FILE)
Path(settings["save_folder"]).mkdir(parents=True, exist_ok=True)


def with_folder(file_name: str):
    return os.path.join(settings["save_folder"], file_name)


# To make a set of results more reproducible, copy the settings to the results
# folder along with the current commit hash.
with open(SETTINGS_FILE, "r") as f:
    settings_json = json.load(f)
settings_json["commit_hexsha"] = git.Repo(
    search_parent_directories=True
).head.object.hexsha
with open(with_folder("tmp_settings.json"), "w") as f:
    json.dump(settings_json, f)


# Create the camera.
camera = CameraBlender(
    tuple(settings["image_shape"].astype(np.int64).tolist()), settings["scene"]
)

# Calculate R and t for the initial and reference poses.
reference_location = settings["reference_location"]
initial_location = settings["initial_location"]
R_reference = make_rotation_matrix(reference_location, settings["reference_target"])
R_initial = make_rotation_matrix(initial_location, settings["initial_target"])

# Render the reference and initial images.
image_a = camera.render_with_pose(R_reference, reference_location)
image_b = camera.render_with_pose(R_initial, initial_location)
cv2.imwrite(with_folder("tmp_target_pose.png"), image_a)
cv2.imwrite(with_folder("tmp_initial_pose.png"), image_b)

# Run Feng's algorithm.
fpe = FivePointEstimator()
rig = CameraRig(
    camera,
    convert_angles_to_matrix(*settings["hand_eye_euler_xyz"].tolist()),
    settings["hand_eye_translation"],
)
rig.set_up_oracle(reference_location)
pose_logger = PoseLogger(settings["save_folder"])
algo = FengDynamicRelocalizer(
    rig, fpe, settings["s_initial"], settings["s_min"], pose_logger
)

s_log, recreation = algo.recreate_image(image_a, R_initial, initial_location)
cv2.imwrite(with_folder("tmp_recreated_pose.png"), recreation)
pose_logger.save(
    with_folder("tmp_intermediate_poses.npz"), R_reference, reference_location
)
plot_t_convergence(
    reference_location,
    rig.translation_log,
    s_log,
    None,
    with_folder("tmp_t_convergence.png"),
)
plot_r_convergence(
    R_reference, rig.rotation_log, None, with_folder("tmp_R_convergence.png")
)
print("Done!")
