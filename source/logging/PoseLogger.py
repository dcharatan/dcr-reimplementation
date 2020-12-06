import numpy as np
from typing import List
from ..algorithm.CameraRig import CameraRig
from ..utilities import is_rotation_matrix, is_translation_vector


class PoseLogger:
    R_log: List[np.ndarray]
    t_log: List[np.ndarray]

    def __init__(self):
        self.R_log = []
        self.t_log = []

    def log_position(self, camera_rig: CameraRig):
        self.R_log.append(camera_rig.get_eye_R())
        self.t_log.append(camera_rig.get_eye_t())

    def save(self, file_name: str, R_target: np.ndarray, t_target: np.ndarray):
        assert is_rotation_matrix(R_target)
        assert is_translation_vector(t_target)
        np.savez(
            file_name,
            R_log=np.stack(self.R_log),
            t_log=np.stack(self.t_log),
            R_target=R_target,
            t_target=t_target,
        )
