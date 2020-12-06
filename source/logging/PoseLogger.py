import numpy as np
from typing import List
from ..algorithm.CameraRig import CameraRig


class PoseLogger:
    R_log: List[np.ndarray]
    t_log: List[np.ndarray]

    def __init__(self):
        self.R_log = []
        self.t_log = []

    def log_position(self, camera_rig: CameraRig):
        self.R_log.append(camera_rig.get_eye_R())
        self.t_log.append(camera_rig.get_eye_t())

    def save(self, file_name: str):
        np.savez(file_name, np.stack(self.R_log), np.stack(self.t_log))