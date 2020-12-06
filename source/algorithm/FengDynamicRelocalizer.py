from abc import ABC, abstractmethod
from .DynamicRelocalizer import DynamicRelocalizer
from ..camera_pose_estimation.CameraPoseEstimator import CameraPoseEstimator
from scipy.spatial.transform import Rotation
from .CameraRig import CameraRig
from ..logging.PoseLogger import PoseLogger
from typing import Tuple, Optional
import numpy as np
import cv2
from ..utilities import convert_angles_to_matrix
from typing import List


class FengDynamicRelocalizer(DynamicRelocalizer):
    # This is the initial guess for the translation scale.
    # If it's too small, the algorithm will be slow to converge. If it's too
    # big, the large translation could prevent the algorithm from working as
    # intended.
    s_initial: float

    # If s goes below s_min, the algorithm gets cut off.
    s_min: float

    def __init__(
        self,
        camera_rig: CameraRig,
        camera_pose_estimator: CameraPoseEstimator,
        s_initial: float,
        s_min: float,
        pose_logger: Optional[PoseLogger],
    ) -> None:
        DynamicRelocalizer.__init__(
            self, camera_rig, camera_pose_estimator, pose_logger
        )
        assert isinstance(s_initial, float)
        assert isinstance(s_min, float)
        self.s_initial = s_initial
        self.s_min = s_min

    def _recreate_image(self, reference_image: np.ndarray) -> (List[float], np.ndarray):
        """This is Feng's algorithm, as described in Algorithm 1. Assume that
        homography-based coarse camera relocalization has already been done.
        """
        s = self.s_initial
        s_log = [s]
        t_previous = np.zeros((3,), dtype=np.float64)

        i = 0

        while s > self.s_min:
            current_image = self.camera_rig.capture_image()

            if self.pose_logger is not None:
                self.pose_logger.log_position(self.camera_rig, current_image)

            R, t = self.camera_pose_estimator.estimate_pose(
                reference_image, current_image, self.camera_rig.camera.get_K()
            )
            if np.dot(t, t_previous) < 0:
                s /= 2
            self.camera_rig.apply_R_and_t(R, s * t)
            t_previous = t

            i += 1
            s_log = s_log + [s]
            print("Current s value: " + str(s))

            if i == 30:
                break

        if self.pose_logger is not None:
            self.pose_logger.log_position(self.camera_rig, current_image)

        return s_log, self.camera_rig.capture_image()
