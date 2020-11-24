from abc import ABC, abstractmethod
from .DynamicRelocalizer import DynamicRelocalizer
from ..camera_pose_estimation.CameraPoseEstimator import CameraPoseEstimator
from scipy.spatial.transform import Rotation
from .CameraRig import CameraRig
from typing import Tuple
import numpy as np
import cv2
from ..utilities import convert_angles_to_matrix


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
    ) -> None:
        DynamicRelocalizer.__init__(self, camera_rig, camera_pose_estimator)
        assert isinstance(s_initial, float)
        assert isinstance(s_min, float)
        self.s_initial = s_initial
        self.s_min = s_min

    def _recreate_image(self, reference_image: np.ndarray) -> np.ndarray:
        """This is Feng's algorithm, as described in Algorithm 1. Assume that
        homography-based coarse camera relocalization has already been done.
        """

        s = self.s_initial
        t_previous = np.zeros((3,), dtype=np.float64)

        i = 0

        while s > self.s_min:
            current_image = self.camera_rig.capture_image()
            cv2.imwrite(f"tmp_estimation_{i}.png", current_image)
            R, t = self.camera_pose_estimator.estimate_pose(
                current_image, reference_image, self.camera_rig.camera.get_K()
            )

            x, y, z = Rotation.from_matrix(R).as_euler("xyz")
            R = Rotation.from_euler("xyz", (-x, y, z)).as_matrix()

            self.camera_rig.apply_rotation(R)
            if np.dot(t, t_previous) < 0:
                s /= 2
            self.camera_rig.apply_translation(s * t)
            t_previous = t

            
            i += 1

        return self.camera_rig.capture_image()
