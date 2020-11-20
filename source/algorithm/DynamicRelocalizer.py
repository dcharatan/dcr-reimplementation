from abc import ABC, abstractmethod
from typing import Tuple
from .CameraRig import CameraRig
from ..camera_pose_estimation.CameraPoseEstimator import CameraPoseEstimator
from ..utilities import is_rotation_matrix, is_translation_vector, is_image
import numpy as np


class DynamicRelocalizer(ABC):
    """Abstract class for a dynamic relocalization algorithm. This mainly exists
    to separate the assertions/sanity checks from the actual Feng algorithm."""

    camera_rig: CameraRig
    camera_pose_estimator: CameraPoseEstimator

    def __init__(
        self, camera_rig: CameraRig, camera_pose_estimator: CameraPoseEstimator
    ) -> None:
        assert isinstance(camera_rig, CameraRig)
        assert isinstance(camera_pose_estimator, CameraPoseEstimator)
        self.camera_rig = camera_rig
        self.camera_pose_estimator = camera_pose_estimator

    def recreate_pose(
        self, reference_image: np.ndarray, initial_R: np.ndarray, initial_t: np.ndarray
    ) -> Tuple[np.ndarray]:
        assert is_image(reference_image)
        assert is_rotation_matrix(initial_R)
        assert is_translation_vector(initial_t)
        R, t = self._recreate_pose(reference_image, initial_R, initial_t)
        assert is_rotation_matrix(R)
        assert is_translation_vector(t)
        return R, t

    @abstractmethod
    def _recreate_pose(
        self, reference_image: np.ndarray, initial_R: np.ndarray, initial_t: np.ndarray
    ) -> Tuple[np.ndarray]:
        raise Exception("Not implemented.")
