from abc import ABC, abstractmethod
from typing import Tuple
from ..utilities import is_rotation_matrix, is_translation_vector
import numpy as np


class CameraPoseEstimator(ABC):
    """Abstract class for a camera pose estimator. Given two images, the camera
    pose estimator should return the R and t that would have to be applied to
    the first image to get the second image's pose. Note that because of metric
    ambiguity, t is only a direction.
    """

    def estimate_pose(
        self, image1: np.ndarray, image2: np.ndarray
    ) -> Tuple[np.ndarray]:
        """Wrapper for the estimator-specific function."""
        assert isinstance(image1, np.ndarray)
        assert isinstance(image2, np.ndarray)
        R, t = self._estimate_pose(image1, image2)
        assert is_rotation_matrix(R)
        assert is_translation_vector(t)
        return R, t

    @abstractmethod
    def _estimate_pose(
        self, image1: np.ndarray, image2: np.ndarray
    ) -> Tuple[np.ndarray]:
        raise Exception("Not implemented.")
