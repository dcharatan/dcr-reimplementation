from abc import ABC, abstractmethod
from typing import Tuple
from ..utilities import is_rotation_matrix, is_translation_vector, is_image
import numpy as np


class Camera(ABC):
    """Abstract class for a camera. Given R and T, a camera should produce an
    image of the shape it was initialized with.
    """

    def __init__(self, image_shape: Tuple[int]) -> None:
        self.image_shape = image_shape

    def render_with_pose(self, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Wrapper for the camera-specific render function."""
        assert is_rotation_matrix(R)
        assert is_translation_vector(t)
        image = self._render_with_pose(R, t)
        assert is_image(image) and image.shape == self.image_shape
        return image

    @abstractmethod
    def _render_with_pose(self, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        raise Exception("Not implemented.")

    def get_K(self) -> np.ndarray:
        """Get the intrinsic camera calibration matrix K."""
        K = self._get_K()
        assert isinstance(K, np.ndarray) and K.shape == (3, 3)
        return K

    @abstractmethod
    def _get_K(self) -> np.ndarray:
        raise Exception("Not implemented.")
