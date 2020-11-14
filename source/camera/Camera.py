from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class Camera(ABC):
    """Abstract class for a camera. A camera should produce """

    def __init__(self, image_shape: Tuple[int]) -> None:
        self.image_shape = image_shape

    def render_with_pose(self, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Wrapper for the camera-specific render function."""
        assert isinstance(R, np.ndarray) and R.shape == (3, 3)
        assert isinstance(t, np.ndarray) and t.shape == (3,)
        image = self._render_with_pose(R, t)
        assert isinstance(image, np.ndarray) and image.shape == self.image_shape
        return image

    @abstractmethod
    def _render_with_pose(self, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        raise Exception("Not implemented.")
