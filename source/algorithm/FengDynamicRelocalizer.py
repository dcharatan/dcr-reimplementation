from abc import ABC, abstractmethod
from .DynamicRelocalizer import DynamicRelocalizer
from typing import Tuple
import numpy as np


class FengDynamicRelocalizer(DynamicRelocalizer):
    @abstractmethod
    def _recreate_pose(
        self, reference_image: np.ndarray, initial_R: np.ndarray, initial_t: np.ndarray
    ) -> Tuple[np.ndarray]:
        raise Exception("Not implemented.")
