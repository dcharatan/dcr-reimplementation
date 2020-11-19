import numpy as np
from ..camera.Camera import Camera
from ..utilities import is_rotation_matrix, is_translation_vector


class CameraRig:
    # This is the camera the rig is holding.
    camera: Camera

    # These hold the unknown transformation from hand to eye.
    hand_eye_R: np.ndarray
    hand_eye_t: np.ndarray

    # These hold the hand's (robot's) current position.
    hand_R: np.ndarray
    hand_t: np.ndarray

    def __init__(
        self, camera: Camera, hand_eye_R: np.ndarray, hand_eye_t: np.ndarray
    ) -> None:
        assert isinstance(camera, Camera)
        assert is_rotation_matrix(hand_eye_R)
        assert is_translation_vector(hand_eye_t)
        self.camera = camera
        self.hand_eye_R = hand_eye_R
        self.hand_eye_t = hand_eye_t

    def set_position(self, hand_R: np.ndarray, hand_t: np.ndarray) -> None:
        assert is_rotation_matrix(hand_R)
        assert is_translation_vector(hand_t)
        self.hand_R = hand_R
        self.hand_t = hand_t

    def apply_rotation(self, R: np.ndarray) -> None:
        assert is_rotation_matrix(R)
        self.hand_R = R @ self.hand_R

    def apply_translation(self, t: np.ndarray) -> None:
        assert is_translation_vector(t)
        self.hand_t += t

    def get_eye_R(self) -> np.ndarray:
        return self.hand_eye_R @ self.hand_R

    def get_eye_t(self) -> np.ndarray:
        return self.hand_eye_t + self.hand_t