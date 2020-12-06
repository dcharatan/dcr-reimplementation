import numpy as np
from typing import Tuple
from scipy.spatial.transform import Rotation
from ..utilities import is_rotation_matrix, convert_angles_to_matrix
from .Camera import Camera
import subprocess
import cv2
import os


class CameraBlender(Camera):
    file_name: str

    def __init__(self, image_shape: Tuple[int], file_name: str) -> None:
        # The image shape must be (rows, cols, channels = 3).
        assert len(image_shape) == 3 and image_shape[2] == 3
        Camera.__init__(self, image_shape)
        self.file_name = file_name

    def _render_with_pose(self, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        pose_correction = convert_angles_to_matrix(180, 0, 0)
        euler = Rotation.from_matrix(R @ pose_correction).as_euler("xyz")

        # Call the blender script.
        temp_image_location = "tmp_blender_image.png"
        command = f'blender -b "{self.file_name}" -P source/camera/blender_script.py -- {temp_image_location} {self.image_shape[1]} {self.image_shape[0]} {euler[0]} {euler[1]} {euler[2]} {t[0]} {t[1]} {t[2]}'
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        process.wait()

        # Collect the K matrix from stdout.
        # This is really hacky.
        k_matrix_index = None
        k_matrix_elements = []
        for line in iter(process.stdout.readline, ""):
            line = str(line)
            if k_matrix_index is not None:
                number = line[2:-4]
                k_matrix_elements.append(float(number))
                k_matrix_index += 1
                if k_matrix_index == 9:
                    break
            if "K MATRIX INCOMING:" in line:
                k_matrix_index = 0

        self.K = np.array(k_matrix_elements).reshape((3, 3))

        # Read, remove and return the image.
        image = cv2.imread(temp_image_location)
        os.remove(temp_image_location)
        return image

    def _get_K(self) -> np.ndarray:
        assert self.K is not None
        return self.K
