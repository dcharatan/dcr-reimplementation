from direct.showbase.ShowBase import ShowBase
from panda3d.core import WindowProperties, LQuaternionf
import numpy as np
from .Camera import Camera
from typing import Tuple
from scipy.spatial.transform import Rotation


class CameraPanda3D(ShowBase, Camera):
    def __init__(self, image_shape: Tuple[int], environment: str) -> None:
        # The image shape must be (rows, cols, channels = 3).
        assert len(image_shape) == 3 and image_shape[2] == 3
        ShowBase.__init__(self)
        Camera.__init__(self, image_shape)
        self.image_shape = image_shape

        # Set up Panda3D.
        self.scene = self.loader.loadModel(environment)
        self.scene.reparentTo(self.render)

        # Set the image resolution.
        props = WindowProperties()
        props.setSize(image_shape[1], image_shape[0])
        self.win.requestProperties(props)

    def _render_with_pose(self, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        # Set the camera position.
        # SciPy quaternions are (x, y, z, w), but Panda3D quaternions are (w, x, y, z). #nice
        q = Rotation.from_matrix(R).as_quat()
        self.camera.setQuat(LQuaternionf(q[3], q[0], q[1], q[2]))
        self.camera.setPos(t[0], t[1], t[2])

        # Get a frame from Panda3D.
        self.graphicsEngine.renderFrame()
        display_region = self.camNode.getDisplayRegion(0)
        tex = display_region.getScreenshot()
        data = tex.getRamImage()
        image = np.frombuffer(data, np.uint8)
        image.shape = (tex.getYSize(), tex.getXSize(), tex.getNumComponents())

        # Copy the image to a numpy array.
        image_np = np.empty(self.image_shape, dtype=np.uint8)
        np.copyto(image_np, image[:, :, 0:3])
        return image_np
