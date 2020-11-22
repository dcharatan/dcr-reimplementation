import numpy as np
from ..camera_pose_estimation.CameraPoseEstimator import CameraPoseEstimator
from typing import Tuple
import cv2 as cv


class FivePointEstimator(CameraPoseEstimator):
    def _estimate_pose(
        self, image1: np.ndarray, image2: np.ndarray, K: np.ndarray
    ) -> Tuple[np.ndarray]:
        # Code adapted from OpenCV documentation example
        # https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
        sift = cv.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(image1, None)
        kp2, des2 = sift.detectAndCompute(image2, None)
        # FLANN parameters for nearest neighbor search
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        good = []
        pts1 = []
        pts2 = []

        # Ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.8 * n.distance:
                good.append(m)
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)

        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)

        E, _ = cv.findEssentialMat(pts1, pts2, K)
        _, R_est, t_est, _ = cv.recoverPose(E, pts1, pts2)
        return R_est, t_est.squeeze()
