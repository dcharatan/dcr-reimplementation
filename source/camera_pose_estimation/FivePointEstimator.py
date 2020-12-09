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
            if m.distance < 0.7 * n.distance:
                good.append(m)
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)

        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)

        # E, _ = cv.findEssentialMat(pts1, pts2, cameraMatrix=K, method='RANSAC', prob=0.999999)
        E, _ = cv.findEssentialMat(
            pts1, pts2, cameraMatrix=K, method=cv.LMEDS, prob=0.999999999999
        )
        # E, _ = cv.findEssentialMat(pts1, pts2, cameraMatrix=K, method=cv.FM_LMEDS)

        # Compute R and t from OpenCV's recoverPose() function
        _, R_est, t_est, _ = cv.recoverPose(E, pts1, pts2)

        # Use our custom triangulation function to check whether t or -t has
        # more inliers, and choose the correct t accordingly
        pos_t_inliers = self._triangulate_and_count_inliers(
            pts1, pts2, R_est, t_est, np.linalg.inv(K)
        )
        neg_t_inliers = self._triangulate_and_count_inliers(
            pts1, pts2, R_est, -t_est, np.linalg.inv(K)
        )
        if neg_t_inliers > pos_t_inliers:
            t_est = -t_est

        return R_est, t_est.squeeze()

    def _triangulate_and_count_inliers(self, pts1, pts2, R, T, inv_K):
        positive_inlier_count = 0
        num_pts = pts1.shape[0]
        cat_ones = np.ones((num_pts, 1))
        pts1_cated = np.concatenate((pts1, cat_ones), 1)
        pts2_cated = np.concatenate((pts2, cat_ones), 1)
        gamma1 = inv_K @ pts1_cated.transpose()
        gamma2 = inv_K @ pts2_cated.transpose()

        for i in range(num_pts):
            R_gamma1_expanded = np.expand_dims(np.matmul(-R, gamma1[:, i]), 1)
            gamma2_expanded = np.expand_dims(gamma2[:, i], 1)
            R_gamma1_gamma2_concat = np.concatenate(
                (R_gamma1_expanded, gamma2_expanded), 1
            )
            R_gamma1_gamma2_pinv = np.linalg.pinv(R_gamma1_gamma2_concat)
            rho1_rho2 = np.matmul(R_gamma1_gamma2_pinv, T)
            if rho1_rho2[0] > 0 and rho1_rho2[1] > 0:
                positive_inlier_count += 1

        return positive_inlier_count