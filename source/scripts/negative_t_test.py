import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def triangulateAndCountInliers(pts1, pts2, R, T, inv_K):
    positive_inlier_count = 0
    num_pts = pts1.shape[0]
    cat_ones = np.ones((num_pts, 1))
    pts1_cated = np.concatenate((pts1, cat_ones), 1)
    pts2_cated = np.concatenate((pts2, cat_ones), 1)
    gamma1 = np.matmul(inv_K, pts1_cated.reshape(3, -1))
    gamma2 = np.matmul(inv_K, pts2_cated.reshape(3, -1))

    for i in range(num_pts):
        R_gamma1_expanded = np.expand_dims(np.matmul(-R, gamma1[:, i]), 1)
        gamma2_expanded = np.expand_dims(gamma2[:, i], 1)
        R_gamma1_gamma2_concat = np.concatenate((R_gamma1_expanded, gamma2_expanded), 1)
        R_gamma1_gamma2_pinv = np.linalg.pinv(R_gamma1_gamma2_concat)
        rho1_rho2 = np.matmul(R_gamma1_gamma2_pinv, T)
        if rho1_rho2[0] > 0 and rho1_rho2[1] > 0:
            positive_inlier_count += 1

    return positive_inlier_count


# Utility function to plot the epipolar lines between 2 images. Useful to see
# if we ever encounter a failure case for the 5-point algorithm
def plotEpipolarLines(
    image1_loc: str, image2_loc: str, gt_R: np.ndarray, gt_t: np.ndarray, K: np.ndarray
):
    # Code adapted from OpenCV documentation example
    # https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
    img1 = cv.imread(image1_loc, 0)
    img2 = cv.imread(image2_loc, 0)

    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
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
        if m.distance < 0.70 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    # This finds the F matrix from these SIFT points
    F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)

    # This below code block finds the F matrix from GT input
    t_x = np.array(
        [[0, -gt_t[2], gt_t[1]], [gt_t[2], 0, -gt_t[0]], [-gt_t[1], gt_t[0], 0]]
    )
    # np.mat([[0,-T2,T1],[T2,0,-T[0]],[-T1,T[0],0]])
    gt_E = np.matmul(gt_R, t_x)
    inv_K = np.linalg.inv(K)
    gt_F = np.matmul(np.linalg.inv(np.transpose(K)), np.matmul(gt_E, inv_K))

    # from IPython import embed

    # embed()

    # We select only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    def drawlines(img1, img2, lines, pts1, pts2):
        """img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines"""
        r, c = img1.shape
        img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
        img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
        for r, pt1, pt2 in zip(lines, pts1, pts2):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
            img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1)
            img1 = cv.circle(img1, tuple(pt1), 5, color, -1)
            img2 = cv.circle(img2, tuple(pt2), 5, color, -1)
        return img1, img2

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)
    plt.subplot(221), plt.imshow(img5)
    plt.subplot(222), plt.imshow(img3)
    # plt.show()

    # Run the above codeblock for GT F
    lines3 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, gt_F)
    lines3 = lines3.reshape(-1, 3)
    img7, img8 = drawlines(img1, img2, lines3, pts1, pts2)
    lines4 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, gt_F)
    lines4 = lines4.reshape(-1, 3)
    img9, img10 = drawlines(img2, img1, lines4, pts2, pts1)
    plt.subplot(224), plt.imshow(img9)
    plt.subplot(223), plt.imshow(img7)
    plt.show()


# R for correct case
# R_delta = np.array(
#     [
#         [0.99988411, 0.00245706, 0.01502412],
#         [-0.00251743, 0.99998883, 0.00400114],
#         [-0.01501412, -0.0040385, 0.99987913],
#     ]
# )
# # t for the correct case
# t_delta = np.array([0.46602022, 0.377932, -0.42737709])

# # R for incorrect case
R_delta = np.array(
    [
        [0.99324032, 0.06042289, -0.09910978],
        [-0.0660434, 0.9963304, -0.05444273],
        [0.0954565, 0.06062027, 0.99358605],
    ]
)

# This t is for the incorrect case
t_delta = np.array([1.0, 1.0, -1.0])
K = np.array(
    [
        [1.71560547e03, 0.00000000e00, 8.00000000e02],
        [0.00000000e00, 1.93005615e03, 6.00000000e02],
        [0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)
plotEpipolarLines(
    "/home/nishanth/Desktop/dcr-reimplementation/tmp_TPA_initial_pose.png",
    "/home/nishanth/Desktop/dcr-reimplementation/tmp_TPA_target_pose.png",
    R_delta,
    t_delta,
    K,
)