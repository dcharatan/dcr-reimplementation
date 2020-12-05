import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

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
R_delta = np.array(
    [
        [0.99988411, 0.00245706, 0.01502412],
        [-0.00251743, 0.99998883, 0.00400114],
        [-0.01501412, -0.0040385, 0.99987913],
    ]
)
# t for the correct case
t_delta = np.array([0.46602022, 0.377932, -0.42737709])

# # R for incorrect case
# R_delta = np.array(
#     [
#         [0.99274945, 0.05999797, -0.10415747],
#         [-0.06612377, 0.9962168, -0.05638909],
#         [0.10038019, 0.06286753, 0.99296097],
#     ]
# )

# # This t is for the incorrect case
# t_delta = np.array([1.0, 1.0, -1.0])
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