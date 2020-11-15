from ..camera_pose_estimation.FivePointEstimator import FivePointEstimator
import cv2 as cv
import numpy as np

imgL = cv.imread('/home/nishanth/Downloads/1.jpg')
imgR = cv.imread('/home/nishanth/Downloads/2.jpg')
imgL = cv.cvtColor(imgL,cv.COLOR_BGR2GRAY)
imgR = cv.cvtColor(imgR,cv.COLOR_BGR2GRAY)

imageL = np.asarray(imgL)
imageR = np.asarray(imgR)
K = np.array([[2759.5,0,1520.7],[0,2764.2,1006.8],[0,0,1]])
fpe = FivePointEstimator()
R, t = fpe.estimate_pose(imageL,imageR,K)
print(R)
print(t)