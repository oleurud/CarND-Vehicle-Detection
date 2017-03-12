"""
Based on lesson 10
"""

import pickle
import os
import numpy as np
import cv2

 # chessboard corners
xCorners = 9
yCorners = 6

def calibrate():
    """
    Calibrate camera
    Save a file with the params of calibration
    Returns the path of the camera calibration file
    """

    # paths
    dir = os.path.dirname(__file__)
    imagesPath = dir + "/../../camera_cal/"
    calibrationFilePath = imagesPath + 'calibration.p'

    if os.path.isfile(calibrationFilePath) is False:
        # Import images
        images = os.listdir(imagesPath)

        # 3D points
        objectPoints = []
        # 2D points
        imagePoints = []

        # Prepare object points
        objectPoint = np.zeros((xCorners * yCorners, 3), np.float32)
        objectPoint[:, :2] = np.mgrid[0:xCorners, 0:yCorners].T.reshape(-1, 2)

        for index, file_name in enumerate(images):
            image = cv2.imread(imagesPath + file_name)
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            ret, corners = cv2.findChessboardCorners(gray_image, (xCorners, yCorners), None)

            # if corners found, save
            if ret:
                objectPoints.append(objectPoint)
                imagePoints.append(corners)

        # calibrate
        imageSize = (image.shape[1], image.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, imageSize, None, None)

        # save values to transform 3D to 2D
        data = {'mtx': mtx, 'dist': dist}

        # save file
        with open(calibrationFilePath, 'wb') as f:
            pickle.dump(data, file=f)

    return calibrationFilePath
