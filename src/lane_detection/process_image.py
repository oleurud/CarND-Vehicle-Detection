"""
Based on lessons 11, 17, 28, 29
"""

import pickle
import cv2
import numpy as np

def preprocessColor(image):
    """
    Returns processed image color
    Process:
        - Get the image binary of the channel S of the HLS image 
        - Get the combined threshold of directional gradient, gradient magnitude and gradient direction
        - Combined both
    """
    #return preprocessColorGray(image)
    #return preprocessColorR(image)
    #return preprocessColorS(image)
    #return preprocessColorThreshold(image)

    S = preprocessColorS(image)
    threshold = preprocessColorThreshold(image)

    combined_binary = np.zeros_like(S)
    combined_binary[(threshold == 1) | (S == 1)] = 1
    return combined_binary


def preprocessColorGray(image):
    """
    Returns the binary of the gray image
    """
    thresh = (180, 255)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    binary = np.zeros_like(gray)
    binary[(gray > thresh[0]) & (gray <= thresh[1])] = 1
    return binary

def preprocessColorR(image):
    """
    Returns the binary of the channel R of the RGB image
    """
    R = image[:,:,0]
    thresh = (200, 255)
    binary = np.zeros_like(R)
    binary[(R > thresh[0]) & (R <= thresh[1])] = 1
    return binary    

def preprocessColorS(image):
    """
    Returns the binary of the channel S of the HLS image
    """
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]
    thresh = (90, 255)
    binary = np.zeros_like(S)
    binary[(S > thresh[0]) & (S <= thresh[1])] = 1
    return binary


def preprocessColorThreshold(image):
    """
    Returns the combined threshold of directional gradient, gradient magnitude and gradient direction
    """

    ksize = 3

    gradx = abs_sobel_thresh(image, orient='x', thresh_min=20, thresh_max=100)
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))

    combined = np.zeros_like(dir_binary)
    combined[(gradx == 1) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined


def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh_min=0, thresh_max=255):
    """
    Returns binary directional gradient of image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
    return sxbinary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    """
    Returns binary gradient magnitude of image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    return binary_output
    
def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    """
    Returns binary gradient direction of image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return binary_output


def drawPolygon(image, transformed=False):
    """
    Returns the image with the perspective transform polygon drawn
    """
    #draw the lines in a new image
    image = np.copy(image)

    corners = getCornersOfView(image)
    if transformed is False:
        corners = np.int32(corners['src'])
    else:
        corners = np.int32(corners['dst'])
    corners = corners.reshape((-1,1,2))

    cv2.polylines(image, [corners], True, (255,0,0), 2)

    return image


def undistort(image, calibrationFilePath):
    """
    undistort image based in camera_calibration
    """

    calibrationData = pickle.load( open(calibrationFilePath, "rb") )
    return cv2.undistort(image, calibrationData['mtx'], calibrationData['dist'], None, calibrationData['mtx'])


def perspectiveTransform(undistortImage, xCorners, yCorners):
    """
    returns image to a top-down view
    """

    M = getPerspectiveTransformMatrix(undistortImage)
    img_size = (undistortImage.shape[1], undistortImage.shape[0])
    return cv2.warpPerspective(undistortImage, M, img_size)


def getPerspectiveTransformMatrix(undistortImage):
    """
    Returns the perspective transform matrix for perspectiveTransform
    """

    corners = getCornersOfView(undistortImage)
    src = np.float32(corners['src'])
    dst = np.float32(corners['dst'])

    return cv2.getPerspectiveTransform(src, dst)


def getInversePerspectiveTransformMatrix(undistortImage):
    """
    Returns the perspective transform matrix for perspectiveTransform
    """

    corners = getCornersOfView(undistortImage)
    src = np.float32(corners['src'])
    dst = np.float32(corners['dst'])

    return cv2.getPerspectiveTransform(dst, src)


def getCornersOfView(undistortImage):
    """
    Returns the poinst for perspective transform
    """
    xsize = undistortImage.shape[1]
    ysize = undistortImage.shape[0]

    xmid = xsize/2
    upper_margin = 85
    lower_margin = 480
    upper_bound = 460
    lower_bound = 690
    dst_margin = 450

    # src corners
    src = [
        [xmid - lower_margin, lower_bound],
        [xmid - upper_margin, upper_bound],
        [xmid + upper_margin, upper_bound],
        [xmid + lower_margin,lower_bound]
    ]

    # dst corners
    dst = [
        [xmid - dst_margin, ysize],
        [xmid - dst_margin, 0],
        [xmid + dst_margin, 0],
        [xmid + dst_margin, ysize]
    ]

    return {'src': src, 'dst': dst}
