import numpy as np
import cv2
import matplotlib.pyplot as plt


def fullSearch(binary_warped):
    """
    Find the seccions of the image where the lines are from scratch
    """
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:, :], axis=0)

    # Find the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Config sliding windows
    nwindows = 9
    window_height = np.int(binary_warped.shape[0]/nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Process the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        """
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        """

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Save these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))


    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    return processSeach(binary_warped, left_lane_inds, right_lane_inds, nonzerox, nonzeroy)


def searchFromFoundLines(binary_warped, left_fit, right_fit):
    """
    Find the seccions of the image where the lines are from previous valid search data
    """
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    return processSeach(binary_warped, left_lane_inds, right_lane_inds, nonzerox, nonzeroy)


def processSeach(binary_warped, left_lane_inds, right_lane_inds, nonzerox, nonzeroy):
    """
    Find the lines based on the fullSearch or searchFromFoundLines methods
    Returns all the parameters needed included the lines functions
    """
     # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    # values in x axis for each pixel
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    return {'left_lane_inds': left_lane_inds,
            'right_lane_inds': right_lane_inds,
            'left_fit': left_fit,
            'right_fit': right_fit,
            'nonzerox': nonzerox,
            'nonzeroy': nonzeroy,
            'ploty': ploty,
            'left_fitx': left_fitx,
            'right_fitx': right_fitx}


def saveFileOfFoundLines(binary_warped, left_lane_inds, right_lane_inds, nonzerox, nonzeroy, ploty, left_fitx, right_fitx, filePath):
    """
    Save an image file with the lines
    """
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    # colors
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # save in file
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.savefig(filePath)
    plt.close()


def drawResult(warped, left_fitx, right_fitx, ploty, Minv, undist, left_curvature, right_curvature, distance_from_center, plot=False):
    """
    Draw the result image of the video with the lane colored
    Draw in the image the curavature and distance from center information
    Returns the image
    """
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (warped.shape[1], warped.shape[0])) 
    # Combine the result with the original image
    image_result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    # add the texts
    font = cv2.FONT_HERSHEY_SIMPLEX
    meanCurvature = (left_curvature + right_curvature) / 2
    curvature = 'Curvature: {:.0f}m '.format(meanCurvature)
    cv2.putText(image_result, curvature, (100, 50), font, 1, (255, 255, 255), 2)

    if distance_from_center > 0:
        distance = 'Vehicle is {:.2f}m left of center '.format(distance_from_center)
    elif distance_from_center <= 0:
        distance = 'Vehicle is {:.2f}m right of center '.format(np.absolute(distance_from_center))
    cv2.putText(image_result, distance, (100, 90), font, 1, (255, 255, 255), 2)

    if plot is not False:
        plt.imshow(image_result)
        plt.savefig('output_images/result' + plot)
        plt.close()

    return image_result


# meters per pixel in y dimension
ym_per_pix = 30/720 
# meters per pixel in x dimension
xm_per_pix = 3.7/700 

def getLinesCurvature(ploty, leftx, rightx):
    """
    Returns the curvature of the lines in the real world
    """
    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

    y_max = np.max(ploty)

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

    # Calculate the curvature in meters
    left_curvature = ((1 + (2*left_fit_cr[0]*y_max*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curvature = ((1 + (2*right_fit_cr[0]*y_max*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    return left_curvature, right_curvature


def getDistanceFromCenter(x_max, left_fitx, right_fitx):
    """
    Returns the position of the vehicle from the center of the lane
    """
    center = (left_fitx[0] + right_fitx[0]) / 2.0
    return (center - x_max / 2.0) * xm_per_pix