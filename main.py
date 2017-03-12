import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

from src.lane_detection import camera_calibration, process_image, process_lines, process_control
from src.car_detection import training, detection


def processImage(image, imageName=False):
    """
    Recives a image and do the process to detect the lane
    Returns the image undistort with the lane colored

    The imageName param is used for testing purpose.
    When this param is recieved, the method save each image created in the process
    """
    # undistort image
    undistort = process_image.undistort(image, calibrationFilePath)
    if imageName is not False:
        plt.imshow(undistort, cmap='gray')
        plt.savefig('output_images/undistort' + imageName)

    # draw lane lines
    if imageName is not False:
        laneLinesImage = process_image.drawPolygon(undistort)
        plt.imshow(laneLinesImage, cmap='gray') 
        plt.savefig('output_images/laneLines' + imageName)

        laneLinesTransformedImage = process_image.perspectiveTransform(undistort, camera_calibration.xCorners, camera_calibration.yCorners)
        laneLinesTransformedImage = process_image.drawPolygon(laneLinesTransformedImage, True)
        plt.imshow(laneLinesTransformedImage, cmap='gray')
        plt.savefig('output_images/laneLinesTransformed' + imageName)


    # process image color
    preprocessColor = process_image.preprocessColor(undistort)
    if imageName is not False:
        plt.imshow(preprocessColor, cmap='gray')
        plt.savefig('output_images/preprocess' + imageName)

    # change image perspective to a top-down view
    transformedImage = process_image.perspectiveTransform(preprocessColor, camera_calibration.xCorners, camera_calibration.yCorners)
    if imageName is not False:
        plt.imshow(transformedImage, cmap='gray')
        plt.savefig('output_images/transformed' + imageName)


    # detect lines
    if ProcessControl.processFromScratch() is True:
        linesData = process_lines.fullSearch(transformedImage)
    else:
        linesData = process_lines.searchFromFoundLines(transformedImage, ProcessControl.getLeftFit(), ProcessControl.getRightFit())

    # plot detected lines
    if imageName is not False:
        process_lines.saveFileOfFoundLines(
            transformedImage,
            linesData['left_lane_inds'],
            linesData['right_lane_inds'],
            linesData['nonzerox'],
            linesData['nonzeroy'],
            linesData['ploty'],
            linesData['left_fitx'],
            linesData['right_fitx'],
            'output_images/lines' + imageName)


    # lines curvature and distance from lane center
    left_curvature, right_curvature = process_lines.getLinesCurvature(linesData['ploty'], linesData['left_fitx'], linesData['right_fitx'])
    distance_from_center = process_lines.getDistanceFromCenter(transformedImage.shape[1], linesData['left_fitx'], linesData['right_fitx'])

    # write log
    """
    print(' ')
    print('left', left_curvature)
    print('right', right_curvature)
    print('distance', distance_from_center)
    """

    # validate result of this frame
    frameValidation = ProcessControl.validateImageResult(left_curvature, right_curvature)
    if frameValidation is True:
        # if ok, save the lines data. If not, we will use the last correct data in the next frame
        ProcessControl.setLeftFit(linesData['left_fit'])
        ProcessControl.setRightFit(linesData['right_fit'])


    # return output image
    laneImageResult = process_lines.drawResult(
        transformedImage,
        linesData['left_fitx'],
        linesData['right_fitx'],
        linesData['ploty'],
        process_image.getInversePerspectiveTransformMatrix(transformedImage),
        undistort,
        left_curvature,
        right_curvature,
        distance_from_center,
        imageName)

    # vehicle detector
    vehicle_detector = detection.VehicleDetector(svc, X_scaler)
    vehicle_detector.set_max_frames_skiped(5)
    return vehicle_detector.find(laneImageResult)


def runTest():
    """
    Test the code with the test images
    """

    # Import test images
    testImagesPath = 'test_images/'
    images = os.listdir(testImagesPath)

    for imageName in images:
        # vehicle detector
        vehicle_detector = detection.VehicleDetector(svc, X_scaler)
        vehicle_detector.set_max_frames_skiped(0)

        print('image: ' + imageName)
        image = mpimg.imread(testImagesPath + imageName)
        out_img = vehicle_detector.find(image)
        plt.imshow(out_img)
        plt.savefig('output_images/' + imageName)


def runVideo(test = False):
    """
    Run the code using the videos
    """
    
    if test:
        output_file = './test_video_result_4an5.mp4'
        input_file = './test_video.mp4'
    else:
        output_file = './project_video_result_4an5.mp4'
        input_file = './project_video.mp4'


    clip = VideoFileClip(input_file)
    out_clip = clip.fl_image(processImage)
    out_clip.write_videofile(output_file, audio=False)



# training
svc, X_scaler = training.training()

# Instance the process control
ProcessControl = process_control.ProcessControl()
# calibrate camera
calibrationFilePath = camera_calibration.calibrate()

runVideo()
#runTest()

