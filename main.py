import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

from src import training, detection

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
        output_file = './test_video_result.mp4'
        input_file = './test_video.mp4'
    else:
        output_file = './project_video_result.mp4'
        input_file = './project_video.mp4'


    # vehicle detector
    vehicle_detector = detection.VehicleDetector(svc, X_scaler)
    vehicle_detector.set_max_frames_skiped(5)

    clip = VideoFileClip(input_file)
    out_clip = clip.fl_image(vehicle_detector.find)
    out_clip.write_videofile(output_file, audio=False)



# training
svc, X_scaler = training.training()

runVideo()
#runTest()

