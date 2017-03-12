#**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./report_images/training_car.png
[image2]: ./report_images/trainig_not_car.png
[image3]: ./report_images/hog-sub.jpg
[image4]: ./report_images/test1.jpg
[image5]: ./report_images/test4.jpg
[image6]: ./report_images/test5.jpg
[image7]: ./report_images/test1_c.jpg
[image8]: ./report_images/test4_c.jpg
[image9]: ./report_images/test5_c.jpg


####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the method `get_hog_features` of the file `lesson_functions.py`

The process started in the file `training.py` where I started by reading all the `vehicle` and `non-vehicle` images. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

Vehicle image example: 

![alt text][image1]

Not vehicle image example: 

![alt text][image2]

Then I call to the method `extract_features` of the file `lesson_functions.py` who calls the method `get_hog_features`. I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).

Finally, I have used a `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`

####2. Explain how you settled on your final choice of HOG parameters.

For this purpose, I have played especially with diferent `pixels_per_cell` and `cells_per_block` trying to get the best result of the trainig and testing.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In this process, is not only important the HOG features. I have played with:

- the classifier: linear SVC get better results (97.3%) than decision tree classifier (93.3%) in test accuracy
- color space: the best option was YCrCb with a test accuracy of 99.13%
- the hog channels: the best options was get all the channels incrasing the test accuracy to 99.44%

As you can see in the method `extract_features` of the file `lesson_functions.py`, in addition to the HOG features, I used spatially binned color and histograms of color to get the color features of the 3 channels of the YCrCb image.

In all cases, I used the `sklearn.preprocessing.StandardScaler` to normalize the results of the features and split randomly the test images into 2 groups: training and testing.

And finaly, I used the `sklearn.svm.LinearSVC` as classifier. The final test accuracy was 99.44%

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The slidding process was included in the vehicle detection process. The code for this step is contained in the method `find_cars` of the class VehicleDetector in the file `detection.py`.

First of all, this method looks for cars inside an small region of the image definid by to 2 heights positions and all the full weight. The reason for that is to look where the vehicles are, not in the sky ;) Allow me to use an image of the course to represent it:

![alt text][image3]

Inside this area, I getting all the HOG features once and then I loop the image in windows of 8x8 cells as you can see in the image. This process is repeated for diferent scales `[1, 1.25, 1.5, 1.75, 2]` to get different sizes of this windows because the vehicles have different sizes depending on the distance to our vehicle.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

This are some examples at the end of the `find_cars` of the class VehicleDetector in the file `detection.py`.

![alt text][image4]
![alt text][image5]
![alt text][image6]


As you can see in the last image, I have multiple detections for each vehicle as expected due to the slidding windows. And some times I have false positives. I will explain how to solve its later.

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_result.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

After the process of finding cars, I need to clean multiple detections and remove false positives. You can see the code in the method `find` of the class VehicleDetector in the file `detection.py`.

```
heatmap = self.create_heatmap(heatmap, bounding_boxes)
heatmap = self.apply_threshold(heatmap, 2)
self.last_labels = label(heatmap)
```
The most of the time, the false positives has only 1 results, and the cars has more than 1. I created a heatmap and then filtered by a minimum of 2 to identify vehicle positions.

Without false positives, then I used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.  

As result of this process, I have this results of the same images that saw before:

![alt text][image7]
![alt text][image8]
![alt text][image9]

In the last image, now I have not any false positive, but I lost one of the cars (because this car only has 1 detection).

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

How I have explained, I lose some detections in the process of the false positives deletion. I tried to play with diferent configurations of the slidding windows, but I have not a google result.

As always, the code is a good starting point, but is impossible to use in real time. I have improved the time process as you can see in the method `find` of the class VehicleDetector in the file `detection.py` with the general idea of not process each frame if the previous result was good.

