## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./md_images/hog_image.jpg
[image3]: ./md_images/sliding_window.png
[image4]: ./md_images/sliding_windows1.png
[image5]: ./md_images/sliding_windows2.png
[image6]: ./md_images/sliding_windows3.png
[image7]: ./md_images/sliding_windows4.png
[heatmap]: ./md_images/bboxes_and_heat.png
[image8]: ./md_images/bboxes_and_heat1.png
[image9]: ./md_images/bboxes_and_heat2.png
[image10]: ./md_images/bboxes_and_heat3.png
[image11]: ./md_images/bboxes_and_heat4.png
[image12]: ./md_images/bboxes_and_heat5.png
[image13]: ./md_images/bboxes_and_heat6.png
[image14]: ./md_images/labels_map.png
[image15]: ./md_images/output_bboxes.png
[video1]: ./output_videos/project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines #145 through #162 of the file called `lesson_functions.py`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

The parameters used in to train the classifier 

|Parameter | value|
|----------|--------|
|color space|'YCrCb'|
|HOG orientations|9|
|HOG pixels per cell|8|
|HOG cells per block|2|
|HOG channel|"ALL"|
|Spatial binning dimensions|(16, 16)|
|Number of histogram bins|16|
|Spatial features on or off|True|
|Histogram features on or off|True|
|HOG features on or off|True|

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the previous parameters, The code is contained in lines #53 through #98 of the file called `classifier.py`.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to use three window sizes (64x64, 96x96, 128x128).

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4] ![alt text][image5]

![alt text][image6] ![alt text][image6]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://github.com/messam2/CarND-Vehicle-Detection/tree/master/output_videos)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code for this step is contained in lines #425 through #435 of the file called `lesson_functoins.py', I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][heatmap]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image14]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image15]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

##### Challenges
* Finding the most efficient feature vector to train the classifier. In addition, finding the best parameters for the HOG feature.

##### Improvements and Future work
* Imporving the tracking algorithm to improve the quality of detections. using some kind of tracking system to keep trak of detected cars.

