## Self-Driving Car Engineer Nanodegree

---

**Vehicle Detection and Tracking - Project 5**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # "Image References"
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.jpg
[image3]: ./output_images/sliding_windows.jpg
[image4]: ./output_images/sliding_window.jpg
[image5]: ./output_images/bboxes_and_heat.jpg
[image6]: ./output_images/labels_map.jpg
[image7]: ./output_images/output_bboxes.jpg
[video1]: ./project_video_out.mp4



---


### Histogram of Oriented Gradients (HOG)

#### 1. Extraction of HOG features from the training images.

The code for this step is contained in the get_hog_features() function (lines #59 through #75 in the "vehicle_det_trac.py' file )

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I tested random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB,YUV and HLS `color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Choice of HOG parameters.

I tried various combinations of parameters and color spaces looking at the different color channels to figure out if it makes sense to use all color channels or if it is sufficient to use just one color channel. In the RGB case ( first two rows in the HOG image above ) it is visible that all channels are similar so probably one channels should be sufficient but after some testing on the 6 test images I noticed a problem with detecting a white car. In the HLS case (bottom two rows in the HOG image above) it looks like the H and S channels carry similar information but the V channel looks different, so it is probably desirable to use more than one channel in this case. Similarly in the YUV case ( middle two rows in the HOG image above)  U and V look similar but Y is different. After some experimentation with the test images I noticed that the problem with the white car I was seeing using the RGB color space went away when I used the YUV or HLS color space. I tried all other color spaces with different parameters but the HLS version using all color channels provided the bet result, so I selected HLS color space for the HOG features.

#### 3. Classifier training using selected HOG features.

I trained a linear SVM classifier using spatial, color and HOG features extracted from the large vehicle and not-vehicle  data set provide by Udacity ([vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) images). Initially I tried using the small dataset only but I ran into a problem detecting the white car reliably. 

The code used to train the classifier can be found at lines from #517 trough #645. First I loaded  the car and not-cars images file names to two arrays: cars and not cars.  Then I loaded the individual images and extracting the spatial, color and HOG features using the extract_features() function at line # 147 in the "vehicle_det_trac.py" file. I experimented with different setting and to create the final project video I used the following parameters :

```
color_space = 'HLS'# Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = "ALL" # 0  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 24#16  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
```

After extracting the features I normalized them at line #590 using the StandardScaler() function from the sklearn package generating the scaled_X data set. Then I randomized and and split the data set at line # 597 generating traing and test data set I used to train the LinerSVC classifier at line #605. I achieved the Testest Accuracy of 0.9929. 

### Sliding Window Search

#### 1. Implementation of sliding window search.

The code for this step can be found in the image_processing_pipeline() function at lines #403 through #419.  To implement the sliding window search I used the find_cars() function from out lessen with three scales (1.0, 1.5 and 2.0) as an input parameter that resulted in 3 window sizes ( 64x64, 96x96, and 128x128) pixels and overlap of 75%. For the 64x64 window I used Y range from 400 to 528, for 96x96 window I used Y range from 408 to 600 and for the 128x128 window i used Y range from 400 to 756.  All windows used to search for cars are shown in the pictures below. After experiments with different scales, offsets and features parameters I settled on the 3 window sizes implementation because it provided acceptable accuracy and decent speed of about 1 frame per second on my laptop.

![alt text][image3]

#### 2. Examples of test images.

For the final solution I decided to use window search with three scales using HLS 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided the best overall result.  Here are some examples of car search results using three window sizes:

![alt text][image4]
---

### Video Implementation

#### 1. Result
Here's a [link to my video result](./project_video_out.mp4)


#### 2. Filtering of false positives.

I recorded the positions of positive detections in each frame of the video in the image_processing_pipeline() function at lines from #432 trough 440 .  From the positive detections from the last 5 frames  I created a heatmap at line #442 and then thresholded that map at line #474 to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` at line #480 to identify individual blobs in the heatmap.   I then constructed bounding boxes at line #491 to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are five frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all five frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

When working on color space selection and best parameters selection for the HOG features I had difficulty finding the correct combination of parameters/features to be able to detect the white car reliably in the project video.  After looking at the data set I noticed that there is not too many white cars so maybe addition of white cars to the dataset could provide a better training result. In addition to extended data set my pipeline could be improved to better predict the expected position of the car in the next frame.  There are still some false detections that can be optimized based on predictions for the next frame 

