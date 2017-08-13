
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### My solution is built along the 3 notebooks files
####1. Data Exploration: create the datasets
####2. HOG: feature extraction
####3. Seach vehicles: using sliding windows

###Writeup / README

###1. Data Exploration

* Since the data is divided between multiple folders, I've concatenated them and divided it: Train, Validation and Test with 70%, 20% and 10% of the images, in that order.
    * Number of samples in cars training set:  6154
    * Number of samples in notcars training set:  6277
    * Number of samples in cars validation set:  1758
    * Number of samples in notcars validation set:  1794
    * Number of samples in cars test set:  880
    * Number of samples in notcars test set:  897
* Save pickle file with the data for easy access in part2

###2. Histogram of Oriented Gradients (HOG)

* First of all I loaded the images from the pickle created in step 1
* The I pass the images, Car and not Cars to a feature extraction funciont utils.extract_features_bulk, which does:
    * First it apply a color conversion to 'HSL'
    * compute binned color features with a size of (16, 16)
    * compute color histogram features with 32 bins
    * extract the HOG features using from skimage.feature import hog
    * append all the features for each img in a single list
    
* Apply normalization to the features using StandardScaler from sklearn
* Split back the training, validation and test features for Cars and Not cars
* Create the labels vector, 1 if car, 0 if not car
* build the model SVM,LinearSVC, and trained using gridsearchcv to find the best parameters: The best parameters are {'C': 0.01} with a score of 0.99
* Save the data for the final part

#####a. Explain how you settled on your final choice of HOG parameters.

* I tried with a several different color spaces and HOG parameters and trained a linear SVM using different combinations of HOG features extracted from the color channels. 
* I discarded RGB color space, it does not work well with saturations of white and black. YUV and YCrCb were not good when using all channels. 
* I choosed then, HLS, with a pixels_per_cell=(8,8). 
* Orient=9 was doing the same as any other value above it. 
* cells_per_block=(2,2) was like the same for any value above it.

####b. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

* I trained a linear SVM using all channels of images converted to HLS space.
* I included spatial features color features as well as all three HLS channels, because using less than all three channels reduced the accuracy considerably. 
* The final feature vector has a length of 6156 elements, most of which are HOG features.

###3.Sliding Window Search

####a. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

* At the Part3 file, I applied the utils.slide_window to find the windows in the test images and the utils.search_windows to find the hot windows
* I choosed the  Min & Max in y to search in slide_window() of [370, 656] by try and error, looking for the best results
* Same method of seach was used for the overlat of .70

####b. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The examples can be found at the part 3 notebook.

### Video Implementation

####a. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a link to the video https://youtu.be/wGEeH4bkGbE (./project_video.mp4)


####b. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The execution will only work at 3FPS, which is not gonna work considering that the human eye sees 29.7 fps, should be equivalent or vetter
I saw a solution called YOLO which works for 20fps, their metodoly is, only look once, the same should be applied here, to improve speed  compute the HOG features only once for the entire region of interest and then select the right feature vectors.
