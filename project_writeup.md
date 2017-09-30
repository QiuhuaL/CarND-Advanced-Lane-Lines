## Advanced Lane Finding Project 

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image_undist]: ./output_images/undistort_output.png "Undistorted"
[image_undist_warped]: ./output_images/undist_warped_cal12.png "Undistorted and Warped"
[image_undist_test1]: ./output_images/undist_test1.png "Road Undistorted"
[image_warp1]: ./output_images/warp1.png "Road UnWarped 1 "
[image_warp2]: ./output_images/warp2.png "Road UnWarped 2"
[image_grad]: ./output_images/gradientThresholded_x.png "Graident Thresholding x Orient"
[image_color]: ./output_images/colorThresholded_s.png "Color Thresholding"
[image_HLS]: ./output_images/HLS.png "HLS channels"
[image_x_s]: ./output_images/thresholded_x_s.png "Combined Thresholding"
[image_test6_binary]: ./output_images/test6_binary.png "Image Test 6 Binary Warped"
[image_test6_result]: ./output_images/test6_result.png "Image Test 6, Detection" 
 
 ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion
 corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in the first function camera_calibration in `./AdvancedLaneLines.ipynb` .  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. The camera calibration matrix and distortion coefficients are saved in the pickle file ./camera_cal/camera_calibration_pickle.p". I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

  ![alt text][image_undist]

I also tried thge persepctive transform on the chess board images with the functions cv2.getPerspectiveTransform and cv2.warpPerspective, with the outer four detected corners as the source points and 4 points on a rectangle shape as the destination points. Here is an example result. 
  
  ![alt text][image_undist_warped]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
For this step, I have a function undistort(img, mtx, dist) which calls the cv2.undistort() function and applied the camera calibration and distortion coefficients mtx and dist to the input image 'img'. And example of original image and the undistorted image is shown below:

![alt text][image_undist_test1]

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes functions called `warp()` and `perspective_tranform()`, which appears in the cell of "Perspective Transform" the file `./AdvancedLaneLines.ipynb` .  The `perspective_transform()` function takes the source (`src`) and destination (`dst`) points as inputs and return perspective transform coefficients (`M_perp`) and (`M_perp_inv`) and the function `warp()` function takes as inputs an image (`img`), and applied the transform coefficients .  

I chose the hardcoded source and destination points from the test image with the straight lines, 'test_images/straight_lines1.jpg', 

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 738, 481       |  980, 0       | 
| 1115, 720      |  980, 720     |
| 196, 720       |  320, 720     |
| 550, 481       |  320, 0       |

Since the source points are from the straight lines,  the perspective transform will transform those two lines into two vertical and parallel lines, as shown below.
  
  ![alt text][image_warp1]

I verified that my perspective transform was working as expected onto other test images and their warped counterparts to verify that the lane lines appear parallel in the warped images.

  ![alt text][image_warp2]

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (Code is in the cells of "Gradient Thresholding" and  "Color Thresholding" in the file `AdvancedLaneLines.ipynb`).

* For the gradient thresholding, Gaussian smoothing was first applied to the gray scale image, and applied the thresholds on the x orientation gradient. The function is called `abs_sobel_thresh()`, here is an example:
    
  ![alt text][image_grad]

* For the color threshoding, the image was first transformed into HLS channels, and the thresholing was applied on the s-channel. The corresponding function is `hls_select()` . Here is an example of the S-thresholed image and the HLS channels.
  
  ![alt text][image_color]
  
  ![alt text][image_HLS]


Then the two thresholding was combined and the perspective transform was applied to the thresholded image. Here's an example of my final output for this step.  
  
  ![alt text][image_x_s]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
First, line finding without continuous tracking from frame to frame in a video is implemented in the "Line Finding" cell, the `function find_lines ()` in the file `AdvancedLaneLines.ipynb`. It includes the following steps:

*  Take the histogram of the bottom half of the binary warped image;

*  Find the peaks of the left and right havles of the histogram as the starting point for the left and right lines; 

*  Devide the images into 9 windows regions with window margin 100, along the y-direction and find the non-zero pixels in the windows

* Concatenate the arrays of indices with non-zero pixels and apply np.polyfit to fit a polynomial to the left and right lane lines

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The function to calculate the radius of curvature and tge base position is in the function `line_curvature(line_fit, y_eval)` and `line_base_pos(line_fit, y_eval)` in `AdvancedLaneLines.ipynb`. 

To calculate the real curvature and base position, the line fit was in the unit of Meters, 
 
```python
   left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
   right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
```    
The curvature is the mean value of the curvature calculated from the left and right lane lines:
```python
   y_eval = y_max*ym_per_pix 
   curvature = ((1 + (2*line_fit[0]*y_eval + line_fit[1])**2)**1.5) / np.absolute(2*line_fit[0])
```   
and the off set is the difference of the center position of the image in meters and the average base position from the left and right lanes, and the base position is calculated as:
```python
   y_eval = y_max*ym_per_pix 
   line_fit[0] *(y_eval**2) + line_fit[1] *y_eval + line_fit[2]
```   
#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `draw_line_curv_on_imag()`, where the lane lines was first drawn onto the warped blank image with the function cv2.fillPoly(), and then it was warped back to the original image space using the inverse perspective matrix, and the detection result and the original image was combined together with cv2.addWeighted(), and the curvature and off set was written on the image with cv2.putText().   

Here is an example of my result on a test image from the above steps without tracking the lane lines in continuous frames in a video:

 ![alt text][image_test6_binary]
 ![alt text][image_test6_result]

#### 7. Describe how I did Lane Line Tracking with Information from Continuous Frames.
While the pipeline with the above steps was able to track all the lane lines in the testing iamges in the folder `./test_images`. It lost the lane lines for several frames in the project video file, so the line finding algorithm was improved in the celll "Line Finding With Frame Lane Tracking" of the same notebook  `AdvancedLaneLines.ipynb`, where a class `LaneLine()` was defined to track the lane lines finding from one frame to another frame, and the quality of the lane line finding result from one frame is evaluated by:
 * the difference of the current fit from the last fit
 * the difference between the left offset and right offset 
 * the difference between the left and right radius of curvature
 * the lane width calculated from the left and right base position

If the quality of the line fit result is not passing the quality check, the best fit from previous frames will be used for this frame. 
      
 
---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The final pipeline was defined in the function `pipeline_track()`. 

Here's a [link to my video result] (./project_video_output.mp4)  to the video (./project_video.mp4). The lane lines were detected in all the frames for the project video.

I also tried this pipeline for the challenging videos.  Here are the links to the results of the challenging videos, (./challenge_video_output.mp4) and (./harder_challenge_video_output.mp4). From this video results, it can be seen that the method for lane lines finding in the first frame that is based on the thresholding needs some improvements, and it has difficulty to find lane lines when the right lane lanes is hardly visible under the leaves. 

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
The first issue that I encountered was how to define the thresholds to apply the gradient and the color thresholding, and the pipeline is likely to fail with different lane color illuminations and light conditions. When the lane lines are hardly visible, such as the right lane lines under the shade of tree leaves, the pipeline failed.  To make the pipe line more robust, I think the following aspects could be improved:

 * Enhancement of the gradient thresholding and color thresholding with a weighting factor based on the the light condition and the real lane colors. Those weighting factor could be learned from previous frames.
 * Apply outlider detection with  RANSAC algorithm to get a better fit parameters of the lane lines. By doing this, the outlier points that are not part of the lane lines would be filtered out and thus the fitting results would be improved. 
 * Image Segementation might be helpful to segement out the lane line regions from other regions, so that the further detection could be more focused.
 * The lane line was fitted by a polynomial function. Other curve type might also be helpful to get a better line fitting result.
  
