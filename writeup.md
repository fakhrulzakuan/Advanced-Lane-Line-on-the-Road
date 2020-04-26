## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

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

[image2]: ./output_images/ROI.jpg "ROI"
[image1]: ./output_images/undistorted_chess.jpg "Undistortion"
[image3]: ./output_images/gradient.jpg "Gradient Combination"
[image4]: ./output_images/s_color.jpg "S-Color Binary"
[image5]: ./output_images/gradient_color.jpg "Gradient and S-Color Combination"
[image6]: ./output_images/verified_src.jpg "Warped Verified"
[image7]: ./output_images/polyline.jpg "Line Fitting"
[image8]: ./output_images/final.jpg "Final Result"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the "./get_pickle.py".
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  The coefficients then I stored in the pickle form (./wide_dist_pickle.p)

![alt text][image1]

### Pipeline (single images)

#### 1. I marked ROI coordinates and draw a polygons lines.

To demonstrate this step, I will describe how I determine the ROI that needs to be warped later on for perspective transform:

I used `cv2.polylines` function to draw the polygons. Then I keep changing the points until I get satisfied result with the ROI I have selected. 

```python
 #To draw src coordinates. 
draw_pts = np.array([[120,720], [1200,720], [700,450], [580,450], [120,720]], np.int32)
img = cv2.polylines(img,[draw_pts],False,(255,0,0), thickness = 10)
```
![alt text][image2]

Once the coordinate points are selected. i will use these same points for perspective transform.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

First, I used a combination of gradient sobel, magnitude and direction thresholds to generate a binary image (thresholding steps at lines 250 through 254 in `P2.py`).  

```python
ksize = 3
gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(20, 100))
grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(20, 100))
mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(30, 200))
dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(0.7, np.pi/2))

#combined all gradient
combined = np.zeros_like(dir_binary)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
```

The result from all the gradient combination is shown below:

![alt text][image3]

Secondly, I converted the original image to HLS and only get the S channel. 

```python
hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
s_channel = hls[:,:,2]
```


With some thresholding for the S channel, I get this result. 

```python
s_thresh_min = 150
s_thresh_max = 255
s_binary = np.zeros_like(s_channel)
s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max) = 1
```

![alt text][image4]

Finally, I have combined both all gradient and s-color binary results. The final binary image looks like this. 

```python
#combined color and gradient
combined_binary = np.zeros_like(combined)
combined_binary[(s_binary == 1) | (combined == 1)] = 1
```

![alt text][image5]

In this final combination the lane lines are clearly visible. 

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `undist_warp()`, which appears in lines 10 through 18 in the file `P2.py`. This function also does the Undistortion first using function `cv2.undistort()`. The `undist_warp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points plus `mtx` and `dist` for undistortion. I chose the hardcode the source and destination points in the following manner:

```python
def undist_warp(img, src, dst, mtx, dist):
    
    img_size = (img.shape[1], img.shape[0])
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(undist, M, img_size)

    return undist, warped, M, Minv
```

```python
offset = 100
src = np.float32([
[120,720], 
[1200,720], 
[700,450], 
[580,450]])

dst = np.float32([
[offset, img.shape[0]], 
[img.shape[1]-offset,img.shape[0]], 
[img.shape[1]-offset, 0], 
[offset, 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 120,720       | 100, 720      | 
| 1200,720      | 1180, 720     |
| 700,450       | 1180, 0       |
| 580,450       | 100, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I search for the lane pixels with this function `find_lane_pixels()`. Once the pixels are identified and grouped between left and right lanes, then we fit a 2nd order polynomial  using `np.polyfit` to draw a fine line to represent the lanes. 
The result is shown in the figure below. 

![alt text][image7]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in function `radius_curvature()` lines 183 through 220 in my code in `P2.py`.

```python
def radius_curvature(binary_warped, left_fit, right_fit):
    
    #Calc again both polynomials using ploty, left_fit and right_fit ###
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Calc new polynomials to x,y but this time in real world space. 
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    # Calculation of R_curve (radius of curvature)
    left_curvature =  ((1 + (2*left_fit_cr[0] *y_eval*ym_per_pix + left_fit_cr[1])**2) **1.5) / np.absolute(2*left_fit_cr[0])
    right_curvature = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # Calculate vehicle center
    # Get the position of line from the most bottom of image because it is closest to the ego vehicle. 
    left_lane_bottom = (left_fit[0]*y_eval)**2 + left_fit[0]*y_eval + left_fit[2]
    right_lane_bottom = (right_fit[0]*y_eval)**2 + right_fit[0]*y_eval + right_fit[2]
    
    # To get the middle point betwwen two lines. Just add both lines then divide it in two. 
    lane_center = (left_lane_bottom + right_lane_bottom)/2.

    #Assuming that the camera is placed at the center of the ego vehicle. Therefore the center position of the car equal to the center of the image which is at X-axis pixel 640.
    center_image = 640

    #Convert the pixel to world space (meters) using the xm per pix ratio. 
    center = (lane_center - center_image)*xm_per_pix 
    
    return left_curvature, right_curvature, center
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 222 through 242 in my code in `P2.py` in the function `map_lane()`.  Here is an example of my result on a test image together with the road curvature for left and right lane lines and position of the ego vehicle on the lane that i did in these code:

```python
#Add text on final image
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,100)
    fontScale              = 2
    fontColor              = (255,255,255)
    lineType               = 3

    cv2.putText(final_image, "Left Radius :" + left_curv_str + " m", 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,200)
    fontScale              = 2
    fontColor              = (255,255,255)
    lineType               = 3

    cv2.putText(final_image, "Right Radius : " + right_curv_str + " m", 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (400,700)
    fontScale              = 2
    fontColor              = (0,0,0)
    lineType               = 4

    cv2.putText(final_image, "Center : " + center_str + " m", 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)
```


![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_final.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One issue I stumbled is to visualize the windows box. 
Pipeline will fail when the road lines is not too visible or missing because in my pipeline I did not keep the previous lines. 
To make it roburst, I need to keep the previous lines and only update when there is only minor changes between previous line and the current line like its gradient, curvature or the position.
