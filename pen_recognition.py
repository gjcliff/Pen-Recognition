#!/usr/bin/python

# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
# print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

def nothing(x):
    pass

#opencv
cv2.namedWindow('image')

# create trackbars for color change
# cv2.createTrackbar('H1','image',220,360, nothing)
# cv2.createTrackbar('S1','image',50,255, nothing)
# cv2.createTrackbar('V1','image',50,255, nothing)

# cv2.createTrackbar('H2','image',270,360, nothing)
# cv2.createTrackbar('S2','image',255,255, nothing)
# cv2.createTrackbar('V2','image',255,255, nothing)

# Streaming loop
try:
    while True:
        # get current positions of four trackbars
        # h1 = cv2.getTrackbarPos('H1','image') / 2
        # s1 = cv2.getTrackbarPos('S1','image')
        # v1 = cv2.getTrackbarPos('V1','image')

        # h2 = cv2.getTrackbarPos('H2','image') / 2
        # s2 = cv2.getTrackbarPos('S2','image')
        # v2 = cv2.getTrackbarPos('V2','image')

        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # convert the BGR image to HSV
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        # print(s)
        # print(type(s))

        lower_bound = np.array([225/2,97,0])
        upper_bound = np.array([305/2,255,255])

        # lower_bound = np.array([170,50,50])
        # upper_bound = np.array([170,255,255])
        

        # print(f"lower bound: {lower_bound}, upper bound: {upper_bound}")

        # threshold the HSV image to get purple colors within our range
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        masked_depth_image = cv2.bitwise_and(depth_image,depth_image, mask= mask)
        # cv2.imshow('image2', depth_image)

        # bitwise-AND mask and original image
        masked_image = cv2.bitwise_and(color_image, color_image, mask= mask)

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, masked_image)

        # Render images:
        #   depth align to color on left
        #   depth on right
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((bg_removed, depth_colormap))

        # drawing contours
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(masked_image, contours, -1, (0,255,0), 3)

        erosion_size = 1
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion_size + 1, 2 * erosion_size + 1),(erosion_size, erosion_size))

        erosion_dst = cv2.erode(masked_image, element)

        dilation_size = 1
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation_size + 1, 2 * dilation_size + 1), (dilation_size, dilation_size))

        dilation_dst = cv2.dilate(erosion_dst, element)

        # erosion_size = 3
        # element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation_size + 1, 2 * dilation_size + 1), (erosion_size, erosion_size))

        # erosion_dst = cv2.erode(dilation_dst, element)

        # dilation_size = 1
        # element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation_size + 1, 2 * dilation_size + 1), (dilation_size, dilation_size))

        # dilation_dst = cv2.dilate(erosion_dst, element)

        temp = cv2.cvtColor(dilation_dst,cv2.COLOR_BGR2GRAY)

        

        # print(f"temp: {temp}")
        # print(np.shape(contours))
        try:
            max_contour = max(contours, key = cv2.contourArea)
            M = cv2.moments(max_contour)
            
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            cv2.circle(dilation_dst,(cx,cy),10,(255,255,255),-1)
        except:
            print("could not find moments")

        # print(f"moment: {M}")

        

        # cv2.imshow('original', hsv)
        # cv2.imshow('mask', mask)
        cv2.imshow('image', dilation_dst)
        key = cv2.waitKey(1) & 0xFF
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

        

finally:
    pipeline.stop()
    cv2.destroyAllWindows()